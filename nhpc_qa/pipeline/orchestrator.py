"""
Stage orchestrator — runs crawl -> parse -> index in order, with stage selection.

    from nhpc_qa.pipeline.orchestrator import run_stages
    run_stages(cfg, stages=["crawl", "parse", "index"], limit=5, dry_run=False)

DESIGN: this module OWNS NO LOGIC. Each stage is the existing, working entry point --
crawler.main(), parse.pipeline.main(), index.loader.main(), index.embedder.main() -- driven
through its own argv. That is deliberate: the restructure was not allowed to change phase
behaviour, and the surest way to honour that is to call the same functions the same way.

STAGES
    crawl  -> reads the ORIGINAL source tree, copies into organized/   (read-only on source)
    parse  -> organized/*/answer_latest.* -> parsed.json
    index  -> parsed.json -> Postgres rows + pgvector embeddings
              (this is TWO steps -- load then embed -- because a load without embeddings
               leaves rows that are stored but unsearchable.)

RESUMABILITY comes from the stages themselves, not from bookkeeping here:
  * crawl re-copies only what changed
  * parse SKIPS a question folder that already has a parsed.json (unless --force)
  * index UPSERTs on deterministic primary keys, and the embedder only touches rows whose
    embedding IS NULL (unless --force)
So a crash mid-run is fixed by re-running the same command: each stage picks up where it
left off, and re-processing anything already done is a no-op rather than a duplicate.

STAGE ISOLATION: each stage is independently runnable (`--stages index`) and idempotent, so
partial runs compose. The output of one is the input of the next, on disk / in the DB --
there is no hidden in-memory handoff that a partial run could miss.
"""

from __future__ import annotations

import time

from nhpc_qa.core.logging import get_logger

log = get_logger("nhpc.pipeline")

# The canonical order. `--from parse` means "parse and everything after it".
STAGES = ["crawl", "parse", "index"]


class StageError(RuntimeError):
    pass


def resolve_stages(stages=None, from_stage=None):
    """
    Work out which stages to run, always in canonical order.

        resolve_stages()                      -> [crawl, parse, index]   (full run)
        resolve_stages(stages=["index"])      -> [index]
        resolve_stages(stages=["index","parse"]) -> [parse, index]  (reordered!)
        resolve_stages(from_stage="parse")    -> [parse, index]
    """
    if from_stage:
        if from_stage not in STAGES:
            raise StageError(f"unknown stage {from_stage!r} (have: {', '.join(STAGES)})")
        return STAGES[STAGES.index(from_stage):]
    if not stages:
        return list(STAGES)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise StageError(f"unknown stage(s) {unknown} (have: {', '.join(STAGES)})")
    # Always canonical order, whatever order the user listed them in: running index
    # before parse would index stale parsed.json files.
    return [s for s in STAGES if s in set(stages)]


# ---------------------------------------------------------------------------
# stage runners — each shells into the existing entry point via its own argv
# ---------------------------------------------------------------------------

def _run_crawl(cfg, *, source, organized_root, dry_run, limit, force, only=None):
    from nhpc_qa.pipeline.crawl import crawler

    argv = ["--source", source, "--out", organized_root]
    if dry_run:
        argv.append("--dry-run")
    if limit:
        argv += ["--limit", str(limit)]
    # NOTE: the crawler is READ-ONLY on the source tree -- it copies out, never writes in.
    return crawler.main(argv)


def _run_parse(cfg, *, source, organized_root, dry_run, limit, force, only=None):
    from nhpc_qa.pipeline.parse import pipeline

    argv = ["--organized", organized_root]
    if dry_run:
        argv.append("--dry-run")
    if limit:
        argv += ["--limit", str(limit)]
    if force:
        argv.append("--force")
    if only:
        argv += ["--only", only]
    # Keep the extraction path the officer-grade one: the LLM locates questions AND
    # answers by line span. Turning this off silently reverts to the old rule-based
    # splitter, so it is passed explicitly rather than left to a default.
    if getattr(cfg, "llm_grouping", False):
        argv.append("--llm-grouping")
    return pipeline.main(argv)


def _run_index(cfg, *, source, organized_root, dry_run, limit, force, only=None):
    """Index = LOAD then EMBED. A load without embeddings leaves rows that are stored but
    unsearchable, so the two always run together."""
    from nhpc_qa.pipeline.index import embedder, loader

    load_argv = []
    if dry_run:
        load_argv.append("--dry-run")
    if limit:
        load_argv += ["--limit", str(limit)]
    if force:
        load_argv.append("--force")
    if only:
        load_argv += ["--only", only]
    rc = loader.main(load_argv)
    if rc not in (0, None):
        raise StageError(f"index/load failed (exit {rc})")

    if dry_run:
        log.info("index/embed: skipped (dry run)")
        return 0

    embed_argv = []
    if limit:
        embed_argv += ["--limit", str(limit)]
    if force:
        embed_argv.append("--force")
    return embedder.main(embed_argv)


_RUNNERS = {"crawl": _run_crawl, "parse": _run_parse, "index": _run_index}


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def run_stages(cfg, stages=None, from_stage=None, *, source=None, dry_run=False,
               limit=0, force=False, only=None):
    """
    Run the selected stages in canonical order. Returns a per-stage result dict.

    A stage that fails raises -- we do NOT press on to the next one, because a failed
    parse would otherwise be followed by an index run that happily indexes yesterday's
    parsed.json and reports success.
    """
    plan = resolve_stages(stages, from_stage)
    organized_root = getattr(cfg, "organized_root", "organized")
    source = source or getattr(cfg, "source_root", None) or "Original Data"

    log.info("pipeline: stages=%s organized=%s%s%s",
             " -> ".join(plan), organized_root,
             " [DRY RUN]" if dry_run else "",
             f" limit={limit}" if limit else "")

    results = {}
    for stage in plan:
        t0 = time.time()
        log.info("stage %s: start", stage)
        rc = _RUNNERS[stage](cfg, source=source, organized_root=organized_root,
                             dry_run=dry_run, limit=limit, force=force, only=only)
        dt = time.time() - t0
        ok = rc in (0, None)
        results[stage] = {"ok": ok, "exit_code": rc, "seconds": round(dt, 1)}
        log.info("stage %s: %s in %.1fs", stage, "ok" if ok else f"FAILED (exit {rc})", dt)
        if not ok:
            raise StageError(
                f"stage '{stage}' failed (exit {rc}); stopping so later stages do not "
                f"run on stale inputs")
    return results
