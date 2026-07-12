"""
Run one query through the graph.

    python -m nhpc_qa.retrieval.graph.run "electricity dues owed by J&K"
    python -m nhpc_qa.retrieval.graph.run --json "जलविद्युत परियोजना की जानकारी"

Builds the providers ONCE (embedder, reranker, entity vocabulary), then executes the
LangGraph pipeline. Used by the CLI, the tests, and (later) the API.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time

from nhpc_qa.core.trace.tracer import new_run_id
from nhpc_qa.core.db.session import connect
from nhpc_qa.core.providers.embeddings import get_embedder
from nhpc_qa.config import Settings, load_dotenv
from nhpc_qa.retrieval.graph.build import build_graph
from nhpc_qa.core.trace.query_tracer import QueryTracer
from nhpc_qa.retrieval.search import entity
from nhpc_qa.core.providers.rerank import get_reranker


@contextlib.contextmanager
def query_engine(cfg=None):
    """
    Construct everything the graph needs, once. Yields a `run(query, ...)` callable.

    The providers are built here -- NOT inside LangGraph -- so the graph stays a pure
    conductor over the existing interfaces.
    """
    cfg = cfg or Settings()
    errs = cfg.validate_all()
    if errs:
        raise SystemExit("CONFIG ERROR:\n  " + "\n  ".join(errs))

    with connect(cfg) as conn:
        deps = {
            "cfg": cfg,
            "conn": conn,
            "embedder": get_embedder(cfg),
            "reranker": get_reranker(cfg) if cfg.rerank_enabled else None,
            "entity_vocab": entity.load_vocabulary(conn),
            "llm": None,          # built lazily only if generation is enabled
            # Langfuse mirror: a no-op unless LANGFUSE_ENABLED (SDK never imported when off)
            "tracer": QueryTracer(cfg),
        }
        graph = build_graph(deps)

        def run(query, user_id="cli", user_role="officer",
                house=None, session=None, nhpc_only=False, run_id=None):
            t0 = time.time()
            state = {
                "run_id": run_id or new_run_id(),
                "query": query,
                "user_id": user_id,
                "user_role": user_role,
                "house": house,
                "session": session,
                "nhpc_only": nhpc_only,
                "widened": False,
                "timings_ms": {},
                "errors": [],
            }
            deps["tracer"].start(state, cfg)
            out = graph.invoke(state)
            deps["tracer"].finish()
            out.setdefault("timings_ms", {})["total"] = int((time.time() - t0) * 1000)
            return out

        yield run, deps


def _print_human(out):
    stats = out.get("fuse_stats") or {}
    print(f"\nrun_id   : {out['run_id']}")
    print(f"query    : {out['query']}")
    print(f"language : {out.get('language')}   (PROCESSING ONLY — never filters retrieval)")
    print(f"entities : {out.get('entities') or '(none — entity retriever ineligible)'}")
    print(f"widened  : {out.get('widened')}"
          + (f"   reason: {out.get('widen_reason')}" if out.get("widen_reason") else ""))
    print(f"fuse     : candidates={stats.get('n_candidates')} "
          f"top={stats.get('top_score')} gap={stats.get('score_gap')} "
          f"eligible={stats.get('n_eligible')} fired={stats.get('n_fired')}")
    if out.get("rerank_failed"):
        print("rerank   : FAILED — degraded to RRF order (results still returned)")
    print(f"timings  : {out.get('timings_ms')}")

    print("\n" + "=" * 78)
    for r in out.get("results", []):
        s = r["signals"]
        print(f"[{r['rank']}] {r['doc_key']}   ({r['house']}, {r['session']})")
        print(f"    Q ({r['part_label']}): {r['question_text'][:88]}")
        ans = (r["answer_text"] or "")[:88].replace("\n", " ")
        print(f"    A [{r['answer_type']}]: {ans}")
        annex = ", ".join(f"{a['ref_label']}({a['status']})" for a in r["annexures"])
        print(f"    files: reply={'yes' if r['reply_file']['available'] else 'no'}"
              + (f" | annexures: {annex}" if annex else ""))
        print(f"    signals(heuristic): rrf={s['rrf_score']} rerank={s['rerank_logit']} "
              f"move={s['rerank_movement']} retrievers={s['retrievers']} "
              f"agreement={s['agreement']}")
        print()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run one query through the Phase-4 graph")
    ap.add_argument("query")
    ap.add_argument("--json", action="store_true", help="emit the raw payload")
    ap.add_argument("--house", default=None)
    ap.add_argument("--session", default=None)
    ap.add_argument("--nhpc-only", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s")

    load_dotenv()
    with query_engine() as (run, _deps):
        out = run(args.query, house=args.house, session=args.session,
                  nhpc_only=args.nhpc_only)

    if args.json:
        print(json.dumps({k: v for k, v in out.items()
                          if k not in ("query_vec", "retrieved", "fused", "reranked")},
                         ensure_ascii=False, indent=2, default=str))
    else:
        _print_human(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
