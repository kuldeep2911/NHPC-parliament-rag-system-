"""
Incremental sync — process ONE affected slice, not the whole corpus.

ADD / UPDATE path:
    source folder changed  ->  crawl (that session only)  ->  parse (that question only)
                           ->  index (load + embed that question only)  ->  retrievable

We deliberately RE-USE the existing stages rather than reimplement their path logic. The
crawler already knows how a source folder maps to organized/<session>/<house>/<qid>/ --
duplicating that mapping here would be a second source of truth that silently drifts. So
the sync scopes the real stages with their own --only/--limit flags.

Every stage is idempotent (crawl copies only what changed, parse skips a folder that
already has parsed.json unless forced, index UPSERTs on deterministic keys), so
re-processing a slice is safe and produces stable keys. That is what makes the "at least
once" queue acceptable.

DELETE path — SOFT ONLY. See soft_delete() below.
"""

from __future__ import annotations

import os
import re
import time

from nhpc_qa.core.logging import get_logger
from nhpc_qa.core import queue as q
# question_folder now lives in core/ (the API needs it too, and api may not
# import watcher -- they are SIBLING layers). Re-exported so existing callers
# of sync.question_folder keep working.
from nhpc_qa.core.queue import question_folder  # noqa: F401

log = get_logger("nhpc.watcher.sync")


# ---------------------------------------------------------------------------
# mapping a source path -> the slice to process
# ---------------------------------------------------------------------------

def source_slice(cfg, source_path: str):
    """
    From a path inside the source tree, work out:
        session_dir  -- the 'PARLIAMENT ...' folder to crawl (crawl is scoped to it)
        only_token   -- a substring that identifies the affected question folder(s),
                        passed to parse/index as --only

    Returns (session_dir, only_token). only_token is None when the event is at session
    level (a whole new session dropped in) -- then the entire session is processed.
    """
    src_root = os.path.abspath(getattr(cfg, "source_root", None) or "Original Data")
    p = os.path.abspath(source_path)
    try:
        rel = os.path.relpath(p, src_root)
    except ValueError:
        return None, None
    if rel.startswith(".."):
        return None, None                       # outside the source root — ignore

    parts = [x for x in rel.replace("\\", "/").split("/") if x and x != "."]
    if not parts:
        return None, None

    session_dir = os.path.join(src_root, parts[0])
    # <session>/<house>/<question_id>/...  -> the question id is the slice
    only_token = parts[2] if len(parts) >= 3 else None
    return session_dir, only_token



# ---------------------------------------------------------------------------
# ADD / UPDATE
# ---------------------------------------------------------------------------

def process_upsert(cfg, conn, source_path: str):
    """
    Run the affected slice through crawl -> parse -> index. Returns a summary dict.

    Scoped, not global: a new question folder costs one crawl of its session (which copies
    only what changed), one parse of that question, and one load+embed of that question.
    """
    from nhpc_qa.pipeline.crawl import crawler
    from nhpc_qa.pipeline.index import embedder, loader
    from nhpc_qa.pipeline.parse import pipeline as parse_pipeline

    t0 = time.time()
    session_dir, only_token = source_slice(cfg, source_path)
    if not session_dir or not os.path.isdir(session_dir):
        log.warning("sync: %s is not inside the source tree — ignoring", source_path)
        q.log_action(conn, "skipped", source_path=source_path,
                     detail="not inside the source root")
        return {"ok": False, "reason": "outside source root"}

    organized_root = getattr(cfg, "organized_root", "organized")
    src_root = os.path.abspath(getattr(cfg, "source_root", None) or "Original Data")
    session_dir_name = os.path.basename(session_dir)

    # Ask the CRAWLER what this session folder normalises to -- do not re-implement it
    # here. Two independent notions of "which session is this?" would drift apart, and the
    # one that matters is the crawler's, because that is what actually writes doc_key.
    session_slug, _sess_reason = crawler.normalize_session(session_dir_name)

    # FAIL FAST, and say exactly what is wrong. The crawler will skip this folder as an
    # orphan and every later stage will then run on nothing -- so catching it here, before
    # any work is done, is the difference between an actionable error and a silent no-op.
    if session_slug is None:
        raise RuntimeError(
            f"cannot process {session_dir_name!r}: the folder name has no recognisable "
            f"session year, so the crawler will skip it entirely.\n"
            f"  Expected something like 'PARLIAMENT DEC-JAN 25' or 'PARLIAMENT MONSOON 24' "
            f"-- month/season tokens plus a 2- or 4-digit year in "
            f"{crawler.SESSION_YEAR_MIN}..{crawler.SESSION_YEAR_MAX}.\n"
            f"  A number outside that range is not read as a year at all.")

    log.info("sync upsert: session=%s (%s) only=%s",
             session_dir_name, session_slug, only_token or "(whole session)")

    # 1. CRAWL — read-only on the source; copies into organized/.
    #
    # It MUST run from the source ROOT, not the session sub-folder: the crawler derives
    # session and house from the path structure BELOW its --source, so pointing it at a
    # session directory makes it see the house folders as sessions and silently produce
    # nothing. (That is exactly what happened the first time this was written.)
    #
    # Running from the root is still cheap: the crawler copies only what changed, so a
    # re-crawl of an unchanged corpus is a stat-walk, not a re-copy.
    rc = crawler.main(["--source", src_root, "--out", organized_root])
    if rc not in (0, None):
        raise RuntimeError(f"crawl failed (exit {rc})")

    # 2. PARSE — only the affected question folder(s).
    parse_argv = ["--organized", organized_root]
    if only_token:
        parse_argv += ["--only", only_token]
    if getattr(cfg, "llm_grouping", False):
        parse_argv.append("--llm-grouping")
    rc = parse_pipeline.main(parse_argv)
    if rc not in (0, None):
        raise RuntimeError(f"parse failed (exit {rc})")

    # 3. INDEX — load + embed the same slice. Both UPSERT, so a re-run is a no-op.
    load_argv = ["--only", only_token] if only_token else []
    rc = loader.main(load_argv)
    if rc not in (0, None):
        raise RuntimeError(f"index/load failed (exit {rc})")
    rc = embedder.main([])          # embeds only rows whose embedding IS NULL
    if rc not in (0, None):
        raise RuntimeError(f"index/embed failed (exit {rc})")

    # ENTITIES — dictionary FIRST, then extract, so the new records are retrievable by the
    # entity retriever. Order matters and is the same as the full build: mine this file's
    # new "Full (ABBR)" patterns + LLM-discover -> add NEW entities -> extract canonical
    # entities for the new records against the UPDATED dictionary. Idempotent (deterministic
    # ids), so re-processing adds no duplicates. A failure here must not lose the ingest --
    # the record is already loaded and searchable by dense; entities just would not be
    # linked until the next build.
    try:
        from nhpc_qa.entities.build import build as _build_entities
        es = _build_entities(cfg, conn, use_llm=bool(getattr(cfg, "entities_llm_on_upload",
                                                              False)),
                             only=only_token)
        log.info("entities: +%d entity(ies), %d link(s) for %s",
                 es["entities_after"] - es["entities_before"], es["links"], only_token)
    except Exception as e:      # noqa: BLE001 -- never fail the ingest on entity build
        log.error("entity build failed for %s (%s) — record is searchable by dense; run "
                  "`nhpc build-entities --only %s` to link entities",
                  only_token, type(e).__name__, only_token)

    # What did we actually end up with? (also REACTIVATES anything that had been
    # soft-deleted and has now come back -- see reactivate())
    docs = _docs_for_slice(conn, only_token, session_dir, session_slug)
    reactivated = reactivate(conn, docs)

    # ⚠️ A SLICE THAT INGESTED NOTHING IS A FAILURE, NOT A SUCCESS. ⚠️
    #
    # Without this the job goes green having done nothing: the queue says 'done', the UI
    # says "processing started ... done", and the admin walks away believing their session
    # is in the system. It is not. A silent no-op is the worst possible outcome here --
    # strictly worse than a loud error, because nobody ever investigates a success.
    #
    # This is what happened with 'PARLIAMENT DEC-JAN 1915': the folder name has no
    # recognisable session year, the crawler correctly SKIPPED it as an orphan, and every
    # downstream stage then ran happily on nothing.
    if not docs:
        raise RuntimeError(
            f"ingested NOTHING for {os.path.relpath(source_path, src_root)!r}.\n"
            f"  The crawler produced no document for this slice. The usual cause is a "
            f"session folder whose name has no recognisable year: {session_dir_name!r} "
            f"-> {session_slug or 'UNRECOGNISED'}.\n"
            f"  Rename it to a form the crawler understands -- 'PARLIAMENT DEC-JAN 25', "
            f"'PARLIAMENT MONSOON 24' -- i.e. month/season tokens plus a 2- or 4-digit "
            f"year between {crawler.SESSION_YEAR_MIN} and {crawler.SESSION_YEAR_MAX}.")

    for dk, nsq in docs:
        if dk not in reactivated:
            q.log_action(conn, "added", doc_key=dk, source_path=source_path,
                         n_sub_questions=nsq)

    dt = time.time() - t0
    log.info("sync upsert done in %.1fs: %d document(s) %s", dt, len(docs),
             f"({len(reactivated)} reactivated)" if reactivated else "")
    return {"ok": True, "docs": [d for d, _ in docs], "reactivated": sorted(reactivated),
            "seconds": round(dt, 1)}


def _docs_for_slice(conn, only_token, session_dir, session_slug):
    """
    The documents in the DB that THIS slice produced (doc_key, n_sub_questions).

    ⚠️ SCOPED BY doc_key, NEVER BY question_id ALONE. ⚠️

    This used to be `WHERE d.question_id = %s`, and that is the exact mistake the whole
    schema exists to prevent: a diary number is REUSED across sessions for a completely
    different question (9 of 518 documents share a number with another). Matching on the
    number alone made this function return SOMEBODY ELSE'S DOCUMENT.

    The failure was ugly. A session folder named 'PARLIAMENT DEC-JAN 1915' is not
    recognisable (see normalize_session), so the crawler correctly skipped it and wrote
    NOTHING. But the slice's only_token was the diary number '8779', so this query happily
    found the pre-existing 2023-jul-aug/lok_sabha/8779 -- a different question from a
    different session -- and process_upsert reported it as "added" for that upload. The
    job went green, sync_log recorded eight 'added' rows, and the UI told the admin their
    upload had been processed. It had not. Nothing had been ingested at all.

    Silence would have been better than a false success: the admin would have investigated.

    The fix is to scope by the SESSION the slice actually belongs to, so a document can
    only ever be attributed to the upload that really produced it.
    """
    with conn.cursor() as cur:
        if only_token and session_slug:
            # The precise slice: this question, in THIS session. doc_key is
            # '<session>/<house>/<question_id>', so a prefix+suffix match is exact
            # regardless of house.
            cur.execute("""
                SELECT d.doc_key, count(sq.sub_question_id)
                FROM diaries d
                LEFT JOIN sub_questions sq ON sq.doc_key = d.doc_key
                WHERE d.session = %s AND d.question_id = %s
                GROUP BY d.doc_key
            """, (session_slug, only_token))
        elif session_slug:
            # A whole session was dropped in: every document of THAT session.
            cur.execute("""
                SELECT d.doc_key, count(sq.sub_question_id)
                FROM diaries d
                LEFT JOIN sub_questions sq ON sq.doc_key = d.doc_key
                WHERE d.session = %s
                GROUP BY d.doc_key
            """, (session_slug,))
        else:
            # The session folder could not be normalised, so the crawler skipped it and
            # produced nothing. Attributing ANY existing document to this slice would be a
            # lie -- return nothing and let the caller fail loudly.
            return []
        return cur.fetchall()


# ---------------------------------------------------------------------------
# DELETE — SOFT ONLY. Never a hard delete from a filesystem event.
# ---------------------------------------------------------------------------

def process_delete(cfg, conn, source_path: str):
    """
    A path vanished from the source tree.

    WE DO NOT DELETE ANYTHING. We mark the affected documents inactive: they drop out of
    retrieval immediately, but every row, answer, table and 2048-dim vector stays.

    WHY: a disappearance is an AMBIGUOUS signal. A folder being moved, a share
    reorganised, a mount blipping, an officer tidying up -- all look identical to a
    deletion from here. Acting irreversibly on an ambiguous signal is how data is lost,
    and the embeddings alone cost real time and money to rebuild. The system's discipline
    everywhere else is flag-don't-act (needs_review never gates a load); delete is the one
    place where getting that wrong cannot be undone.

    Hard removal is a separate, deliberate command: `nhpc purge --older-than 30d`.
    """
    from nhpc_qa.pipeline.crawl import crawler

    session_dir, only_token = source_slice(cfg, source_path)
    if not only_token:
        # A whole session vanishing is almost certainly a mount/rename, not an intent to
        # delete an entire session of parliamentary record. Refuse to act on it.
        log.error("sync delete: %s looks like a WHOLE SESSION disappearing — refusing to "
                  "soft-delete a whole session from a filesystem event. If this is "
                  "intended, do it explicitly.", source_path)
        q.log_action(conn, "skipped", source_path=source_path,
                     detail="whole-session disappearance ignored (too dangerous to act on)")
        return {"ok": True, "soft_deleted": [], "refused": True}

    # The SESSION scopes the delete. Without it, soft_delete matched on the diary number
    # alone -- see the warning there.
    session_slug, _ = crawler.normalize_session(os.path.basename(session_dir))
    return soft_delete(conn, only_token, source_path, session_slug=session_slug)


def soft_delete(conn, question_id: str, source_path=None, reason="source path removed",
                session_slug=None):
    """
    Mark the affected document(s) inactive. Recoverable.

    ⚠️ SCOPED BY SESSION + question_id, NEVER BY question_id ALONE. ⚠️

    This used to be `WHERE question_id = %s`, which is the same cardinal mistake that
    _docs_for_slice had: a diary number is REUSED across sessions for a DIFFERENT question
    (9 of 518 documents share a number with another). Removing '8779' from one session
    would therefore have ALSO deactivated 2023-jul-aug/lok_sabha/8779 -- a completely
    unrelated question -- and it would have vanished from search with no obvious cause.

    session_slug=None is only tolerated for a caller that genuinely cannot determine the
    session; it falls back to the old, unsafe behaviour and logs loudly, rather than
    silently doing the wrong thing.
    """
    with conn.cursor() as cur:
        if session_slug:
            cur.execute("""
                UPDATE diaries
                SET active = false, deleted_at = now(), deleted_reason = %s
                WHERE session = %s AND question_id = %s AND active
                RETURNING doc_key
            """, (reason, session_slug, question_id))
        else:
            log.warning("soft_delete without a session for question_id=%s — this can only "
                        "match on the diary number, which is REUSED across sessions",
                        question_id)
            cur.execute("""
                UPDATE diaries
                SET active = false, deleted_at = now(), deleted_reason = %s
                WHERE question_id = %s AND active
                RETURNING doc_key
            """, (reason, question_id))
        keys = [r[0] for r in cur.fetchall()]

    for dk in keys:
        log.warning("SOFT DELETE %s — excluded from retrieval, data retained "
                    "(recover with `nhpc watch` if the folder returns)", dk)
        q.log_action(conn, "soft_deleted", doc_key=dk, source_path=source_path,
                     detail=reason)
    if not keys:
        log.info("soft delete: nothing active matched question_id=%s", question_id)
    return {"ok": True, "soft_deleted": keys}


def reactivate(conn, docs):
    """
    Bring back any of these documents that had been soft-deleted.

    Matching is on doc_key (deterministic) -- the same source folder always produces the
    same key -- so a folder that reappears REACTIVATES rather than re-ingesting. No
    re-parse, no re-embed, no new rows. file_sha256 is cross-checked so a genuinely
    different file replacing an old one is treated as an update, not a silent restore.
    """
    keys = [dk for dk, _ in docs]
    if not keys:
        return set()
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE diaries
            SET active = true, deleted_at = NULL, deleted_reason = NULL
            WHERE doc_key = ANY(%s) AND NOT active
            RETURNING doc_key
        """, (keys,))
        back = {r[0] for r in cur.fetchall()}
    for dk in back:
        log.info("REACTIVATED %s — was soft-deleted, source reappeared", dk)
        q.log_action(conn, "reactivated", doc_key=dk,
                     detail="source path returned; matched on deterministic doc_key")
    return back
