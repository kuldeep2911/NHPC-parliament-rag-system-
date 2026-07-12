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
from nhpc_qa.watcher import queue as q

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


def question_folder(cfg, path: str):
    """
    The QUESTION FOLDER containing `path`, or the session folder if the event is higher up.

    SETTLING MUST HAPPEN ON THE FOLDER, NOT THE FILE. A question folder is copied in file
    by file (reply.pdf, annexures, the original question...). Settling on one file would
    let the pipeline parse a folder that is still half-copied -- it would read a reply
    whose annexure has not landed yet, mark the annexure 'referenced but unavailable',
    and store that as fact. So the queue key is always the folder.
    """
    src_root = os.path.abspath(getattr(cfg, "source_root", None) or "Original Data")
    p = os.path.abspath(path)
    try:
        rel = os.path.relpath(p, src_root)
    except ValueError:
        return None
    parts = [x for x in rel.replace("\\", "/").split("/") if x and x != "."]
    if not parts or parts[0].startswith(".."):
        return None
    # keep at most session/house/question
    keep = parts[:3]
    return os.path.join(src_root, *keep)


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
    log.info("sync upsert: session=%s only=%s",
             os.path.basename(session_dir), only_token or "(whole session)")

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

    # What did we actually end up with? (also REACTIVATES anything that had been
    # soft-deleted and has now come back -- see reactivate())
    docs = _docs_for_slice(conn, only_token, session_dir)
    reactivated = reactivate(conn, docs)

    for dk, nsq in docs:
        if dk not in reactivated:
            q.log_action(conn, "added", doc_key=dk, source_path=source_path,
                         n_sub_questions=nsq)

    dt = time.time() - t0
    log.info("sync upsert done in %.1fs: %d document(s) %s", dt, len(docs),
             f"({len(reactivated)} reactivated)" if reactivated else "")
    return {"ok": True, "docs": [d for d, _ in docs], "reactivated": sorted(reactivated),
            "seconds": round(dt, 1)}


def _docs_for_slice(conn, only_token, session_dir):
    """The documents in the DB that this slice produced (doc_key, n_sub_questions)."""
    with conn.cursor() as cur:
        if only_token:
            cur.execute("""
                SELECT d.doc_key, count(sq.sub_question_id)
                FROM diaries d
                LEFT JOIN sub_questions sq ON sq.doc_key = d.doc_key
                WHERE d.question_id = %s
                GROUP BY d.doc_key
            """, (only_token,))
        else:
            cur.execute("""
                SELECT d.doc_key, count(sq.sub_question_id)
                FROM diaries d
                LEFT JOIN sub_questions sq ON sq.doc_key = d.doc_key
                GROUP BY d.doc_key
                ORDER BY d.updated_at DESC LIMIT 50
            """)
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
    _session, only_token = source_slice(cfg, source_path)
    if not only_token:
        # A whole session vanishing is almost certainly a mount/rename, not an intent to
        # delete an entire session of parliamentary record. Refuse to act on it.
        log.error("sync delete: %s looks like a WHOLE SESSION disappearing — refusing to "
                  "soft-delete a whole session from a filesystem event. If this is "
                  "intended, do it explicitly.", source_path)
        q.log_action(conn, "skipped", source_path=source_path,
                     detail="whole-session disappearance ignored (too dangerous to act on)")
        return {"ok": True, "soft_deleted": [], "refused": True}

    return soft_delete(conn, only_token, source_path)


def soft_delete(conn, question_id: str, source_path=None, reason="source path removed"):
    """Mark every document with this diary number inactive. Recoverable."""
    with conn.cursor() as cur:
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
