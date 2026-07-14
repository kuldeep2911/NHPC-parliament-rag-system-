"""
Admin-only source-tree browser + delete.

    GET    /admin/tree?path=<rel>       list one level of the source tree
    DELETE /admin/tree?path=<rel>       HARD delete: files + DB rows + vectors

WHY A TREE BROWSER EXISTS AT ALL. The upload endpoint used to write wherever the browser's
webkitRelativePath said, which meant:

  * "Choose files..." (no relative path) put a bare 'reply.pdf' at the SOURCE ROOT -- a
    file belonging to no session, no house and no question. The crawler ignores it, so it
    could never be ingested. It was a silent black hole.
  * "Choose folder..." always created a TOP-LEVEL session. There was no way to add one
    question folder to an existing house.

Now the admin picks a destination NODE in the real tree and uploads into it, so a file can
only ever land somewhere the pipeline can actually see.

⚠️ DELETE IS HARD AND IRREVERSIBLE HERE -- AND THAT IS DELIBERATE. ⚠️

That is NOT a contradiction of the watcher's soft-delete rule; it is the other half of it.
The watcher soft-deletes because a path VANISHING is an AMBIGUOUS signal: a moved folder, a
reorganised share, a blipping mount and an intentional deletion all look identical from a
filesystem event. Acting irreversibly on an ambiguous signal is how data is lost.

An admin clicking "Delete" in this UI is not ambiguous. It is an authenticated,
audited, confirmed statement of intent. So it does what it says.

The two paths must not be confused, so they are separate code:
    watcher/sync.process_delete()  -- ambiguous signal  -> SOFT delete (recoverable)
    this module                    -- deliberate action -> HARD delete (audited, no undo)
"""

from __future__ import annotations

import os
import shutil

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from nhpc_qa.core.logging import get_logger
from nhpc_qa.api.security import deps, upload_guard as guard, users
from nhpc_qa.api.security.upload_guard import Rejected
from nhpc_qa.core import queue as q
from nhpc_qa.pipeline.crawl.crawler import normalize_session

log = get_logger("nhpc.tree")

router = APIRouter()


def _st(request: Request):
    return request.app.state.nhpc


def _safe_target(cfg, rel: str) -> str:
    """
    Resolve a client-supplied relative path inside the source root, or 404/400.

    Every path from the client goes through the SAME realpath jail the upload uses. A tree
    browser is a directory-traversal primitive by nature -- 'list me ../../../../etc' -- so
    it gets exactly the same control, not a weaker one.
    """
    src_root = os.path.abspath(cfg.source_root)
    rel = (rel or "").strip().strip("/").strip("\\")
    if not rel:
        return src_root
    try:
        return guard.jail(src_root, guard.sanitize_relpath(rel))
    except Rejected as e:
        raise HTTPException(400, f"invalid path: {e}")


# ---------------------------------------------------------------------------
# GET /admin/tree — one level at a time
# ---------------------------------------------------------------------------
@router.get("/admin/tree")
def list_tree(request: Request, path: str = Query(default=""),
              admin=Depends(deps.require_admin)):
    """
    List ONE level. Lazily, not the whole tree: the corpus is 5,000+ files and walking it
    on every click would be slow and pointless.

    `depth` tells the UI what the admin is looking at, so it can label the level
    (session / house / question) and decide what may be uploaded or deleted there.
    """
    cfg = _st(request)["cfg"]
    src_root = os.path.abspath(cfg.source_root)
    target = _safe_target(cfg, path)

    if not os.path.isdir(target):
        raise HTTPException(404, "no such folder")

    rel = "" if target == src_root else os.path.relpath(target, src_root).replace("\\", "/")
    depth = 0 if not rel else len(rel.split("/"))

    dirs, files = [], []
    with os.scandir(target) as it:
        for e in it:
            if e.name.startswith(".") or e.name.startswith("~$"):
                continue                        # staging dir, OS noise
            if e.is_dir():
                # At the top level, say whether the crawler can even read this session
                # name -- so a folder that will never ingest is visible AS such, rather
                # than looking healthy and quietly doing nothing.
                slug = None
                if depth == 0:
                    slug, _ = normalize_session(e.name)
                dirs.append({"name": e.name, "type": "dir",
                             "path": f"{rel}/{e.name}" if rel else e.name,
                             "session_slug": slug,
                             "ingestible": (slug is not None) if depth == 0 else None})
            else:
                try:
                    size = e.stat().st_size
                except OSError:
                    size = None
                files.append({"name": e.name, "type": "file",
                              "path": f"{rel}/{e.name}" if rel else e.name,
                              "size": size})

    # level names follow the crawler's own expectation: <session>/<house>/<question>/...
    LEVEL = {0: "root", 1: "session", 2: "house", 3: "question"}
    return {
        "path": rel,
        "depth": depth,
        "level": LEVEL.get(depth, "inside question"),
        "parent": None if not rel else "/".join(rel.split("/")[:-1]),
        "dirs": sorted(dirs, key=lambda d: d["name"].lower()),
        "files": sorted(files, key=lambda f: f["name"].lower()),
    }


# ---------------------------------------------------------------------------
# DELETE /admin/tree — HARD delete
# ---------------------------------------------------------------------------
@router.delete("/admin/tree")
def delete_path(request: Request, path: str = Query(...),
                confirm: str = Query(default=""),
                admin=Depends(deps.require_admin)):
    """
    Permanently remove a file, a question folder, a house, or a whole session -- from the
    SOURCE TREE, from organized/, and from the DATABASE (rows, answers, tables, annexures
    and the 2048-dim vectors).

    The affected documents stop appearing in search IMMEDIATELY, because their rows are
    gone -- not merely flagged.

    `confirm` must echo the path being deleted. Not decoration: a DELETE with a mistyped or
    stale `path` cannot then destroy something the admin never looked at.
    """
    cfg, conn = _st(request)["cfg"], _st(request)["conn"]
    src_root = os.path.abspath(cfg.source_root)
    target = _safe_target(cfg, path)

    if target == src_root:
        raise HTTPException(400, "refusing to delete the entire source root")
    if not os.path.exists(target):
        raise HTTPException(404, "no such file or folder")

    rel = os.path.relpath(target, src_root).replace("\\", "/")
    if confirm.strip().strip("/") != rel:
        raise HTTPException(400, {
            "message": "Confirmation does not match.",
            "detail": f"To delete this, `confirm` must be exactly {rel!r}. This guards "
                      f"against deleting something other than what you are looking at.",
        })

    ip = deps.client_ip(request)
    ua = request.headers.get("user-agent")
    actor = {"user_id": admin.get("db_user_id"), "email": admin["email"]}

    # ---- 1. WHICH DOCUMENTS DIE? ------------------------------------------
    # Worked out BEFORE anything is removed: once the files are gone we can no longer ask
    # the filesystem what they were.
    doc_keys = _docs_under(conn, cfg, rel, target)

    is_dir = os.path.isdir(target)
    n_files = sum(len(fs) for _, _, fs in os.walk(target)) if is_dir else 1

    # ---- 2. REMOVE FROM THE DATABASE --------------------------------------
    # FIRST, deliberately. If the DB delete fails, the files are still on disk and NOTHING
    # is lost -- the admin retries. Doing it the other way round (files first) would, on a
    # DB error, leave documents that are retrievable but whose source files 404 when an
    # officer clicks through to them. Fail towards the recoverable state.
    try:
        purged = _purge_docs(conn, doc_keys)
    except Exception as e:      # noqa: BLE001
        conn.rollback()         # leave the connection usable; a poisoned tx breaks every
                                # later request on this shared connection
        log.exception("hard delete: DB purge failed for %s", rel)
        raise HTTPException(500, f"nothing was deleted — the database purge failed: {e}")

    # ---- 3. REMOVE organized/ ---------------------------------------------
    # The crawler's copy. Left behind it would be re-parsed on the next crawl and the
    # document would rise from the dead.
    org_removed = _remove_organized(cfg, doc_keys)

    # ---- 4. REMOVE THE SOURCE FILES ---------------------------------------
    try:
        if is_dir:
            shutil.rmtree(target)
        else:
            os.unlink(target)
    except OSError as e:
        raise HTTPException(500, f"database rows were removed, but the files could not be "
                                 f"deleted: {e}")

    # ---- 5. DROP ANY QUEUED WORK FOR THOSE PATHS --------------------------
    # A pending job for a path that no longer exists would fail noisily on the next tick
    # and then retry three times before giving up. Cancel it instead.
    with conn.cursor() as cur:
        cur.execute("""DELETE FROM sync_queue
                       WHERE source_path = %s OR source_path LIKE %s""",
                    (target, target + os.sep + "%"))
        cancelled = cur.rowcount
    conn.commit()

    for dk in doc_keys:
        q.log_action(conn, "hard_deleted", doc_key=dk, source_path=target,
                     detail=f"admin {admin['email']} deleted {rel!r} (irreversible)")
    conn.commit()

    users.audit(conn, "tree_deleted", success=True, actor=actor,
                reason=f"HARD delete {rel!r}: {n_files} file(s), "
                       f"{len(doc_keys)} document(s), {purged['sub_questions']} vector(s)",
                ip=ip, user_agent=ua)
    log.warning("HARD DELETE by %s: %s — %d file(s), %d document(s) destroyed",
                admin["email"], rel, n_files, len(doc_keys))

    return {
        "ok": True,
        "deleted": rel,
        "files_removed": n_files,
        "documents_removed": len(doc_keys),
        "doc_keys": doc_keys,
        "sub_questions_removed": purged["sub_questions"],
        "answers_removed": purged["answer_groups"],
        "organized_removed": org_removed,
        "queued_jobs_cancelled": cancelled,
        "message": (f"Permanently deleted. {len(doc_keys)} document(s) removed from the "
                    f"database and from search." if doc_keys else
                    "Files deleted. No indexed document was affected."),
    }


def _docs_under(conn, cfg, rel: str, target: str) -> list[str]:
    """
    The doc_keys that this path represents.

    ⚠️ MAPPED THROUGH doc_key, NEVER question_id ALONE. ⚠️
    A diary number is reused across sessions for a DIFFERENT question. Deleting '8779'
    from one session by matching the number alone would also destroy 8779 in every other
    session -- which is exactly the class of bug that already cost this project 9
    documents once.
    """
    parts = [p for p in rel.split("/") if p]
    if not parts:
        return []

    session_slug, _ = normalize_session(parts[0])
    if session_slug is None:
        return []                       # never ingested -> nothing in the DB to remove

    with conn.cursor() as cur:
        if len(parts) == 1:
            # a whole session
            cur.execute("SELECT doc_key FROM diaries WHERE session = %s", (session_slug,))
        elif len(parts) == 2:
            # a house within a session. The crawler normalises the house name, so ask it
            # rather than guessing the slug here.
            from nhpc_qa.pipeline.crawl.crawler import normalize_house
            house_slug, _is_cat, _orphan = normalize_house(parts[1])
            if not house_slug:
                return []
            cur.execute("SELECT doc_key FROM diaries WHERE session = %s AND house = %s",
                        (session_slug, house_slug))
        else:
            # a question folder, or a file inside one -> exactly ONE document
            from nhpc_qa.pipeline.crawl.crawler import normalize_house
            house_slug, _is_cat, _orphan = normalize_house(parts[1])
            if not house_slug:
                return []
            cur.execute("""SELECT doc_key FROM diaries
                           WHERE session = %s AND house = %s AND question_id = %s""",
                        (session_slug, house_slug, parts[2]))
        return [r[0] for r in cur.fetchall()]


def _purge_docs(conn, doc_keys: list[str]) -> dict:
    """
    Destroy the documents. Rows, answers, tables, annexures and vectors.

    Deleting the diaries row would cascade anyway (the FKs are ON DELETE CASCADE), but the
    children are counted first so the admin is told exactly what was destroyed. "Deleted."
    is not an adequate answer for an irreversible action.
    """
    if not doc_keys:
        return {"sub_questions": 0, "answer_groups": 0, "diaries": 0}
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM sub_questions WHERE doc_key = ANY(%s)", (doc_keys,))
        n_sq = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM answer_groups WHERE doc_key = ANY(%s)", (doc_keys,))
        n_ag = cur.fetchone()[0]

        # Explicit, in FK order -- not relying on cascade behaviour that a future migration
        # could quietly change.
        #
        # answer_table_rows has no doc_key of its own; it hangs off answer_tables via
        # table_id (NOT 'answer_table_id' -- checked against the schema, not guessed).
        cur.execute("""DELETE FROM answer_table_rows WHERE table_id IN (
                           SELECT table_id FROM answer_tables WHERE doc_key = ANY(%s))""",
                    (doc_keys,))
        for tbl in ("answer_tables", "annexures", "answer_groups", "sub_questions"):
            cur.execute(f"DELETE FROM {tbl} WHERE doc_key = ANY(%s)", (doc_keys,))
        cur.execute("DELETE FROM diaries WHERE doc_key = ANY(%s)", (doc_keys,))
        n_d = cur.rowcount
    conn.commit()
    return {"sub_questions": n_sq, "answer_groups": n_ag, "diaries": n_d}


def _remove_organized(cfg, doc_keys: list[str]) -> int:
    """
    Remove the crawler's copies under organized/<session>/<house>/<qid>/.

    Not housekeeping: organized/ is what the PARSE stage reads. Leaving a copy behind means
    the next crawl re-parses it and re-inserts the document -- the delete would silently
    undo itself.
    """
    org_root = os.path.abspath(getattr(cfg, "organized_root", "organized"))
    n = 0
    for dk in doc_keys:
        d = os.path.join(org_root, *dk.split("/"))
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    return n
