"""
Admin-only upload — the intake path in FRONT of the existing pipeline.

    POST /admin/upload                  multipart: files[] + paths[] + on_conflict
    GET  /admin/upload/{id}/status      queued | processing | done | failed
    GET  /admin/uploads                 recent uploads

WHAT THIS DOES NOT DO: it does not parse, embed, index, or crawl anything. It writes files
into the source tree and calls queue.enqueue() -- THE SAME FUNCTION the watchdog handler
calls (watcher/runner.py:_enqueue). There is exactly one pipeline and this is not a second
one; it is a second way of putting work on the existing queue.

THE WRITE PATH

    receive (streamed to disk, never buffered in RAM)
        -> STAGE      <staging>/<upload_id>/...       a temp dir, NOT the source tree
        -> VALIDATE   extension AND magic bytes AND size; sanitise + realpath-jail
        -> COLLIDE?   if a target exists -> 409, require an explicit admin choice
        -> MOVE       os.replace() -- atomic within a volume
        -> ENQUEUE    queue.enqueue(folder, 'upsert', settle_seconds)
        -> 202        {upload_id, accepted, rejected[], queued_folders}

Staging is on the SAME VOLUME as the source root (config validates this), so the move is a
true atomic rename rather than a copy. The watcher therefore can never observe a
half-written file -- which is not a theoretical worry: parsing a folder mid-copy would read
a reply whose annexure has not landed and record 'referenced but unavailable' AS FACT.

ALL-OR-NOTHING. If any file fails validation, the whole upload is refused and NOTHING is
written. A session folder is a unit; ingesting 39 of 40 files produces a subtly corrupt
document rather than an obviously missing one, and the subtle failure is far worse.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
import uuid

from fastapi import (APIRouter, Depends, File, Form, HTTPException, Query, Request,
                     UploadFile)

from nhpc_qa.core.logging import get_logger
from nhpc_qa.api.security import deps, upload_guard as guard, users
from nhpc_qa.api.security.upload_guard import Rejected
# The queue lives in core/, NOT watcher/: `api` and `watcher` are SIBLING layers and
# neither may import the other (tests/test_layering.py enforces it). A durable work
# queue is INFRASTRUCTURE, not watcher policy, so core/ is where it belongs -- and both
# siblings may use it. This is the same enqueue() the watchdog handler calls.
from nhpc_qa.core import queue as q
from nhpc_qa.core.queue import question_folder
# Ask the CRAWLER whether a session folder name is usable -- never re-implement that rule
# here. Two independent notions of "which session is this?" would drift apart, and the
# crawler's is the one that actually decides doc_key. (api -> pipeline is a legal
# direction; see tests/test_layering.py.)
from nhpc_qa.pipeline.crawl.crawler import (SESSION_YEAR_MAX, SESSION_YEAR_MIN,
                                            normalize_session)

log = get_logger("nhpc.upload")

router = APIRouter()

_CHUNK = 1024 * 1024          # stream in 1 MiB chunks; never read a file into memory


def _st(request: Request):
    return request.app.state.nhpc


# ---------------------------------------------------------------------------
# POST /admin/upload
# ---------------------------------------------------------------------------
@router.post("/admin/upload")
async def upload(request: Request,
                 files: list[UploadFile] = File(...),
                 paths: list[str] = Form(default=[]),
                 dest: str = Form(default=""),
                 on_conflict: str = Form(default="fail"),
                 admin=Depends(deps.require_admin)):
    """
    `paths[i]` is the browser's webkitRelativePath for `files[i]` -- the folder structure
    the admin selected. It is CLIENT-SUPPLIED and therefore hostile until sanitised and
    jailed (see upload_guard).

    The relative structure is PRESERVED byte-for-byte. It is deliberately NOT normalised:
    the real corpus is messy ('PARLIAMENT FEB MAR 24/likely issues/information received/'),
    and the crawler already infers session/house from whatever it finds. Imposing a shape
    here would break the crawler's own inference.
    """
    cfg, conn = _st(request)["cfg"], _st(request)["conn"]
    if not cfg.upload_enabled:
        raise HTTPException(503, "uploads are disabled (UPLOAD_ENABLED=false)")
    if on_conflict not in ("fail", "skip", "replace"):
        raise HTTPException(400, "on_conflict must be fail | skip | replace")

    upload_id = f"up_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:8]}"
    ip = deps.client_ip(request)
    ua = request.headers.get("user-agent")
    actor = {"user_id": admin.get("db_user_id"), "email": admin["email"]}

    src_root = os.path.abspath(cfg.source_root)
    stage_dir = os.path.join(cfg.staging_root(), upload_id)
    allowed = cfg.allowed_exts()
    max_file = cfg.upload_max_file_mb * 1024 * 1024
    max_total = cfg.upload_max_total_mb * 1024 * 1024

    if len(files) > cfg.upload_max_files:
        _record(conn, upload_id, actor, len(files), 0, len(files), 0, "rejected",
                f"too many files ({len(files)} > {cfg.upload_max_files})", [], ip, ua, [])
        raise HTTPException(413, f"too many files: {len(files)} "
                                 f"(max {cfg.upload_max_files})")

    # ---- 0a. THE DESTINATION -----------------------------------------------
    # `dest` is the folder in the tree the admin selected (''=source root). Every uploaded
    # path is written UNDER it. This is what makes "Choose files..." meaningful at all:
    # a bare file has no webkitRelativePath, so before this it fell back to its own
    # filename and landed at the SOURCE ROOT -- belonging to no session, no house, no
    # question. The crawler ignores such a file, so it could never be ingested: a silent
    # black hole. A file with nowhere to go is now refused rather than swallowed.
    dest = (dest or "").strip().strip("/").strip("\\")
    if dest:
        try:
            dest = guard.sanitize_relpath(dest)
            dest_abs = guard.jail(src_root, dest)      # the same jail as every other path
        except Rejected as e:
            raise HTTPException(400, f"invalid destination: {e}")
        if not os.path.isdir(dest_abs):
            raise HTTPException(404, f"destination folder does not exist: {dest}")

    def _full_rel(client_path: str) -> str:
        """The path this file will occupy, relative to the source root."""
        cp = (client_path or "").replace("\\", "/").strip("/")
        return f"{dest}/{cp}" if dest else cp

    # A loose FILE dropped at the root can never be ingested -- the crawler needs
    # <session>/<house>/<question>/. Refuse it, loudly, instead of writing it nowhere.
    if not dest:
        loose = [((paths[i] if i < len(paths) and paths[i] else f.filename) or "")
                 for i, f in enumerate(files)]
        if any("/" not in p.replace("\\", "/").strip("/") for p in loose):
            raise HTTPException(400, {
                "message": "These files have no destination.",
                "detail": "A file uploaded to the top level belongs to no session, no "
                          "house and no question, so the pipeline can never ingest it. "
                          "Pick a destination folder in the tree first, or upload a whole "
                          "session folder.",
                "rejected": [{"client_path": p, "reason": "no destination folder chosen"}
                             for p in loose
                             if "/" not in p.replace("\\", "/").strip("/")],
            })

    # ---- 0b. IS THE SESSION FOLDER EVEN INGESTIBLE? ------------------------
    # Reject at the DOOR, before a single byte is written. The crawler derives the session
    # slug from the TOP-LEVEL folder name; if it cannot, it skips the folder as an orphan
    # and the files sit in the source tree forever, ingested by nothing.
    #
    # This is what 'PARLIAMENT DEC-JAN 1915' did: 55 files were accepted, written, queued
    # and reported "done" -- having ingested precisely nothing. Accepting an upload we
    # KNOW cannot be processed is not politeness, it is a lie.
    #
    # The session is the FIRST component of the FULL path (dest + client path), so this
    # holds whether the admin uploaded a new session at the root or a question folder into
    # an existing house.
    bad_sessions = {}
    for i, f in enumerate(files):
        cp = (paths[i] if i < len(paths) and paths[i] else f.filename) or ""
        top = _full_rel(cp).split("/")[0].strip()
        if not top or top in bad_sessions:
            continue
        slug, _ = normalize_session(top)
        if slug is None:
            bad_sessions[top] = None
    if bad_sessions:
        names = ", ".join(repr(n) for n in bad_sessions)
        _record(conn, upload_id, actor, len(files), 0, len(files), 0, "rejected",
                f"unrecognisable session folder(s): {names}", [], ip, ua, [])
        users.audit(conn, "upload_rejected", success=False, actor=actor,
                    reason=f"unrecognisable session folder: {names}", ip=ip, user_agent=ua)
        raise HTTPException(400, {
            "message": f"The session folder name cannot be understood: {names}.",
            # The bounds are READ from the crawler, never restated as literals -- a
            # message that disagrees with the rule it describes is worse than no message.
            "detail": (f"Nothing was written. The crawler derives the session from this "
                       f"folder's name and needs month/season tokens plus a 2- or 4-digit "
                       f"year between {SESSION_YEAR_MIN} and {SESSION_YEAR_MAX} — e.g. "
                       f"'PARLIAMENT DEC-JAN 25', 'PARLIAMENT MONSOON 24', "
                       f"'PARLIAMENT FEB MAR 2026'. A number outside that range is not "
                       f"read as a year at all. Rename the folder and upload again."),
            "rejected": [{"client_path": n, "reason": "unrecognisable session folder name"}
                         for n in bad_sessions],
        })

    os.makedirs(stage_dir, exist_ok=True)
    accepted, rejected = [], []
    total = 0

    try:
        # ---- 1. STAGE + VALIDATE -------------------------------------------
        for i, f in enumerate(files):
            client_path = (paths[i] if i < len(paths) and paths[i] else f.filename) or ""
            try:
                # Rooted at the chosen destination, so a file can only land somewhere the
                # pipeline can actually see it.
                rel = guard.sanitize_relpath(_full_rel(client_path))
                ext = guard.check_extension(rel, allowed)

                # Prove the destination is inside the root BEFORE writing a single byte.
                # (Re-proved after the realpath below; this is the cheap early exit.)
                target = guard.jail(src_root, rel)

                staged = os.path.join(stage_dir, rel.replace("/", os.sep))
                os.makedirs(os.path.dirname(staged), exist_ok=True)

                size, digest = await _stream_to_disk(f, staged, max_file)
                total += size
                if total > max_total:
                    raise Rejected(
                        f"upload exceeds the total limit "
                        f"({cfg.upload_max_total_mb} MB)")

                # Content sniff AFTER the bytes are on disk, BEFORE anything moves into the
                # live tree. Extension alone is worthless -- anyone can rename evil.exe.
                guard.sniff(staged, ext)

                accepted.append({"client_path": client_path, "rel": rel,
                                 "staged": staged, "target": target,
                                 "size": size, "sha256": digest})
            except Rejected as e:
                rejected.append({"client_path": client_path, "reason": str(e)})
                log.warning("upload %s REJECTED %r: %s", upload_id, client_path, e)

        # ---- 2. ALL-OR-NOTHING ---------------------------------------------
        if rejected:
            _record(conn, upload_id, actor, len(files), 0, len(rejected), 0, "rejected",
                    "one or more files failed validation; nothing was written",
                    [], ip, ua, rejected + [{"client_path": a["client_path"],
                                             "reason": "not written (upload aborted)",
                                             "ok": False} for a in accepted])
            users.audit(conn, "upload_rejected", success=False, actor=actor,
                        reason=f"{len(rejected)}/{len(files)} files failed validation",
                        ip=ip, user_agent=ua)
            raise HTTPException(400, {
                "message": "Upload refused — nothing was written to the source data.",
                "detail": "A session folder is a unit: ingesting some of its files would "
                          "produce a document with silently missing annexures.",
                "rejected": rejected,
            })

        if not accepted:
            raise HTTPException(400, "no files in the upload")

        # ---- 3. COLLISIONS — never a silent overwrite ----------------------
        collisions = [a for a in accepted if os.path.exists(a["target"])]
        if collisions and on_conflict == "fail":
            _record(conn, upload_id, actor, len(files), 0, 0, 0, "conflict",
                    f"{len(collisions)} file(s) already exist", [], ip, ua, [])
            users.audit(conn, "upload_conflict", success=False, actor=actor,
                        reason=f"{len(collisions)} existing file(s)", ip=ip, user_agent=ua)
            raise HTTPException(409, {
                "message": "These files already exist in the source data.",
                "detail": "Nothing was written. Choose 'skip' to upload only the new "
                          "files, or 'replace' to overwrite (this is logged).",
                "conflicts": [os.path.relpath(c["target"], src_root).replace("\\", "/")
                              for c in collisions],
            })

        # ---- 4. ATOMIC MOVE into the source tree ---------------------------
        to_write, skipped = [], []
        for a in accepted:
            if os.path.exists(a["target"]):
                if on_conflict == "skip":
                    skipped.append(a)
                    continue
                a["replaced"] = True          # on_conflict == replace
            to_write.append(a)

        written = []
        for a in to_write:
            os.makedirs(os.path.dirname(a["target"]), exist_ok=True)
            # os.replace is ATOMIC within a volume: the file appears complete or not at
            # all. The watcher can never see it half-written.
            os.replace(a["staged"], a["target"])
            written.append(a)
            if a.get("replaced"):
                log.warning("upload %s REPLACED existing file: %s", upload_id, a["rel"])

        # ---- 5. HAND OFF TO THE EXISTING PIPELINE --------------------------
        # queue.enqueue() is THE SAME call the watchdog handler makes. If the watcher is
        # also running it will fire its own events for these paths -- and ON CONFLICT DO
        # UPDATE coalesces them into this same job. No double processing, and the upload
        # still works when the watcher is down.
        folders = sorted({question_folder(cfg, a["target"]) for a in written} - {None})
        for folder in folders:
            qid, coalesced = q.enqueue(conn, folder, "upsert", cfg.watch_settle_seconds)
            log.info("upload %s -> job %s %s%s", upload_id, qid,
                     os.path.relpath(folder, src_root),
                     " [coalesced with a watcher event]" if coalesced else "")
        conn.commit()

        rel_folders = [os.path.relpath(f, src_root).replace("\\", "/") for f in folders]
        bytes_written = sum(a["size"] for a in written)

        _record(conn, upload_id, actor, len(files), len(written), len(skipped),
                bytes_written, "accepted", None, rel_folders, ip, ua,
                [{"client_path": a["client_path"], "ok": True,
                  "stored": os.path.relpath(a["target"], src_root).replace("\\", "/"),
                  "size": a["size"], "sha256": a["sha256"],
                  "reason": "replaced" if a.get("replaced") else None} for a in written]
                + [{"client_path": a["client_path"], "ok": False,
                    "reason": "skipped (already exists)"} for a in skipped])

        users.audit(conn, "upload_accepted", success=True, actor=actor,
                    reason=f"{len(written)} file(s), {bytes_written} bytes, "
                           f"{len(folders)} folder(s) queued", ip=ip, user_agent=ua)

        watcher_alive = _watcher_alive(conn)
        return {
            "upload_id": upload_id,
            "accepted": len(written),
            "skipped": len(skipped),
            "bytes": bytes_written,
            "queued_folders": rel_folders,
            "files": [{"path": os.path.relpath(a["target"], src_root).replace("\\", "/"),
                       "size": a["size"],
                       "replaced": bool(a.get("replaced"))} for a in written],
            "watcher_running": watcher_alive,
            "message": ("Uploaded; processing started."
                        if watcher_alive else
                        "Uploaded and QUEUED — but no watcher is running, so nothing is "
                        "processing it yet. Start `nhpc watch` (or the nhpc-watch service) "
                        "and it will be picked up automatically. Nothing is lost."),
        }

    finally:
        # Staging is temporary by definition. Anything left here was rejected or already
        # moved; either way it must not accumulate.
        shutil.rmtree(stage_dir, ignore_errors=True)


async def _stream_to_disk(f: UploadFile, dest: str, max_bytes: int):
    """
    Stream the upload to disk in chunks, enforcing the size cap AS WE GO.

    Checking the size after the fact would mean a 10 GB upload had already been written to
    disk before we refused it -- which is the disk-exhaustion attack, not a defence against
    it. We stop at the first byte over the limit.
    """
    h = hashlib.sha256()
    size = 0
    with open(dest, "wb") as out:
        while True:
            chunk = await f.read(_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                out.close()
                os.unlink(dest)
                raise Rejected(f"file exceeds the {max_bytes // (1024*1024)} MB limit")
            h.update(chunk)
            out.write(chunk)
    return size, h.hexdigest()


def _watcher_alive(conn) -> bool:
    """
    Is a watcher actually running? READS ITS HEARTBEAT -- it does not guess.

    This used to INFER liveness from the shape of sync_queue: "nothing claimed in the last
    120s AND nothing pending past its settle time -> a watcher must be running". That
    inference is wrong in the most ordinary situation there is. On an IDLE system with an
    empty queue and NO WATCHER AT ALL, nothing has been claimed recently (there was nothing
    to claim) and nothing is overdue (nothing is pending) -- so it answered "healthy". It
    only reported the truth once jobs were already piling up, i.e. after the admin had
    noticed the problem for themselves.

    The watcher now writes a heartbeat row on a fixed cadence, from its own thread, so a
    BUSY watcher (minutes inside Docling + the LLM) is not mistaken for a DEAD one either.
    """
    alive, _detail = q.watcher_alive(conn)
    return alive


def _record(conn, upload_id, actor, offered, acc, rej, nbytes, outcome, reason,
            folders, ip, ua, files):
    """Write the audit rows. Rejections are recorded too -- 'why is that file not in the
    system?' must always be answerable."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO uploads (upload_id, actor_user_id, actor_email,
                                     n_files_offered, n_files_accepted, n_files_rejected,
                                     bytes_accepted, outcome, reason, queued_folders,
                                     ip, user_agent)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (upload_id) DO NOTHING
            """, (upload_id, actor.get("user_id"), actor.get("email"),
                  offered, acc, rej, nbytes, outcome, reason, folders, ip, ua))
            for f in files:
                cur.execute("""
                    INSERT INTO upload_files (upload_id, client_path, stored_path,
                                              size_bytes, sha256, accepted, reason)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                """, (upload_id, f.get("client_path"), f.get("stored"),
                      f.get("size"), f.get("sha256"), bool(f.get("ok")), f.get("reason")))
        conn.commit()
    except Exception as e:      # noqa: BLE001 -- audit must never break the upload
        conn.rollback()
        log.error("upload audit write failed: %s: %s", type(e).__name__, e)


# ---------------------------------------------------------------------------
# GET /admin/upload/{id}/status  — reads the EXISTING pipeline records
# ---------------------------------------------------------------------------
@router.get("/admin/upload/{upload_id}/status")
def upload_status(request: Request, upload_id: str, admin=Depends(deps.require_admin)):
    """
    Status comes from sync_queue and sync_log -- the pipeline's own records. No new state
    is invented, so the status cannot drift from what actually happened.
    """
    cfg, conn = _st(request)["cfg"], _st(request)["conn"]
    src_root = os.path.abspath(cfg.source_root)

    with conn.cursor() as cur:
        cur.execute("""SELECT upload_id, actor_email, n_files_accepted, n_files_rejected,
                              bytes_accepted, outcome, reason, queued_folders, created_at
                       FROM uploads WHERE upload_id = %s""", (upload_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "no such upload")
        cols = [c.name for c in cur.description]
        up = dict(zip(cols, row))

        abs_folders = [os.path.join(src_root, f.replace("/", os.sep))
                       for f in (up["queued_folders"] or [])]

        jobs = []
        if abs_folders:
            cur.execute("""SELECT source_path, status, attempts, last_error, settle_until
                           FROM sync_queue WHERE source_path = ANY(%s)""", (abs_folders,))
            jcols = [c.name for c in cur.description]
            jobs = [dict(zip(jcols, r)) for r in cur.fetchall()]

        # What actually LANDED. sync_log is written by the pipeline itself.
        landed = []
        if abs_folders:
            cur.execute("""SELECT action, doc_key, n_sub_questions, detail, created_at
                           FROM sync_log
                           WHERE source_path = ANY(%s) AND created_at >= %s
                           ORDER BY created_at""", (abs_folders, up["created_at"]))
            lcols = [c.name for c in cur.description]
            landed = [dict(zip(lcols, r)) for r in cur.fetchall()]

    watcher_ok, watcher_detail = q.watcher_alive(conn)

    # A queued folder with NO job row means the watcher already finished and cleaned it.
    by_status = {}
    for j in jobs:
        by_status[j["status"]] = by_status.get(j["status"], 0) + 1

    if up["outcome"] != "accepted":
        overall = up["outcome"]
    elif not jobs:
        overall = "done" if landed else "queued"
    elif by_status.get("failed"):
        overall = "failed"
    elif by_status.get("processing"):
        overall = "processing"
    elif by_status.get("pending"):
        overall = "queued"
    else:
        overall = "done"

    return {
        "upload_id": upload_id,
        "status": overall,
        "outcome": up["outcome"],
        "uploaded_by": up["actor_email"],
        "files_accepted": up["n_files_accepted"],
        "created_at": up["created_at"].isoformat(),
        "jobs": [{"folder": os.path.relpath(j["source_path"], src_root).replace("\\", "/"),
                  "status": j["status"], "attempts": j["attempts"],
                  "error": j["last_error"]} for j in jobs],
        "indexed": [{"action": l["action"], "doc_key": l["doc_key"],
                     "sub_questions": l["n_sub_questions"]} for l in landed],
        "documents_indexed": len([l for l in landed if l["action"] in ("add", "update")]),
        "watcher_running": watcher_ok,
        # WHY it is considered down -- 'no watcher has ever registered' vs 'last heartbeat
        # 340s ago' are different problems and want different fixes.
        "watcher": watcher_detail,
    }


@router.get("/admin/uploads")
def recent_uploads(request: Request,
                   page: int = Query(default=1, ge=1),
                   per_page: int = Query(default=5, ge=1, le=50),
                   admin=Depends(deps.require_admin)):
    """
    Paginated. The upload history only grows -- one row per upload, forever -- so returning
    all of it would get slower every week and eventually render a page nobody can read.
    Paginate in SQL (LIMIT/OFFSET), not in the browser: fetching 10,000 rows to display 5
    is the same mistake with extra steps.
    """
    conn = _st(request)["conn"]
    offset = (page - 1) * per_page
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM uploads")
        total = cur.fetchone()[0]
        cur.execute("""SELECT upload_id, actor_email, n_files_accepted, n_files_rejected,
                              bytes_accepted, outcome, created_at, queued_folders
                       FROM uploads ORDER BY created_at DESC
                       LIMIT %s OFFSET %s""", (per_page, offset))
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    pages = max(1, (total + per_page - 1) // per_page)
    return {
        "uploads": [{
            "upload_id": r["upload_id"], "by": r["actor_email"],
            "accepted": r["n_files_accepted"], "rejected": r["n_files_rejected"],
            "bytes": r["bytes_accepted"], "outcome": r["outcome"],
            "folders": r["queued_folders"],
            "at": r["created_at"].isoformat(),
        } for r in rows],
        "page": page, "per_page": per_page, "total": total, "pages": pages,
        "has_prev": page > 1, "has_next": page < pages,
    }
