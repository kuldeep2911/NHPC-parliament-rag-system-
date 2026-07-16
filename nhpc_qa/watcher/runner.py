"""
`nhpc watch` — observe the source tree and sync incrementally.

TWO HALVES, deliberately separated:

  OBSERVER (watchdog thread)  ->  enqueue events into the DURABLE queue, and nothing else
  WORKER  (main loop)         ->  claim settled jobs and run the pipeline slice

They talk only through the Postgres queue. That separation is what makes the service
crash-safe: the observer never does slow work (so it cannot miss events while parsing),
and the worker can be killed at any moment without losing a pending event, because the
event is already in the DB rather than in a Python list.

SETTLING. A file event does NOT trigger processing. The affected QUESTION FOLDER is
enqueued with a deadline `settle_until = now + WATCH_SETTLE_SECONDS`, and every further
event for that folder PUSHES THE DEADLINE OUT. So copying a 40-file session folder
produces ONE job that fires once the copying has stopped -- never a parse of a half-copied
folder, and never 40 racing jobs.

READ-ONLY on the source: the observer only watches. The crawl stage copies OUT of the
source into organized/; nothing ever writes back in.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time

from nhpc_qa.core.logging import get_logger, setup as setup_logging
from nhpc_qa.core import queue as q
from nhpc_qa.watcher import sync

log = get_logger("nhpc.watcher")

_STOP = threading.Event()

# Editor/OS noise that must never trigger a pipeline run.
_IGNORE = (".tmp", ".swp", ".crdownload", ".part", "~$")


def _ignored(path: str) -> bool:
    """
    Should this path never trigger a pipeline run?

    ⚠️ ANY dot-component, not just the basename. This used to check only
    os.path.basename(path), which meant a file inside a dot-DIRECTORY was NOT ignored:

        .upload_staging/PARLIAMENT MAR 26/LOK SABHA/1234/reply.pdf
        ^^^^^^^^^^^^^^^ dotted dir            basename is 'reply.pdf' -> not ignored

    The admin upload endpoint stages files under <source_root>/.upload_staging before
    atomically moving them into place. With the basename-only check the watcher would
    enqueue those files WHILE THEY WERE STILL BEING WRITTEN -- exactly the half-copied
    parse that the staging + atomic-move design exists to prevent. Ignoring the whole
    dotted subtree closes that.
    """
    parts = os.path.normpath(path).replace("\\", "/").split("/")
    for seg in parts:
        if seg.startswith(".") and seg not in (".", ".."):
            return True
        if seg.startswith("~$"):            # Word/Excel lock files
            return True
    name = parts[-1] if parts else ""
    return any(name.endswith(x) for x in _IGNORE)


# ---------------------------------------------------------------------------
# observer half — enqueue only, never process
# ---------------------------------------------------------------------------

def _make_handler(cfg, conn_factory):
    from watchdog.events import FileSystemEventHandler

    class Handler(FileSystemEventHandler):
        """Every event is coalesced onto its QUESTION FOLDER and pushed to the queue."""

        def _enqueue(self, path, event_type):
            if _ignored(path):
                return
            folder = sync.question_folder(cfg, path)
            if not folder:
                return
            try:
                with conn_factory() as conn:
                    qid, coalesced = q.enqueue(conn, folder, event_type,
                                               cfg.watch_settle_seconds)
                log.info("event %-6s %s -> job %s%s (settling %ss)",
                         event_type, os.path.relpath(folder, cfg.source_root), qid,
                         " [coalesced]" if coalesced else "", cfg.watch_settle_seconds)
            except Exception as e:      # noqa: BLE001 -- the observer must never die
                log.error("enqueue failed for %s: %s: %s", path, type(e).__name__, e)

        # A created/modified/moved path is an upsert; the pipeline is idempotent, so we do
        # not try to distinguish "new" from "changed" here.
        #
        # ⚠️ DIRECTORY creation is IGNORED, and that is not an optimisation -- it is a
        # correctness fix. question_folder() maps a path to AT MOST session/house/question,
        # but it cannot INVENT depth it was not given: handed the bare session directory it
        # returns the SESSION, and the resulting job re-crawls every question in it.
        #
        # Creating one nested question folder fires on_created for each new directory
        # level, so a single upload of
        #     PARLIAMENT MAR APR 26/LOK SABHA/9911/reply.pdf
        # enqueued THREE overlapping jobs -- the session, the house, and the question. They
        # are distinct source_path values, so the queue's ON CONFLICT dedup cannot see that
        # they are nested, and the session-level job then re-crawls the whole session while
        # the question-level job waits behind it. (Observed: a job stuck 'processing' for
        # 10+ minutes, blocking the upload's own job.)
        #
        # A directory is EMPTY at the moment it is created. Nothing is lost by ignoring it:
        # every file that lands inside it fires its own event, and each of those carries
        # the full depth question_folder() needs. on_modified already had this guard.
        def on_created(self, e):
            if not e.is_directory:
                self._enqueue(e.src_path, "upsert")

        def on_modified(self, e):
            if not e.is_directory:
                self._enqueue(e.src_path, "upsert")

        def on_moved(self, e):
            # A directory MOVE is different from a directory CREATE: a moved-in folder
            # arrives with its files already inside, and those files fire no events of
            # their own. So a moved directory must still be enqueued -- but on its
            # question folder, which for a deep move is exactly what question_folder gives.
            self._enqueue(e.dest_path, "upsert")
            self._enqueue(e.src_path, "delete")

        def on_deleted(self, e):
            # SOFT delete only -- see sync.process_delete. Nothing is removed here.
            self._enqueue(e.src_path, "delete")

    return Handler()


def _make_supporting_handler(cfg, conn_factory):
    """
    Watches organized/supporting_documents/. A file dropped straight into a category folder
    is parsed + stored (into supporting_* tables, NOT the Q&A index); a file removed is
    soft-deleted.

    Supporting docs are single files, not multi-file question folders, so this settles
    PER FILE with an in-process debounce timer rather than the durable queue: a file is
    ingested only once it has stopped changing for settle_seconds, so a still-copying file
    is never parsed half-written. Ingestion is idempotent (sha256), so a duplicate timer
    firing is harmless.
    """
    import threading
    from watchdog.events import FileSystemEventHandler
    from nhpc_qa.supporting import ingest as sup_ingest

    settle = max(2, int(getattr(cfg, "watch_settle_seconds", 10)))
    timers = {}
    lock = threading.Lock()

    def _process(path, kind):
        try:
            with conn_factory() as conn:
                if kind == "delete":
                    sup_ingest.soft_delete_path(cfg, conn, path)
                else:
                    # The upload endpoint ALSO ingests inline; this only fires for files
                    # placed in the folder by hand. LLM as-of is best-effort here (None if
                    # no LLM), the admin can correct it in the Documents screen.
                    llm = None
                    try:
                        from nhpc_qa.core.providers import get_llm
                        llm = get_llm(cfg) if getattr(cfg, "supporting_llm_asof", False) else None
                    except Exception:      # noqa: BLE001
                        llm = None
                    sup_ingest.ingest_path(cfg, conn, path, uploaded_by="watcher", llm=llm)
        except Exception as e:      # noqa: BLE001 -- the watcher must never die
            log.error("supporting ingest failed for %s: %s: %s", path, type(e).__name__, e)

    def _schedule(path, kind):
        if _ignored(path) or os.path.isdir(path):
            return
        # only real files under a category folder; a staging dotfile is already ignored
        with lock:
            t = timers.get(path)
            if t:
                t.cancel()
            # a delete fires immediately (nothing to settle); an add/modify settles first
            delay = 0.1 if kind == "delete" else settle
            timers[path] = threading.Timer(delay, lambda: (_process(path, kind),
                                                           timers.pop(path, None)))
            timers[path].daemon = True
            timers[path].start()
        log.info("supporting event %-6s %s (settling %ss)",
                 kind, os.path.basename(path), 0 if kind == "delete" else settle)

    class SupHandler(FileSystemEventHandler):
        def on_created(self, e):
            if not e.is_directory:
                _schedule(e.src_path, "upsert")

        def on_modified(self, e):
            if not e.is_directory:
                _schedule(e.src_path, "upsert")

        def on_moved(self, e):
            _schedule(e.dest_path, "upsert")
            _schedule(e.src_path, "delete")

        def on_deleted(self, e):
            if not e.is_directory:
                _schedule(e.src_path, "delete")

    return SupHandler()


# ---------------------------------------------------------------------------
# heartbeat half — "I am alive", independent of whether work is in progress
# ---------------------------------------------------------------------------

def _heartbeat_loop(cfg, source):
    """
    Beat on a fixed cadence, on its OWN connection, until _STOP.

    Deliberately independent of the work loop. The API asks "is a watcher running?" and
    must get the truth even while a single job is spending three minutes inside Docling +
    the LLM + the embedder. Tying the beat to the work loop would make a BUSY watcher look
    like a DEAD one.
    """
    from nhpc_qa.core.db.session import connect

    interval = max(5, min(cfg.watch_poll_seconds, q.HEARTBEAT_STALE_SECONDS // 3))
    while not _STOP.is_set():
        try:
            with connect(cfg) as hb_conn:
                while not _STOP.is_set():
                    q.heartbeat(hb_conn, source_root=source)
                    _STOP.wait(interval)
        except Exception:       # noqa: BLE001
            # A dropped DB connection must not kill the beat: back off and reconnect.
            # (If the DB is genuinely down, the API cannot read the heartbeat either, so
            # nothing is misreported -- it just cannot answer.)
            log.exception("heartbeat connection failed; retrying in %ds", interval)
            _STOP.wait(interval)


# ---------------------------------------------------------------------------
# worker half — claim settled jobs and run the slice
# ---------------------------------------------------------------------------

def _work_once(cfg, conn):
    """Claim and process every job whose quiet period has elapsed. Returns how many ran."""
    q.recover_stale(conn, cfg.watch_stale_seconds)

    done = 0
    while not _STOP.is_set():
        jobs = q.claim_ready(conn, limit=1)
        if not jobs:
            break
        job = jobs[0]
        rel = os.path.relpath(job["source_path"], cfg.source_root)
        log.info("processing job %s: %s %s (%d event(s) coalesced)",
                 job["id"], job["event_type"], rel, job["event_count"])
        try:
            if job["event_type"] == "delete":
                # The path may have come BACK during the settle window (a move, a retry).
                # If it exists again, this is not a deletion at all -- treat it as an
                # upsert, which will also reactivate the record if it was soft-deleted.
                if os.path.exists(job["source_path"]):
                    log.info("job %s: path reappeared during settling — treating as upsert",
                             job["id"])
                    sync.process_upsert(cfg, conn, job["source_path"])
                else:
                    sync.process_delete(cfg, conn, job["source_path"])
            else:
                if not os.path.exists(job["source_path"]):
                    log.info("job %s: path vanished during settling — nothing to ingest",
                             job["id"])
                    q.log_action(conn, "skipped", source_path=job["source_path"],
                                 detail="path disappeared before it settled")
                else:
                    sync.process_upsert(cfg, conn, job["source_path"])
            q.complete(conn, job["id"])
            done += 1
        except Exception as e:      # noqa: BLE001
            log.exception("job %s failed", job["id"])
            q.fail(conn, job["id"], f"{type(e).__name__}: {e}", cfg.watch_max_attempts)
            q.log_action(conn, "failed", source_path=job["source_path"],
                         detail=f"{type(e).__name__}: {e}")
    return done


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main(args=None):
    import contextlib

    from nhpc_qa.config import Settings, load_dotenv
    from nhpc_qa.core.db.session import connect

    setup_logging()
    load_dotenv()
    cfg = Settings()
    errs = cfg.validate_all(need_rerank=False)
    if errs:
        for e in errs:
            print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 1

    source = (getattr(args, "source", None) or cfg.source_root)
    source = os.path.abspath(source)
    if not os.path.isdir(source):
        print(f"source directory not found: {source}", file=sys.stderr)
        return 1
    cfg.source_root = source

    once = bool(getattr(args, "once", False))

    # Each caller gets its own short-lived connection: the observer thread and the worker
    # must not share one (psycopg connections are not thread-safe).
    @contextlib.contextmanager
    def conn_factory():
        with connect(cfg) as c:
            yield c

    # ---- drain mode: process what is already queued, then exit -------------
    if once:
        with connect(cfg) as conn:
            n = _work_once(cfg, conn)
            st = q.stats(conn)
        log.info("drained %d job(s); queue now %s", n, st or "empty")
        return 0

    # ---- service mode -----------------------------------------------------
    from watchdog.observers import Observer

    observer = Observer()
    observer.schedule(_make_handler(cfg, conn_factory), source, recursive=True)

    # Second watch: supporting reference documents (organized/supporting_documents/). A
    # file dropped there auto-parses into the supporting_* tables; a removed file is
    # soft-deleted. Separate tables, separate handler -- it never touches the Q&A queue.
    if getattr(cfg, "supporting_enabled", False):
        sup_root = cfg.supporting_root_abs()
        os.makedirs(sup_root, exist_ok=True)
        observer.schedule(_make_supporting_handler(cfg, conn_factory), sup_root,
                          recursive=True)
        log.info("also watching supporting documents: %s", sup_root)

    observer.start()

    def _stop(_sig, _frm):
        log.info("shutting down (signal)")
        _STOP.set()

    signal.signal(signal.SIGINT, _stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _stop)

    log.info("watching %s | settle=%ss poll=%ss | soft-delete only (purge is separate)",
             source, cfg.watch_settle_seconds, cfg.watch_poll_seconds)

    try:
        with connect(cfg) as conn:
            # Anything left claimed by a worker that died is released on startup -- this
            # is what makes a mid-processing restart safe.
            recovered = q.recover_stale(conn, 0)
            if recovered:
                log.info("recovered %d job(s) left claimed by a previous run", recovered)

            # Register immediately, so the API stops saying "no watcher is running" the
            # moment this process is up -- not one poll interval later.
            q.heartbeat(conn, source_root=source)

            # THE HEARTBEAT RUNS ON ITS OWN THREAD, not inside the work loop.
            #
            # A single _work_once() tick can legitimately take MINUTES: Docling, the
            # extraction LLM and the embedder all run inside it for a big question folder.
            # If the beat only happened between ticks it would go stale DURING that work,
            # and the API would report the watcher dead while it was in fact busy -- a
            # false alarm that is just as misleading as the false "all clear" this whole
            # change exists to remove.
            #
            # The thread gets its OWN connection: psycopg connections are not thread-safe,
            # and the worker holds `conn` for the length of a job.
            heart = threading.Thread(target=_heartbeat_loop, args=(cfg, source),
                                     name="nhpc-heartbeat", daemon=True)
            heart.start()

            while not _STOP.is_set():
                try:
                    _work_once(cfg, conn)
                except Exception:       # noqa: BLE001 -- the loop must survive anything
                    log.exception("worker tick failed; continuing")
                _STOP.wait(cfg.watch_poll_seconds)

            # Clean shutdown: deregister so the API knows AT ONCE. A crash skips this,
            # which is why staleness -- not this line -- is the real liveness check.
            q.worker_stopped(conn)
    finally:
        observer.stop()
        observer.join(timeout=5)
    log.info("watcher stopped")
    return 0
