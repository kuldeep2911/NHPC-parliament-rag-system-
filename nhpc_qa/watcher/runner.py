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
from nhpc_qa.watcher import queue as q
from nhpc_qa.watcher import sync

log = get_logger("nhpc.watcher")

_STOP = threading.Event()

# Editor/OS noise that must never trigger a pipeline run.
_IGNORE = (".tmp", ".swp", ".crdownload", ".part", "~$")


def _ignored(path: str) -> bool:
    name = os.path.basename(path)
    if name.startswith(".") or name.startswith("~$"):
        return True
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
        def on_created(self, e):
            self._enqueue(e.src_path, "upsert")

        def on_modified(self, e):
            if not e.is_directory:
                self._enqueue(e.src_path, "upsert")

        def on_moved(self, e):
            # the destination is an upsert; the source may now be gone
            self._enqueue(e.dest_path, "upsert")
            self._enqueue(e.src_path, "delete")

        def on_deleted(self, e):
            # SOFT delete only -- see sync.process_delete. Nothing is removed here.
            self._enqueue(e.src_path, "delete")

    return Handler()


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
            while not _STOP.is_set():
                try:
                    _work_once(cfg, conn)
                except Exception:       # noqa: BLE001 -- the loop must survive anything
                    log.exception("worker tick failed; continuing")
                _STOP.wait(cfg.watch_poll_seconds)
    finally:
        observer.stop()
        observer.join(timeout=5)
    log.info("watcher stopped")
    return 0
