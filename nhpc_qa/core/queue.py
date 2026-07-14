"""
The watcher's DURABLE work queue (a Postgres table, not an in-memory list).

WHY DURABLE. The watcher is a long-lived service. If it restarts -- deploy, crash, reboot,
OOM -- an in-memory queue loses every pending event, and the source files behind them are
silently never ingested. Nothing looks broken; the data is simply missing. A DB-backed
queue survives the restart and the work is picked up again on the next tick.

Re-processing a path is always safe: crawl copies only what changed, parse skips a folder
that already has a parsed.json, and index UPSERTs on deterministic keys. So "at least
once" delivery is exactly what we want, and exactly-once is unnecessary.

DEDUP + DEBOUNCE. A unique index on (source_path) WHERE status IN ('pending','processing')
means rapid successive events for one folder COALESCE into a single row. `enqueue` uses
ON CONFLICT DO UPDATE to push the settle deadline out rather than inserting a second row,
so copying a 40-file session folder produces ONE job, settled once at the end -- not 40
jobs racing each other.

CLAIMING is done with SELECT ... FOR UPDATE SKIP LOCKED, so several workers (or a worker
plus a manual `nhpc watch --once`) can never claim the same job.
"""

from __future__ import annotations

import datetime as dt
import os
import socket

from nhpc_qa.core.logging import get_logger

log = get_logger("nhpc.queue")

WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"


def enqueue(conn, source_path: str, event_type: str, settle_seconds: int):
    """
    Record an event. Returns (queue_id, coalesced: bool).

    If a live job for this path already exists, its settle deadline is PUSHED OUT and the
    event counter incremented -- the job is not duplicated. That is the debounce: while
    files keep landing in a folder, the deadline keeps moving, and the folder is only
    processed once it has been quiet for `settle_seconds`.
    """
    if event_type not in ("upsert", "delete"):
        raise ValueError(f"event_type must be upsert|delete, got {event_type!r}")

    settle_until = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=settle_seconds)

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO sync_queue (source_path, event_type, settle_until)
            VALUES (%s, %s, %s)
            ON CONFLICT (source_path) WHERE status IN ('pending', 'processing')
            DO UPDATE SET
                -- a newer event resets the quiet period: keep waiting until the copy stops
                settle_until  = GREATEST(sync_queue.settle_until, EXCLUDED.settle_until),
                last_event_at = now(),
                event_count   = sync_queue.event_count + 1,
                -- a delete followed by a re-add (or vice versa) must reflect the LATEST
                -- state of the filesystem, not the first event we happened to see
                event_type    = EXCLUDED.event_type
            RETURNING id, (xmax <> 0) AS was_update
        """, (source_path, event_type, settle_until))
        qid, coalesced = cur.fetchone()
    return qid, bool(coalesced)


def claim_ready(conn, limit=1):
    """
    Claim jobs whose quiet period has elapsed.

    FOR UPDATE SKIP LOCKED: two workers can never grab the same job, and a slow job never
    blocks a fast one behind it.
    """
    with conn.cursor() as cur:
        cur.execute("""
            WITH ready AS (
                SELECT id FROM sync_queue
                WHERE status = 'pending' AND settle_until <= now()
                ORDER BY settle_until
                FOR UPDATE SKIP LOCKED
                LIMIT %s
            )
            UPDATE sync_queue q
            SET status = 'processing', claimed_by = %s, claimed_at = now(),
                attempts = q.attempts + 1
            FROM ready
            WHERE q.id = ready.id
            RETURNING q.id, q.source_path, q.event_type, q.attempts, q.event_count
        """, (limit, WORKER_ID))
        cols = ("id", "source_path", "event_type", "attempts", "event_count")
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def complete(conn, queue_id: int):
    with conn.cursor() as cur:
        cur.execute("UPDATE sync_queue SET status='done', last_error=NULL "
                    "WHERE id=%s", (queue_id,))


def fail(conn, queue_id: int, error: str, max_attempts=3):
    """
    Mark a job failed. Below max_attempts it goes back to 'pending' so the next tick
    retries it; at the limit it stays 'failed' and is left for an operator -- retrying
    forever would hide a genuinely broken document behind an infinite loop.
    """
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE sync_queue
            SET status = CASE WHEN attempts >= %s THEN 'failed' ELSE 'pending' END,
                -- back off a little before the retry
                settle_until = now() + (interval '30 seconds' * attempts),
                last_error = %s
            WHERE id = %s
            RETURNING status, attempts
        """, (max_attempts, error[:2000], queue_id))
        row = cur.fetchone()
    if row:
        status, attempts = row
        if status == "failed":
            log.error("job %s FAILED permanently after %d attempts: %s",
                      queue_id, attempts, error[:200])
        else:
            log.warning("job %s failed (attempt %d), will retry: %s",
                        queue_id, attempts, error[:200])


def recover_stale(conn, stale_after_seconds=900):
    """
    Return jobs stuck in 'processing' to 'pending'.

    A worker killed mid-job leaves its row claimed forever; nothing else will ever pick it
    up. Called at startup and periodically. This is the piece that actually makes the
    queue crash-safe -- durability alone is not enough if a crashed claim is never released.
    """
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE sync_queue
            SET status = 'pending', claimed_by = NULL, claimed_at = NULL
            WHERE status = 'processing'
              AND claimed_at < now() - (%s * interval '1 second')
            RETURNING id, source_path
        """, (stale_after_seconds,))
        rows = cur.fetchall()
    for qid, path in rows:
        log.warning("recovered stale job %s (%s) — a worker died holding it", qid, path)
    return len(rows)


def stats(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT status, count(*) FROM sync_queue GROUP BY status")
        return dict(cur.fetchall())


def log_action(conn, action: str, doc_key=None, source_path=None, detail=None,
               n_sub_questions=None):
    """Append to sync_log. Every add / soft-delete / reactivation / purge lands here, so
    'why did this record vanish from search?' is always answerable."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sync_log (action, doc_key, source_path, detail, n_sub_questions)
                VALUES (%s, %s, %s, %s, %s)
            """, (action, doc_key, source_path, detail, n_sub_questions))
    except Exception as e:      # noqa: BLE001 -- logging must never break the sync
        log.error("sync_log write failed: %s: %s", type(e).__name__, e)


# ---------------------------------------------------------------------------
# THE QUEUE KEY
# ---------------------------------------------------------------------------
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
# WORKER HEARTBEAT — liveness as a FACT, not an inference
# ---------------------------------------------------------------------------
# The API must be able to tell an admin, truthfully, whether their upload is going to be
# processed. It used to INFER that from the shape of sync_queue ("nothing claimed recently
# AND nothing overdue -> a watcher must be running"), which is wrong exactly when it
# matters most: on an idle system with an empty queue and NO watcher at all, nothing has
# been claimed (there was nothing to claim) and nothing is overdue (nothing is pending), so
# the inference concluded "healthy". It only told the truth once jobs were already piling
# up -- i.e. after the user had noticed anyway.
#
# The worker now says "I am alive" on every tick, and the API reads that. No guessing.

HEARTBEAT_STALE_SECONDS = 60      # ~4x the default 5s poll: tolerates a slow tick, not a
                                  # dead process. A single long parse holds the tick, so
                                  # the beat is written BEFORE the work, not after.


def heartbeat(conn, source_root: str = None, kind: str = "watcher"):
    """Record that this worker is alive. Called every tick of the watcher loop."""
    host, _, pid = WORKER_ID.partition(":")
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO workers (worker_id, kind, source_root, hostname, pid)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (worker_id) DO UPDATE
               SET last_seen_at = now(),
                   source_root  = EXCLUDED.source_root
        """, (WORKER_ID, kind, source_root, host, int(pid) if pid.isdigit() else None))
    conn.commit()


def worker_stopped(conn):
    """Deregister on a clean shutdown, so the API knows IMMEDIATELY rather than waiting for
    the heartbeat to go stale. A crash skips this -- which is precisely why staleness is
    the real check and this is only an optimisation."""
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM workers WHERE worker_id = %s", (WORKER_ID,))
        conn.commit()
    except Exception as e:      # noqa: BLE001 -- shutdown must never raise
        log.warning("could not deregister worker: %s", e)


def watcher_alive(conn, stale_seconds: int = HEARTBEAT_STALE_SECONDS):
    """
    Is a watcher actually running? Returns (alive: bool, detail: dict).

    Reads the heartbeat. Nothing is inferred from the queue.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT worker_id, source_root,
                   extract(epoch FROM (now() - last_seen_at))::int AS age_s
            FROM workers
            WHERE kind = 'watcher'
            ORDER BY last_seen_at DESC
            LIMIT 1
        """)
        row = cur.fetchone()

    if not row:
        return False, {"reason": "no watcher has ever registered"}
    worker_id, source_root, age = row
    if age > stale_seconds:
        return False, {"reason": f"last heartbeat was {age}s ago (stale > {stale_seconds}s)",
                       "worker_id": worker_id, "last_seen_seconds_ago": age}
    return True, {"worker_id": worker_id, "source_root": source_root,
                  "last_seen_seconds_ago": age}
