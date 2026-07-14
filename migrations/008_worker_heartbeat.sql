-- 008 — watcher heartbeat.
--
-- WHY. The API needs to answer one question honestly: "is anything actually going to
-- process this upload?" It used to GUESS, by looking for side-effects in sync_queue:
--
--     no job claimed in the last 120s?  and nothing pending past its settle time?
--         -> assume a watcher is running
--
-- That inference is wrong in the most common case there is. On an IDLE system with an
-- empty queue and NO WATCHER AT ALL, nothing has been claimed recently (there was nothing
-- to claim) and nothing is overdue (there is nothing pending) -- so it concluded the
-- watcher was healthy. It only reported the truth when jobs happened to be stuck, i.e.
-- after the damage was already visible. A liveness check that is right only once the
-- symptom appears is not a liveness check.
--
-- A heartbeat replaces the guess with a fact: the worker writes a row on every tick, and
-- the API reads it. If the row is stale, no worker is running -- whatever the queue
-- happens to look like.

CREATE TABLE IF NOT EXISTS workers (
    worker_id    text        PRIMARY KEY,   -- '<host>:<pid>' (queue.WORKER_ID)
    kind         text        NOT NULL,      -- 'watcher' (room for others later)
    started_at   timestamptz NOT NULL DEFAULT now(),
    last_seen_at timestamptz NOT NULL DEFAULT now(),
    source_root  text,                      -- WHICH tree it is watching
    hostname     text,
    pid          int
);

-- The only query that matters: "is any watcher's heartbeat fresh?"
CREATE INDEX IF NOT EXISTS idx_workers_alive ON workers (kind, last_seen_at DESC);
