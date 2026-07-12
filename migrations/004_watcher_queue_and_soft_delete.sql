-- ============================================================================
-- Migration 004 — incremental sync: durable work queue + SOFT DELETE
--
-- ADDITIVE ONLY. No existing table's meaning changes; retrieval behaviour is unchanged
-- for every currently-loaded record (they are all active).
-- ============================================================================


-- ---------------------------------------------------------------------------
-- sync_queue — the watcher's DURABLE work queue.
--
-- WHY A DB TABLE AND NOT AN IN-MEMORY QUEUE: the watcher is a long-lived service. If it
-- is restarted (deploy, crash, reboot) mid-processing, an in-memory queue loses every
-- pending event and the corresponding source files are silently never ingested -- the
-- worst kind of failure, because nothing looks broken. A DB-backed queue survives the
-- restart and the work is picked up again. Re-processing is safe anyway: every stage is
-- idempotent and keys are deterministic.
--
-- DEDUP: a UNIQUE index on (source_path) WHERE status IN ('pending','processing') means
-- rapid successive events for the same folder COALESCE into one row rather than piling
-- up N duplicate runs. The enqueue path uses ON CONFLICT DO UPDATE to refresh the
-- settle deadline instead of inserting again.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sync_queue (
    id              bigserial PRIMARY KEY,
    -- the QUESTION FOLDER in the original source tree (not an individual file): a
    -- session folder is copied in file by file, so settling on one file would parse a
    -- half-copied question.
    source_path     text NOT NULL,
    event_type      text NOT NULL CHECK (event_type IN ('upsert', 'delete')),

    status          text NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'processing', 'done', 'failed')),

    -- SETTLING: do not touch the folder until it has stopped changing. Every new event
    -- for the same path pushes this deadline out, so a large multi-file copy settles
    -- once, at the end, rather than being processed mid-copy.
    settle_until    timestamptz NOT NULL,
    last_event_at   timestamptz NOT NULL DEFAULT now(),
    event_count     int NOT NULL DEFAULT 1,      -- how many events coalesced into this row

    attempts        int NOT NULL DEFAULT 0,
    last_error      text,
    claimed_by      text,                        -- worker id, for visibility
    claimed_at      timestamptz,

    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

-- One live row per path: rapid events coalesce instead of queueing N times.
CREATE UNIQUE INDEX IF NOT EXISTS uq_sync_queue_live
    ON sync_queue (source_path)
    WHERE status IN ('pending', 'processing');

CREATE INDEX IF NOT EXISTS idx_sync_queue_ready
    ON sync_queue (status, settle_until);
CREATE INDEX IF NOT EXISTS idx_sync_queue_created
    ON sync_queue (created_at DESC);

DROP TRIGGER IF EXISTS trg_sync_queue_updated_at ON sync_queue;
CREATE TRIGGER trg_sync_queue_updated_at
    BEFORE UPDATE ON sync_queue
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- ---------------------------------------------------------------------------
-- SOFT DELETE on diaries.
--
-- POLICY — FLAG, DON'T ACT. A file disappearing from the source tree does NOT delete
-- anything. It marks the record inactive: it drops out of retrieval immediately, but the
-- row, its answers, its tables and its 2048-dim vector all remain.
--
-- WHY: a disappearance is very often transient -- a folder being moved, a share
-- reorganised, a network mount blipping, an officer tidying up. Hard-deleting on a
-- filesystem event would make an irreversible decision from an ambiguous signal, and the
-- embeddings alone cost real money and time to rebuild. The whole system's discipline is
-- flag-don't-act (needs_review never gates a load; a low-confidence parse is flagged, not
-- dropped) and delete is the one place where getting that wrong is UNRECOVERABLE.
--
-- REACTIVATION: if the same document reappears, we match it on the deterministic doc_key
-- (and cross-check file_sha256) and simply flip active back on -- no re-parse, no
-- re-embed. That is why file_sha256 is already stored for all 517 documents.
--
-- HARD DELETE is a SEPARATE, DELIBERATE action (`nhpc purge --older-than 30d`), never
-- triggered by a filesystem event.
-- ---------------------------------------------------------------------------
ALTER TABLE diaries ADD COLUMN IF NOT EXISTS active boolean NOT NULL DEFAULT true;
ALTER TABLE diaries ADD COLUMN IF NOT EXISTS deleted_at timestamptz;
ALTER TABLE diaries ADD COLUMN IF NOT EXISTS deleted_reason text;

-- Retrieval filters on this, so it must be indexed: every dense/keyword/entity query
-- joins diaries and excludes inactive rows.
CREATE INDEX IF NOT EXISTS idx_diaries_active ON diaries (active);
CREATE INDEX IF NOT EXISTS idx_diaries_deleted_at ON diaries (deleted_at)
    WHERE deleted_at IS NOT NULL;

-- Every existing document stays active: this migration changes no current behaviour.
UPDATE diaries SET active = true WHERE active IS NULL;


-- ---------------------------------------------------------------------------
-- sync_log — what the watcher did, and when. Append-only.
--
-- Every add, every soft-delete, every reactivation, every purge is recorded. On
-- government data, "the record quietly vanished from search" must always be answerable.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sync_log (
    id            bigserial PRIMARY KEY,
    action        text NOT NULL CHECK (action IN
                    ('added', 'updated', 'soft_deleted', 'reactivated', 'purged',
                     'failed', 'skipped')),
    doc_key       text,                 -- NULL when the event never resolved to a document
    source_path   text,
    detail        text,
    n_sub_questions int,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sync_log_created ON sync_log (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sync_log_action  ON sync_log (action);
CREATE INDEX IF NOT EXISTS idx_sync_log_doc     ON sync_log (doc_key);
