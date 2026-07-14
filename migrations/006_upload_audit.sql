-- 006 — upload audit.
--
-- PURELY ADDITIVE. Phases 1-4 and the auth tables are untouched.
--
-- Two tables, because an upload has two lifetimes: the HTTP request (who uploaded what,
-- and was it accepted?) and the processing that follows it asynchronously (did it parse,
-- did it land in the index?). The second is answered by the EXISTING sync_queue and
-- sync_log -- we do not duplicate them, we join to them.

-- One row per upload REQUEST.
CREATE TABLE IF NOT EXISTS uploads (
    upload_id      text        PRIMARY KEY,       -- opaque id, returned to the admin
    actor_user_id  bigint      REFERENCES users(user_id) ON DELETE SET NULL,
    actor_email    text        NOT NULL,          -- kept even if the user is later removed

    n_files_offered  int       NOT NULL,
    n_files_accepted int       NOT NULL,
    n_files_rejected int       NOT NULL,
    bytes_accepted   bigint    NOT NULL DEFAULT 0,

    -- 'accepted' | 'rejected' | 'conflict'. A rejected upload wrote NOTHING to the source
    -- tree -- a session folder is a unit, and ingesting 39 of 40 files would produce a
    -- document whose annexure is recorded as 'referenced but unavailable' AS FACT.
    outcome        text        NOT NULL,
    reason         text,

    -- the question folders handed to queue.enqueue() -- the join back to the pipeline
    queued_folders text[]      NOT NULL DEFAULT '{}',

    ip             text,
    user_agent     text,
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_uploads_created ON uploads (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_uploads_actor   ON uploads (actor_email);


-- One row per FILE in an upload -- accepted or rejected, with the reason.
-- Rejections are recorded, never silently dropped: "why is that file not in the system?"
-- must always be answerable.
CREATE TABLE IF NOT EXISTS upload_files (
    id            bigserial   PRIMARY KEY,
    upload_id     text        NOT NULL REFERENCES uploads(upload_id) ON DELETE CASCADE,

    client_path   text        NOT NULL,   -- what the browser claimed (pre-sanitisation)
    stored_path   text,                   -- relative to the source root; NULL if rejected
    size_bytes    bigint,
    sha256        text,                   -- lets a re-upload of identical bytes be spotted

    accepted      boolean     NOT NULL,
    reason        text,                   -- why it was rejected, or 'replaced'/'skipped'
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_upload_files_upload ON upload_files (upload_id);
