-- ============================================================================
-- Phase 4 — migration 003: query trace, feedback, and audit
--
-- ADDITIVE ONLY. Phases 1-3 tables are not touched. These four tables are the
-- durable record of what the retrieval system did and what officers thought of it.
--
-- WHY THE TRACE IS RELATIONAL, NOT JUST A LOG: a 👎 must be debuggable to root cause.
-- feedback joins to query_results on (run_id, doc_key), and query_results records WHICH
-- retrievers surfaced that document, at what rank, its RRF score, and how far the
-- reranker moved it. So "this answer was wrong" resolves to "dense had it at rank 14,
-- keyword missed it, and the reranker promoted it 9 places" -- an actionable fact.
--
-- FEEDBACK NEVER MUTATES RANKINGS. It is captured only. Live self-mutation would make
-- the system unstable and unauditable, which is unacceptable for government data. The
-- table is shaped to be exported later as a labelled evaluation set.
-- ============================================================================


-- ---------------------------------------------------------------------------
-- query_runs — one row per query
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS query_runs (
    run_id              text PRIMARY KEY,
    query_text          text NOT NULL,
    -- detected for PROCESSING only; it never filtered the candidate set
    query_language      text,
    entities            text[],

    user_id             text,
    user_role           text,

    -- what actually happened
    retrievers_eligible text[],       -- entity is INELIGIBLE when the query names none
    retrievers_fired    text[],       -- of the eligible ones, which returned >= 1 hit
    rrf_weights         jsonb,
    rrf_k               int,
    widened             boolean NOT NULL DEFAULT false,
    widen_reason        text,         -- logged so the branch is tunable, not a black box
    rerank_enabled      boolean,
    rerank_failed       boolean NOT NULL DEFAULT false,   -- optional layer degraded
    generation_enabled  boolean NOT NULL DEFAULT false,

    -- CONFIDENCE HEURISTICS — not correctness guarantees
    top_score           double precision,
    score_gap           double precision,
    n_candidates        int,

    timings_ms          jsonb,
    errors              text[],
    created_at          timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_query_runs_created ON query_runs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_query_runs_user    ON query_runs (user_id);


-- ---------------------------------------------------------------------------
-- query_results — what was SHOWN, and how each result got there
-- (this is what makes a 👎 debuggable)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS query_results (
    run_id           text NOT NULL REFERENCES query_runs(run_id) ON DELETE CASCADE,
    -- IDENTITY IS doc_key, never question_id: a diary number is reused across sessions
    -- for a DIFFERENT question (9 of the 517 documents share a number).
    doc_key          text NOT NULL REFERENCES diaries(doc_key) ON DELETE CASCADE,
    sub_question_id  text NOT NULL REFERENCES sub_questions(sub_question_id) ON DELETE CASCADE,
    rank             int NOT NULL,

    -- which retrievers fired for THIS document, and where they placed it
    dense_rank       int,
    keyword_rank     int,
    entity_rank      int,
    retrievers       text[],
    agreement        double precision,   -- fired-retrievers that surfaced it / n_fired

    rrf_score        double precision,
    rerank_logit     double precision,   -- unbounded score, NOT a probability
    rerank_movement  int,                -- + = the cross-encoder promoted it

    PRIMARY KEY (run_id, doc_key)
);

CREATE INDEX IF NOT EXISTS idx_query_results_doc  ON query_results (doc_key);
CREATE INDEX IF NOT EXISTS idx_query_results_rank ON query_results (run_id, rank);


-- ---------------------------------------------------------------------------
-- feedback — officer verdicts. CAPTURE ONLY; never feeds back into ranking.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS feedback (
    id          bigserial PRIMARY KEY,
    run_id      text NOT NULL REFERENCES query_runs(run_id) ON DELETE CASCADE,
    -- NULL doc_key = feedback on the QUERY as a whole ("none of these helped").
    -- A non-null doc_key = feedback on one specific result.
    doc_key     text REFERENCES diaries(doc_key) ON DELETE CASCADE,
    verdict     text NOT NULL CHECK (verdict IN ('up', 'down')),
    reason      text,
    user_id     text NOT NULL,
    user_role   text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- One vote per officer per result. An officer MUST be able to CHANGE their mind
-- (👎 -> 👍), so the store UPSERTs on this constraint and updates verdict/reason/
-- timestamp -- it never throws (Change 1).
--
-- A partial unique index is required rather than a table constraint: doc_key is
-- NULLable (whole-query feedback), and in Postgres NULLs are never equal, so a plain
-- UNIQUE would let one officer file unlimited whole-query votes.
CREATE UNIQUE INDEX IF NOT EXISTS uq_feedback_result
    ON feedback (run_id, doc_key, user_id) WHERE doc_key IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_feedback_query
    ON feedback (run_id, user_id) WHERE doc_key IS NULL;

CREATE INDEX IF NOT EXISTS idx_feedback_run     ON feedback (run_id);
CREATE INDEX IF NOT EXISTS idx_feedback_doc     ON feedback (doc_key);
CREATE INDEX IF NOT EXISTS idx_feedback_verdict ON feedback (verdict);

DROP TRIGGER IF EXISTS trg_feedback_updated_at ON feedback;
CREATE TRIGGER trg_feedback_updated_at
    BEFORE UPDATE ON feedback
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- ---------------------------------------------------------------------------
-- file_access_audit — every file open, allowed or denied
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS file_access_audit (
    id             bigserial PRIMARY KEY,
    run_id         text,                       -- the query that surfaced the file (if any)
    doc_key        text NOT NULL,
    file_kind      text NOT NULL CHECK (file_kind IN ('reply', 'annexure')),
    ref_label      text,                       -- annexure label, NULL for the reply
    resolved_path  text,                       -- relative to the organized/ root
    user_id        text,
    user_role      text,
    allowed        boolean NOT NULL,           -- DENIALS ARE AUDITED TOO
    denial_reason  text,
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_file_audit_created ON file_access_audit (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_file_audit_user    ON file_access_audit (user_id);
CREATE INDEX IF NOT EXISTS idx_file_audit_doc     ON file_access_audit (doc_key);


-- ---------------------------------------------------------------------------
-- query_audit — every query issued, allowed or denied (who, what, when)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS query_audit (
    id             bigserial PRIMARY KEY,
    run_id         text,
    query_text     text,
    user_id        text,
    user_role      text,
    allowed        boolean NOT NULL,
    denial_reason  text,
    n_results      int,
    doc_keys_shown text[],                     -- WHICH documents were surfaced
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_query_audit_created ON query_audit (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_query_audit_user    ON query_audit (user_id);
