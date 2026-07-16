-- 012 — supporting documents: reference files an officer can pull into a draft.
--
-- ADDITIVE ONLY. This migration CREATES tables; it alters nothing that the Q&A pipeline
-- touches. diaries / sub_questions / answer_groups / answer_tables are not referenced here.
--
-- ⚠️ THESE TABLES ARE NOT A SEARCH INDEX. ⚠️
-- There is deliberately NO embedding column and NO tsvector column. Supporting documents
-- are draft-time context the officer SELECTS by hand -- they must never surface in the
-- hybrid retrieval that answers "has this been asked before?". The absence of those columns
-- is the structural guarantee: the dense/keyword/entity retrievers query sub_questions and
-- answer_table_rows, and cannot see a table that has no vector and no tsv.

CREATE TABLE IF NOT EXISTS supporting_documents (
    id                bigserial PRIMARY KEY,

    -- category comes from the config registry (SUPPORTING_CATEGORIES), not a hardcoded
    -- enum, so a new category is a config entry + a folder, not a schema change.
    category          text NOT NULL,

    -- Deterministic identity: '<category>/<sha256[:16]>'. Re-uploading the SAME bytes
    -- yields the same doc_key, so the upsert is a no-op rather than a duplicate -- the same
    -- idempotency discipline as the Q&A loader's doc_key.
    doc_key           text NOT NULL UNIQUE,

    display_name      text NOT NULL,
    file_path         text NOT NULL,          -- relative to the configured root; served
                                              -- ONLY through the /file realpath jail
    original_filename text,
    sha256            text NOT NULL,
    page_count        int,

    -- MANDATORY VINTAGE. A snapshot figure with no as-of date is misleading in an official
    -- reply, so the draft prompt requires it on every supporting-doc figure. as_of_date is
    -- the single date where one exists ("as on 30.06.2026"); period_label is the
    -- human-readable span, which for a multi-year digest is a RANGE ("FY 2020-21 to
    -- 2024-25") -- the specific year of a cited figure comes from the table column at draft
    -- time, not from here.
    as_of_date        date,
    period_label      text,

    -- The whole document text. These files are small (1-5 pages); we pass them WHOLE to the
    -- LLM at draft time. No chunking -- chunking would split the dense tables that are the
    -- entire point of these documents.
    document_text     text,

    -- flag, don't guess: a low-confidence table extraction is recorded, never silently
    -- trusted. A wrong financial figure in a draft is the exact failure to avoid.
    parse_flags       text[] NOT NULL DEFAULT '{}',
    needs_review      boolean NOT NULL DEFAULT false,
    raw_parse         jsonb,

    uploaded_by       text,
    uploaded_at       timestamptz NOT NULL DEFAULT now(),

    -- soft-delete + reactivation, same discipline as the Q&A watcher: an inactive document
    -- drops out of the dropdown but its row and its tables are retained.
    is_active         boolean NOT NULL DEFAULT true,
    deleted_at        timestamptz
);

CREATE INDEX IF NOT EXISTS idx_supdoc_category ON supporting_documents (category)
    WHERE is_active;
CREATE INDEX IF NOT EXISTS idx_supdoc_active   ON supporting_documents (is_active);


-- Tables mirror the Q&A answer_tables / answer_table_rows shape, so the SAME parse-layer
-- table objects serialise into them with no new representation.
CREATE TABLE IF NOT EXISTS supporting_document_tables (
    id                bigserial PRIMARY KEY,
    supporting_doc_id bigint NOT NULL REFERENCES supporting_documents(id) ON DELETE CASCADE,
    table_index       int NOT NULL,
    page              int,
    columns           jsonb,                  -- [{name, role, language}] — same as RawTable
    n_rows            int,
    -- the transposed UC-projects layout (projects as COLUMNS) is captured here so the draft
    -- prompt can tell the LLM the orientation rather than the LLM having to guess it.
    orientation       text NOT NULL DEFAULT 'rows',   -- 'rows' | 'transposed'
    extraction_confidence text NOT NULL DEFAULT 'high',
    nl_rendering      text                    -- natural-language flattening for the prompt
);

CREATE INDEX IF NOT EXISTS idx_supdoc_tables_doc
    ON supporting_document_tables (supporting_doc_id);

CREATE TABLE IF NOT EXISTS supporting_document_rows (
    id                bigserial PRIMARY KEY,
    table_id          bigint NOT NULL REFERENCES supporting_document_tables(id) ON DELETE CASCADE,
    row_index         int,
    cells             jsonb,
    row_language      text,
    nl_rendering      text
);

CREATE INDEX IF NOT EXISTS idx_supdoc_rows_table
    ON supporting_document_rows (table_id);


COMMENT ON TABLE supporting_documents IS
    'Admin-uploaded reference files (financial reports, project progress, CSR) that an '
    'officer selects at DRAFT time. NOT part of Q&A retrieval -- no embeddings, no tsv. '
    'See migration 012 header.';
