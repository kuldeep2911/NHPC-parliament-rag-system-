-- ============================================================================
-- NHPC Phase 3 — migration 001: schema, pgvector, indexes
--
-- Mirrors the Phase-2 parsed.json (schema v2.1) hierarchy relationally, using the
-- DETERMINISTIC ids from the JSON as natural primary keys (8773_a, 8773_g3,
-- 8773_g3_t1, 8773_g3_t1_r1). Verified across the corpus: 0 duplicate ids and
-- 0 dangling answer_group_id, so these are safe as PKs and make UPSERT idempotent.
--
-- POLICY: needs_review and extraction_flags are DEVELOPER WARNINGS stored for
-- audit/debugging. They are ordinary queryable columns and NEVER gate loading,
-- embedding, or retrieval. There is no quarantine. Every valid parsed.json becomes
-- an active, embedded, searchable record.
--
-- Idempotent: safe to re-run (IF NOT EXISTS / guarded DO blocks).
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- answer_type is stored and INDEXED so Phase 4 can optionally prefer substantive
-- answers at QUERY time. Storing it is not gating: all rows load regardless.
DO $$ BEGIN
    CREATE TYPE answer_type_t AS ENUM
        ('substantive', 'deferred_to_ministry', 'nil', 'not_applicable');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;


-- ---------------------------------------------------------------------------
-- diaries — one row per parsed.json (one parliamentary question folder)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS diaries (
    question_id            text PRIMARY KEY,
    diary_numbers          text[],          -- a reply may cover several diary numbers
    house                  text NOT NULL,
    session                text NOT NULL,
    session_year           int,             -- derived from session ('2020-feb-mar' -> 2020)
    state                  text,            -- always NULL in the corpus today; kept nullable
    subject                text,
    starred                boolean,         -- NULLABLE: unknown for most replies
    reply_format           text,
    is_nhpc_relevant       boolean,
    document_language      text,
    layout_structure       text,
    layout_case_detected   text,
    qa_table               boolean,

    -- provenance / file metadata (paths RELATIVE to the organized/ root)
    answer_file_path       text,
    source_answer_file     text,
    original_filename      text,
    phase1_source_path     text,
    answer_file_selection_reason text,
    file_extension         text,
    file_sha256            text,
    file_size_bytes        bigint,
    page_count             int,
    file_last_modified     timestamptz,

    -- pipeline provenance
    parsed_schema_version  text,
    parser_used            text,
    run_id                 text,
    backend                text,
    models_used            jsonb,
    page_routing           jsonb,
    embedding_unit         text,
    parsed_at              timestamptz,

    -- developer WARNINGS — queryable, never a gate
    needs_review           boolean NOT NULL DEFAULT false,
    extraction_flags       text[],

    -- annexure roll-ups kept at document level for convenience
    annexures_referenced   text[],
    annexure_content_present boolean,
    tables_index           text[],

    raw_json               jsonb NOT NULL,  -- full audit copy for reprocessing
    created_at             timestamptz NOT NULL DEFAULT now(),
    updated_at             timestamptz NOT NULL DEFAULT now(),

    -- vidhan_sabha exists in the corpus alongside lok_sabha / rajya_sabha
    CONSTRAINT diaries_house_chk
        CHECK (house IN ('lok_sabha', 'rajya_sabha', 'vidhan_sabha'))
);


-- ---------------------------------------------------------------------------
-- answer_groups — the answer, stored ONCE; several sub-questions may share one
-- (created before sub_questions: sub_questions FKs into this table)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS answer_groups (
    answer_group_id  text PRIMARY KEY,
    question_id      text NOT NULL REFERENCES diaries(question_id) ON DELETE CASCADE,
    answers_parts    text[],               -- which sub-question parts this answer covers
    answer_text      text,
    answer_type      answer_type_t,
    answer_language  text,
    answer_is_table  boolean,
    answer_blocks    jsonb,                -- empty in the corpus today; part of the v2.1 contract
    annexure_refs    text[],
    confidence       text
);


-- ---------------------------------------------------------------------------
-- sub_questions — THE EMBEDDING UNIT (embedding_unit = 'sub_question.question_text')
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sub_questions (
    sub_question_id  text PRIMARY KEY,
    question_id      text NOT NULL REFERENCES diaries(question_id) ON DELETE CASCADE,
    answer_group_id  text NOT NULL REFERENCES answer_groups(answer_group_id) ON DELETE CASCADE,
    part_label       text,
    question_text    text NOT NULL,
    question_language text,
    annexure_refs    text[],

    -- dim 2048 = the MEASURED output width of nvidia/llama-nemotron-embed-1b-v2.
    -- (The spec's llama-3.2-nv-embedqa-1b-v2 is EOL -> HTTP 410.) Vectors come back
    -- L2-normalised, so COSINE is the correct metric; the HNSW index below matches.
    -- The loader FAILS FAST if a returned vector length != this dim.
    embedding            vector(2048),
    embedding_model      text,
    embedding_created_at timestamptz,

    -- keyword half of Phase-4 hybrid retrieval. English config for now; Hindi FTS is
    -- a follow-up (Postgres has no built-in Hindi dictionary — see 'Hindi FTS' note).
    question_tsv     tsvector GENERATED ALWAYS AS
                        (to_tsvector('english', coalesce(question_text, ''))) STORED
);


-- ---------------------------------------------------------------------------
-- answer_tables / answer_table_rows — tables live INSIDE their answer group
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS answer_tables (
    table_id              text PRIMARY KEY,
    answer_group_id       text NOT NULL REFERENCES answer_groups(answer_group_id) ON DELETE CASCADE,
    question_id           text NOT NULL REFERENCES diaries(question_id) ON DELETE CASCADE,
    caption               text,
    table_role            text,
    answer_is_table       boolean,
    columns               jsonb,           -- [{name, role, language}, ...]
    stitched_across_pages boolean,
    extraction_confidence text
);

CREATE TABLE IF NOT EXISTS answer_table_rows (
    row_id        text PRIMARY KEY,
    table_id      text NOT NULL REFERENCES answer_tables(table_id) ON DELETE CASCADE,
    row_index     int,                     -- preserve source order
    cells         jsonb,                   -- {column_name: value}
    row_language  text,
    nl_rendering  text,
    entities      text[]
);


-- ---------------------------------------------------------------------------
-- diary_level_tables — a table not claimed by any answer group (rare; 0 today)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS diary_level_tables (
    table_id              text PRIMARY KEY,
    question_id           text NOT NULL REFERENCES diaries(question_id) ON DELETE CASCADE,
    caption               text,
    table_role            text,
    answer_is_table       boolean,
    columns               jsonb,
    stitched_across_pages boolean,
    extraction_confidence text
);

CREATE TABLE IF NOT EXISTS diary_level_table_rows (
    row_id       text PRIMARY KEY,
    table_id     text NOT NULL REFERENCES diary_level_tables(table_id) ON DELETE CASCADE,
    row_index    int,
    cells        jsonb,
    row_language text,
    nl_rendering text,
    entities     text[]
);


-- ---------------------------------------------------------------------------
-- annexures — referenced files, PATH CAPTURE ONLY (contents are not parsed)
-- The JSON has no id, so the PK is synthesised deterministically as
-- '<question_id>::<ref_label>' -- a serial would break idempotent re-runs.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS annexures (
    annexure_id         text PRIMARY KEY,
    question_id         text NOT NULL REFERENCES diaries(question_id) ON DELETE CASCADE,
    ref_label           text NOT NULL,
    referenced_in_parts text[],
    file_path           text,              -- relative to organized/; NULL when not found
    file_present        boolean,
    match_confidence    text,
    CONSTRAINT annexures_uq UNIQUE (question_id, ref_label)
);


-- ---------------------------------------------------------------------------
-- indexes
-- ---------------------------------------------------------------------------

-- VECTOR INDEX: HNSW + cosine, matching the model's L2-normalised output.
--
-- pgvector caps an HNSW index at 2000 dimensions, but this model emits 2048. So we
-- keep the column as full-fidelity vector(2048) -- nothing is discarded, and Phase 4
-- can rescore exactly against it -- and build the HNSW index on the halfvec CAST,
-- whose limit is 4000 dims. Half precision costs negligible recall for retrieval and
-- halves index size. Phase-4 ANN search must order by the SAME expression:
--     ORDER BY embedding::halfvec(2048) <=> $query::halfvec(2048)
-- (Alternative considered: ask the model for dimensions=1024 via Matryoshka
-- truncation. Rejected -- it throws away half the representation.)
--
-- At this corpus size an exact scan is also viable: PHASE3_VECTOR_INDEX=none skips
-- building this index entirely.
CREATE INDEX IF NOT EXISTS idx_sub_questions_embedding_hnsw
    ON sub_questions USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- KEYWORD (BM25-ish half of hybrid retrieval)
CREATE INDEX IF NOT EXISTS idx_sub_questions_tsv
    ON sub_questions USING gin (question_tsv);

-- FILTER columns (Phase-4 pre/post-filters)
CREATE INDEX IF NOT EXISTS idx_diaries_house         ON diaries (house);
CREATE INDEX IF NOT EXISTS idx_diaries_session       ON diaries (session);
CREATE INDEX IF NOT EXISTS idx_diaries_session_year  ON diaries (session_year);
CREATE INDEX IF NOT EXISTS idx_diaries_nhpc_relevant ON diaries (is_nhpc_relevant);
CREATE INDEX IF NOT EXISTS idx_diaries_needs_review  ON diaries (needs_review);
CREATE INDEX IF NOT EXISTS idx_answer_groups_type    ON answer_groups (answer_type);

-- FK-side lookups
CREATE INDEX IF NOT EXISTS idx_sub_questions_qid     ON sub_questions (question_id);
CREATE INDEX IF NOT EXISTS idx_sub_questions_agid    ON sub_questions (answer_group_id);
CREATE INDEX IF NOT EXISTS idx_sub_questions_model   ON sub_questions (embedding_model);
CREATE INDEX IF NOT EXISTS idx_answer_groups_qid     ON answer_groups (question_id);
CREATE INDEX IF NOT EXISTS idx_answer_tables_agid    ON answer_tables (answer_group_id);
CREATE INDEX IF NOT EXISTS idx_answer_tables_qid     ON answer_tables (question_id);
CREATE INDEX IF NOT EXISTS idx_answer_table_rows_tid ON answer_table_rows (table_id);
CREATE INDEX IF NOT EXISTS idx_annexures_qid         ON annexures (question_id);

-- ARRAY containment
CREATE INDEX IF NOT EXISTS idx_diaries_diary_numbers ON diaries USING gin (diary_numbers);
CREATE INDEX IF NOT EXISTS idx_diaries_flags         ON diaries USING gin (extraction_flags);
CREATE INDEX IF NOT EXISTS idx_rows_entities         ON answer_table_rows USING gin (entities);
CREATE INDEX IF NOT EXISTS idx_sub_questions_annex   ON sub_questions USING gin (annexure_refs);
CREATE INDEX IF NOT EXISTS idx_answer_groups_annex   ON answer_groups USING gin (annexure_refs);


-- ---------------------------------------------------------------------------
-- updated_at maintenance
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_diaries_updated_at ON diaries;
CREATE TRIGGER trg_diaries_updated_at
    BEFORE UPDATE ON diaries
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
