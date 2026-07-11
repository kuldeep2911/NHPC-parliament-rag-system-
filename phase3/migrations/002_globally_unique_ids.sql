-- ============================================================================
-- NHPC Phase 3 — migration 002: make the primary keys GLOBALLY unique
--
-- BUG THIS FIXES (silent data loss):
--   question_id is the folder name (the parliament diary number). It is unique only
--   WITHIN a session+house, NOT globally: the same number is reused in later
--   sessions for a completely different question. e.g. diary 1894 exists as both
--       2023-jan-apr/lok_sabha/1894   and   2025-monsoon/rajya_sabha/1894
--   Migration 001 made question_id the PRIMARY KEY of diaries, so the second
--   document UPSERTed OVER the first and the earlier question vanished. The
--   Phase-2 child ids (1894_a, 1894_g1, ...) inherit the same collision.
--
--   Measured on the full corpus: 9 diaries, 25 sub_questions and 15 answer_groups
--   were being silently overwritten (517 files -> 508 rows; 1914 -> 1889).
--
-- THE FIX:
--   The true identity of a document is (session, house, question_id). Every primary key
--   is now namespaced with that document key:
--       doc_key         = '<session>/<house>/<question_id>'      e.g. 2023-jan-apr/lok_sabha/1894
--       sub_question_id = '<doc_key>#<phase2 id>'                e.g. 2023-jan-apr/lok_sabha/1894#1894_a
--   The original Phase-2 ids are PRESERVED in *_local columns, so nothing from the
--   parsed.json is lost and the JSON needs no change (Phase 2 is untouched).
--
-- This migration DROPS AND RECREATES the tables: the existing rows are already
-- lossy (9 documents were overwritten), so they must be reloaded from parsed.json
-- rather than migrated. Re-run:  python -m phase3.loader && python -m phase3.embed_runner
-- ============================================================================

DROP TABLE IF EXISTS diary_level_table_rows CASCADE;
DROP TABLE IF EXISTS diary_level_tables CASCADE;
DROP TABLE IF EXISTS answer_table_rows CASCADE;
DROP TABLE IF EXISTS answer_tables CASCADE;
DROP TABLE IF EXISTS annexures CASCADE;
DROP TABLE IF EXISTS sub_questions CASCADE;
DROP TABLE IF EXISTS answer_groups CASCADE;
DROP TABLE IF EXISTS diaries CASCADE;


CREATE TABLE diaries (
    -- '<session>/<house>/<question_id>' — globally unique, deterministic, and exactly
    -- the folder path under organized/, so it is trivially traceable back to the file.
    doc_key                text PRIMARY KEY,
    -- the parliament diary number. NOT unique across sessions — keep it as an ordinary
    -- (indexed) column for lookups, never as a key.
    question_id            text NOT NULL,
    diary_numbers          text[],
    house                  text NOT NULL,
    session                text NOT NULL,
    session_year           int,
    state                  text,
    subject                text,
    starred                boolean,
    reply_format           text,
    is_nhpc_relevant       boolean,
    document_language      text,
    layout_structure       text,
    layout_case_detected   text,
    qa_table               boolean,

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

    annexures_referenced   text[],
    annexure_content_present boolean,
    tables_index           text[],

    raw_json               jsonb NOT NULL,
    created_at             timestamptz NOT NULL DEFAULT now(),
    updated_at             timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT diaries_house_chk
        CHECK (house IN ('lok_sabha', 'rajya_sabha', 'vidhan_sabha')),
    -- the natural key: one document per (session, house, diary number)
    CONSTRAINT diaries_natural_uq UNIQUE (session, house, question_id)
);


CREATE TABLE answer_groups (
    answer_group_id  text PRIMARY KEY,          -- '<doc_key>#<phase2 answer_group_id>'
    answer_group_local text NOT NULL,           -- the Phase-2 id, e.g. '1894_g1'
    doc_key          text NOT NULL REFERENCES diaries(doc_key) ON DELETE CASCADE,
    question_id      text NOT NULL,
    answers_parts    text[],
    answer_text      text,
    answer_type      answer_type_t,
    answer_language  text,
    answer_is_table  boolean,
    answer_blocks    jsonb,
    annexure_refs    text[],
    confidence       text
);


CREATE TABLE sub_questions (
    sub_question_id  text PRIMARY KEY,          -- '<doc_key>#<phase2 sub_question_id>'
    sub_question_local text NOT NULL,           -- the Phase-2 id, e.g. '1894_a'
    doc_key          text NOT NULL REFERENCES diaries(doc_key) ON DELETE CASCADE,
    question_id      text NOT NULL,
    answer_group_id  text NOT NULL REFERENCES answer_groups(answer_group_id) ON DELETE CASCADE,
    part_label       text,
    question_text    text NOT NULL,
    question_language text,
    annexure_refs    text[],

    embedding            vector(2048),
    embedding_model      text,
    embedding_created_at timestamptz,

    question_tsv     tsvector GENERATED ALWAYS AS
                        (to_tsvector('english', coalesce(question_text, ''))) STORED
);


CREATE TABLE answer_tables (
    table_id              text PRIMARY KEY,     -- '<doc_key>#<phase2 table_id>'
    table_local           text NOT NULL,
    answer_group_id       text NOT NULL REFERENCES answer_groups(answer_group_id) ON DELETE CASCADE,
    doc_key               text NOT NULL REFERENCES diaries(doc_key) ON DELETE CASCADE,
    question_id           text NOT NULL,
    caption               text,
    table_role            text,
    answer_is_table       boolean,
    columns               jsonb,
    stitched_across_pages boolean,
    extraction_confidence text
);

CREATE TABLE answer_table_rows (
    row_id        text PRIMARY KEY,             -- '<doc_key>#<phase2 row_id>'
    row_local     text NOT NULL,
    table_id      text NOT NULL REFERENCES answer_tables(table_id) ON DELETE CASCADE,
    row_index     int,
    cells         jsonb,
    row_language  text,
    nl_rendering  text,
    entities      text[]
);


CREATE TABLE diary_level_tables (
    table_id              text PRIMARY KEY,
    table_local           text NOT NULL,
    doc_key               text NOT NULL REFERENCES diaries(doc_key) ON DELETE CASCADE,
    question_id           text NOT NULL,
    caption               text,
    table_role            text,
    answer_is_table       boolean,
    columns               jsonb,
    stitched_across_pages boolean,
    extraction_confidence text
);

CREATE TABLE diary_level_table_rows (
    row_id       text PRIMARY KEY,
    row_local    text NOT NULL,
    table_id     text NOT NULL REFERENCES diary_level_tables(table_id) ON DELETE CASCADE,
    row_index    int,
    cells        jsonb,
    row_language text,
    nl_rendering text,
    entities     text[]
);


CREATE TABLE annexures (
    annexure_id         text PRIMARY KEY,       -- '<doc_key>#<ref_label>'
    doc_key             text NOT NULL REFERENCES diaries(doc_key) ON DELETE CASCADE,
    question_id         text NOT NULL,
    ref_label           text NOT NULL,
    referenced_in_parts text[],
    file_path           text,
    file_present        boolean,
    match_confidence    text,
    CONSTRAINT annexures_uq UNIQUE (doc_key, ref_label)
);


-- ---------------------------------------------------------------------------
-- indexes (as in 001; the vector index still rides the halfvec cast because
-- pgvector caps HNSW at 2000 dims and this model emits 2048)
-- ---------------------------------------------------------------------------
CREATE INDEX idx_sub_questions_embedding_hnsw
    ON sub_questions USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX idx_sub_questions_tsv    ON sub_questions USING gin (question_tsv);

CREATE INDEX idx_diaries_question_id  ON diaries (question_id);   -- NOT unique
CREATE INDEX idx_diaries_house        ON diaries (house);
CREATE INDEX idx_diaries_session      ON diaries (session);
CREATE INDEX idx_diaries_session_year ON diaries (session_year);
CREATE INDEX idx_diaries_nhpc_rel     ON diaries (is_nhpc_relevant);
CREATE INDEX idx_diaries_needs_review ON diaries (needs_review);
CREATE INDEX idx_answer_groups_type   ON answer_groups (answer_type);

CREATE INDEX idx_sub_questions_doc    ON sub_questions (doc_key);
CREATE INDEX idx_sub_questions_qid    ON sub_questions (question_id);
CREATE INDEX idx_sub_questions_agid   ON sub_questions (answer_group_id);
CREATE INDEX idx_sub_questions_model  ON sub_questions (embedding_model);
CREATE INDEX idx_answer_groups_doc    ON answer_groups (doc_key);
CREATE INDEX idx_answer_tables_agid   ON answer_tables (answer_group_id);
CREATE INDEX idx_answer_tables_doc    ON answer_tables (doc_key);
CREATE INDEX idx_answer_table_rows_t  ON answer_table_rows (table_id);
CREATE INDEX idx_annexures_doc        ON annexures (doc_key);

CREATE INDEX idx_diaries_diary_nums   ON diaries USING gin (diary_numbers);
CREATE INDEX idx_diaries_flags        ON diaries USING gin (extraction_flags);
CREATE INDEX idx_rows_entities        ON answer_table_rows USING gin (entities);
CREATE INDEX idx_sub_questions_annex  ON sub_questions USING gin (annexure_refs);
CREATE INDEX idx_answer_groups_annex  ON answer_groups USING gin (annexure_refs);

DROP TRIGGER IF EXISTS trg_diaries_updated_at ON diaries;
CREATE TRIGGER trg_diaries_updated_at
    BEFORE UPDATE ON diaries
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
