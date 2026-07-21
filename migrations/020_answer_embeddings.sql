-- 020 — answer-group embeddings (ADDITIVE, experiment-gated at query time).
--
-- Adds an embedding to answer_groups so dense retrieval can OPTIONALLY search answer text
-- as well as sub-question text. This migration only creates the column + index; whether the
-- retriever uses them is controlled by USE_ANSWER_EMBEDDINGS (default FALSE). With the flag
-- off, nothing reads these columns and behaviour is byte-for-byte unchanged.
--
-- SAME DISCIPLINE AS sub_questions.embedding (migrations 001/002):
--   * column stays full-fidelity vector(2048) — the model emits 2048 dims and pgvector
--     caps an HNSW index at 2000, so the INDEX rides the halfvec(2048) cast, and every
--     query MUST order by embedding::halfvec(2048) <=> $q::halfvec(2048) to use it.
--   * embedding_model + embedding_created_at record WHICH model produced each vector, so a
--     model swap is detectable and re-embeddable (the same --stale logic as sub_questions).
--
-- WHY answer_groups (not a new table): the answer is already stored ONCE per group here,
-- and several sub-questions share one group. Embedding the group means one vector per
-- distinct answer, and an answer-hit maps back to its sub-question(s) by answer_group_id.

ALTER TABLE answer_groups
    ADD COLUMN IF NOT EXISTS embedding            vector(2048),
    ADD COLUMN IF NOT EXISTS embedding_model      text,
    ADD COLUMN IF NOT EXISTS embedding_created_at timestamptz;

-- The ANN index, on the halfvec cast — identical construction to
-- idx_sub_questions_embedding_hnsw. Same ops class, same m / ef_construction, so the two
-- indexes behave the same and the EXPLAIN test asserts the same plan shape.
CREATE INDEX IF NOT EXISTS idx_answer_groups_embedding_hnsw
    ON answer_groups USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- so --stale can find answer groups embedded by a different model cheaply
CREATE INDEX IF NOT EXISTS idx_answer_groups_embed_model ON answer_groups (embedding_model);

COMMENT ON COLUMN answer_groups.embedding IS
    'Passage-mode embedding of answer_text (same model as sub_questions.embedding). Read '
    'only when USE_ANSWER_EMBEDDINGS=true; an answer-hit maps to its sub-question(s) by '
    'answer_group_id. Index rides the halfvec(2048) cast — order by embedding::halfvec(2048).';
