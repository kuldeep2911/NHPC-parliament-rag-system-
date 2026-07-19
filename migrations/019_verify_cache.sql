-- 019 — LLM verification verdict cache.
--
-- The LLM verify pass is non-deterministic: identical input can yield slightly different
-- verdicts run to run, so two SYNONYM queries (canonicalised to the same string) could keep
-- different candidates even though retrieval was byte-identical. Caching the verdict by
-- (canonical query, doc_key, question text) makes the same canonical query deterministic --
-- and saves repeat API calls. Additive; a cache miss just calls the LLM as before.

CREATE TABLE IF NOT EXISTS verify_cache (
    query_hash    text NOT NULL,             -- sha256 of the CANONICALISED query
    doc_key       text NOT NULL,
    sub_question_id text NOT NULL,
    verdict       text NOT NULL,             -- similar | not_similar
    reason        text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (query_hash, sub_question_id)
);

CREATE INDEX IF NOT EXISTS idx_verify_cache_created ON verify_cache (created_at);

COMMENT ON TABLE verify_cache IS
    'Deterministic cache of LLM similarity verdicts, keyed on the canonicalised query. Makes '
    'synonym-equivalent queries return the SAME verified set, and avoids repeat LLM calls.';
