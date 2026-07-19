-- 018 — context-synonym groups for query expansion.
--
-- ADDITIVE. Separate from entities/entity_aliases on purpose: an entity alias is a surface
-- variant of ONE proper noun ("HP" == "Himachal Pradesh"); a concept synonym is a set of
-- DIFFERENT words meaning the same thing in this domain ("ongoing" == "under construction").
-- Synonyms rewrite the QUERY TEXT before embedding so two phrasings embed identically; they
-- do NOT feed the entity retriever (they are not entities). Keeping them in their own table
-- keeps that distinction impossible to blur.

CREATE TABLE IF NOT EXISTS concept_synonyms (
    -- normalised synonym phrase -> the canonical representative it is rewritten to.
    -- e.g. 'ongoing' -> 'under construction'
    phrase_norm   text PRIMARY KEY,
    canonical     text NOT NULL,             -- the representative phrase (dominant corpus wording)
    concept_id    text NOT NULL,             -- groups members of one concept
    source        text NOT NULL DEFAULT 'seed',   -- seed | llm | manual
    needs_review  boolean NOT NULL DEFAULT false,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_concept_syn_concept ON concept_synonyms (concept_id);

COMMENT ON TABLE concept_synonyms IS
    'Context-synonym groups (ongoing==under construction). Rewrites query text before '
    'embedding so synonymous phrasings return the same results. NOT entities -- never feeds '
    'the entity retriever.';
