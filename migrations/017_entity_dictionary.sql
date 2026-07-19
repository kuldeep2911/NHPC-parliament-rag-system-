-- 017 — self-updating entity dictionary + per-record entity links.
--
-- ADDITIVE. Nothing existing is altered. The dictionary is corpus VOCABULARY (canonical
-- entities + their surface aliases); the links table records which canonical entities each
-- sub-question mentions, from the question, the answer, or both.
--
-- WHY IN THE DB, NOT A FLAT FILE. It has to be queryable (the entity retriever joins on it),
-- updatable by the build script AND by the watcher on upload, and auditable. A file cannot
-- do the join and cannot be updated transactionally alongside the records.

-- ── canonical entities ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS entities (
    -- DETERMINISTIC canonical id: slug of the primary name, e.g. 'himachal_pradesh',
    -- 'teesta_vi', 'subansiri_lower'. Deterministic so a re-run of the build produces the
    -- SAME id and upserts rather than duplicating.
    entity_id     text PRIMARY KEY,

    canonical     text NOT NULL,              -- the display name ("Himachal Pradesh")
    entity_type   text NOT NULL,              -- project | state | scheme | organization | other

    -- provenance: seed_states | seed_projects | abbr_mining | llm | manual. Multiple
    -- sources can contribute to one entity over time; this is the FIRST/primary source.
    source        text NOT NULL DEFAULT 'manual',

    -- LLM-discovered entities are USABLE but flagged for optional review (flag, don't gate).
    needs_review  boolean NOT NULL DEFAULT false,
    confidence    text NOT NULL DEFAULT 'high',   -- high | low

    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_entities_type ON entities (entity_type);


-- ── aliases (surface variants) ─────────────────────────────────────────────
-- One canonical entity has many aliases. The alias is stored NORMALISED (lowercased,
-- collapsed whitespace) so query/index matching is a straight lookup. UNIQUE on the
-- normalised alias so 'hp' cannot map to two different entities -- an ambiguous alias is a
-- bug we want the DB to reject, not silently accept.
CREATE TABLE IF NOT EXISTS entity_aliases (
    alias_norm    text PRIMARY KEY,           -- normalised surface form, e.g. 'hp'
    alias         text NOT NULL,              -- as first seen ("H.P.")
    entity_id     text NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    source        text NOT NULL DEFAULT 'manual',
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_aliases_entity ON entity_aliases (entity_id);


-- ── per-record entity links ────────────────────────────────────────────────
-- Which canonical entities a sub-question mentions. This is what the ENTITY RETRIEVER joins
-- against -- deterministic, auditable, and (unlike matching raw text at query time) already
-- canonicalised so 'HP' and 'Himachal Pradesh' records are indistinguishable here.
CREATE TABLE IF NOT EXISTS sub_question_entities (
    sub_question_id text NOT NULL REFERENCES sub_questions(sub_question_id) ON DELETE CASCADE,
    entity_id       text NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    doc_key         text NOT NULL,            -- denormalised for the retriever's active filter
    -- where the entity was found: 'question' | 'answer' | 'both'. The question gives the
    -- topic; the answer gives the specific project names -- both matter.
    found_in        text NOT NULL DEFAULT 'question',
    PRIMARY KEY (sub_question_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_sqe_entity  ON sub_question_entities (entity_id);
CREATE INDEX IF NOT EXISTS idx_sqe_dockey  ON sub_question_entities (doc_key);


COMMENT ON TABLE entities IS
    'Canonical entity dictionary (corpus vocabulary), built by build_entities from state '
    'seed lists + NHPC project names + "Full (ABBR)" mining + offline LLM discovery. '
    'Deterministic ids; the LLM never touches live retrieval.';
