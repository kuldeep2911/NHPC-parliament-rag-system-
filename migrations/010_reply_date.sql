-- 010 — reply dates on diaries.
--
-- PURELY ADDITIVE. No existing column is touched, no row is rewritten. Every new column is
-- nullable, so the corpus stays fully loadable and fully retrievable while the backfill is
-- still running -- a document with no date yet is not broken, it is merely undated.
--
-- WHY THE RAW STRING IS STORED. reply_date_raw keeps the exact text that produced the date
-- ('to be answered on 03.08.2023'). A parsed date with no provenance is unauditable: when a
-- document turns out to be sorted into the wrong decade, the only way to tell a bad REGEX
-- from a bad SOURCE DOCUMENT is to see what was actually matched. It costs a few bytes.

ALTER TABLE diaries
    -- The canonical date used for recency sorting.
    ADD COLUMN IF NOT EXISTS reply_date        date,

    -- Both captured separately when present. In this corpus they are the same thing said
    -- two ways ('to be answered on' / 'dated'), but the schema does not assume that -- if a
    -- future session separates the answer-due date from the letter date, the distinction is
    -- already recorded rather than lost.
    ADD COLUMN IF NOT EXISTS answer_due_date   date,
    ADD COLUMN IF NOT EXISTS letter_date       date,

    -- WHERE the date came from:
    --   rule  -- regex over the subject column or the PDF text layer (the backfill)
    --   parse -- captured during parsing of a newly ingested document
    --   llm   -- the LLM fallback, used ONLY on documents the rules could not read
    --   none  -- no date found anywhere; reply_date is NULL and that is recorded, not hidden
    ADD COLUMN IF NOT EXISTS reply_date_source text,

    -- The exact matched text, for audit. See the note above.
    ADD COLUMN IF NOT EXISTS reply_date_raw    text;

-- Sorting/filtering index. NULLS LAST matches the display rule (an undated document must
-- never silently float to the top of a 'most recent first' list).
CREATE INDEX IF NOT EXISTS idx_diaries_reply_date
    ON diaries (reply_date DESC NULLS LAST);

-- Cheap sanity guard. A parliamentary reply date outside this window is a parsing error,
-- not a fact -- almost always a body-prose date ('MOU signed on 20.07.2020') that leaked
-- past the anchor, or a two-digit year read as 1923. Rejecting it at the DB boundary means
-- a bad regex cannot quietly poison the sort order.
ALTER TABLE diaries DROP CONSTRAINT IF EXISTS diaries_reply_date_sane;
ALTER TABLE diaries ADD CONSTRAINT diaries_reply_date_sane CHECK (
    reply_date IS NULL OR (reply_date >= DATE '2000-01-01'
                       AND reply_date <  DATE '2051-01-01')
);

COMMENT ON COLUMN diaries.reply_date IS
    'Date the question was to be answered in Parliament. The recency signal for sorting. '
    'NULL = not found; see reply_date_source.';
COMMENT ON COLUMN diaries.reply_date_raw IS
    'The exact text the date was parsed from, e.g. "to be answered on 03.08.2023". Kept so '
    'a wrong date can be traced to a bad regex vs a bad source document.';
