-- 011 — collapse the date columns to ONE: reply_date.
--
-- 010 added reply_date + answer_due_date + letter_date + reply_date_source + reply_date_raw
-- on the theory that a document might carry an answer-due date and a separate letter date
-- worth distinguishing. On the real corpus it does not: they are the same fact phrased two
-- ways, and reply_date already holds whichever was found.
--
-- Verified against the live data BEFORE dropping anything:
--     rows where reply_date IS NULL but another date column has a value ..... 0
-- So no date is lost here. The 417 already-extracted dates survive untouched: this only
-- removes columns that duplicate reply_date or describe how it was obtained.
--
-- Extra columns are not free. Every one is another thing the loader must populate, the
-- assemble SQL must select, and the next person must reason about -- and three date columns
-- invite the question "which one is the real one?" on every read. One column cannot be
-- ambiguous.

ALTER TABLE diaries
    DROP COLUMN IF EXISTS answer_due_date,
    DROP COLUMN IF EXISTS letter_date,
    DROP COLUMN IF EXISTS reply_date_source,
    DROP COLUMN IF EXISTS reply_date_raw;

-- reply_date and its index survive from 010. Restated here so this file is a complete
-- description of the end state rather than a diff someone has to reconstruct.
CREATE INDEX IF NOT EXISTS idx_diaries_reply_date
    ON diaries (reply_date DESC NULLS LAST);

-- The sanity guard stays. A parliamentary reply date outside this window is an extraction
-- error, not a fact -- and it is the LLM that will be producing these now, so the guard
-- matters MORE, not less: a model that hallucinates a plausible-looking wrong date must not
-- be able to poison the sort order.
ALTER TABLE diaries DROP CONSTRAINT IF EXISTS diaries_reply_date_sane;
ALTER TABLE diaries ADD CONSTRAINT diaries_reply_date_sane CHECK (
    reply_date IS NULL OR (reply_date >= DATE '2000-01-01'
                       AND reply_date <  DATE '2051-01-01')
);

COMMENT ON COLUMN diaries.reply_date IS
    'The date the question was to be answered in Parliament. Extracted by the LLM during '
    'parsing (rule-based regex is the fallback). Used ONLY to order the displayed results '
    'most-recent-first -- it never affects retrieval. NULL = not found, shown as '
    '"date unknown" and sorted last.';
