"""
Entity / metadata retriever.

Two jobs:
  1. extract entities from the query (project/state/station names) by matching against
     the entity vocabulary actually present in the corpus
     (answer_table_rows.entities, GIN-indexed, 1681 distinct values), and
  2. surface sub-questions whose DOCUMENT mentions those entities, optionally narrowed
     by metadata (house / session / is_nhpc_relevant).

ELIGIBILITY (Change 2): this retriever is only ELIGIBLE when the query actually contains
a known entity. A query like "what are the electricity dues" names no project, so entity
cannot fire -- and that must not be counted as a retriever "disagreeing". The caller
records eligible-vs-fired separately so retriever_agreement is not misleading.

FILTER vs BOOST (Change 4): normally entity acts as a FILTER (only documents mentioning
the entity). When the graph WIDENs, it relaxes to BOOST-ONLY -- the entity no longer
restricts the candidate set, it only contributes rank via RRF. That is one of the three
things that make the widened retry materially broader than the first pass.

LANGUAGE: never filters by language. Entity strings are matched case-insensitively
against the query text; `question_language` never enters any WHERE clause here.
"""

from __future__ import annotations

import re

# Junk that lands in the entities array (numbers, units) — never treat these as entities.
_JUNK = re.compile(r"^[\d.,%/\s-]*$")
_MIN_LEN = 3

# Generic words that appear in the entity column but carry no discriminating power as a
# FILTER ("Power Station" matches most of the corpus). Keeping them would make the entity
# retriever fire on almost every query and stop being a signal. They are excluded from
# EXTRACTION only -- they remain in the corpus data, untouched.
_TOO_GENERIC = {
    "power station", "power project", "hydro power", "hydroelectric project",
    "project", "power", "station", "nhpc", "government", "ministry", "state",
    "india", "total", "nil", "he project", "power house", "unit",
}

# The entity column is noisy (Docling sometimes concatenates a whole column into one
# "entity"). An entity longer than this is a mangled row, not a name.
_MAX_LEN = 60


def load_vocabulary(conn):
    """
    The distinct entity strings present in the corpus. Cached by the caller (it is a
    single cheap query, ~1.7k rows).
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT e
            FROM answer_table_rows, unnest(entities) e
            WHERE length(trim(e)) BETWEEN %s AND %s
        """, (_MIN_LEN, _MAX_LEN))
        vocab = [r[0].strip() for r in cur.fetchall()]
    return [v for v in vocab
            if v
            and not _JUNK.match(v)
            and v.lower() not in _TOO_GENERIC]


def extract_entities(query_text, vocabulary):
    """
    Entities mentioned in the query, longest-match first.

    Deliberately simple and exact (case-insensitive, word-boundary): these are proper
    nouns from the corpus itself, so a fuzzy match would invent entities that do not
    exist. Returns [] when the query names none -- which makes the retriever INELIGIBLE.
    """
    if not (query_text or "").strip():
        return []
    q = query_text.lower()
    hits = []
    for ent in vocabulary:
        e = ent.lower()
        # word-boundary match so 'Chamba' doesn't fire inside 'Chambal'
        if re.search(r"(?<!\w)" + re.escape(e) + r"(?!\w)", q):
            hits.append(ent)
    # longest first: prefer 'UT of J&K' over 'J&K' when both matched
    hits.sort(key=len, reverse=True)
    # drop entities fully contained in a longer accepted one
    out = []
    for h in hits:
        if not any(h.lower() in o.lower() and h != o for o in out):
            out.append(h)
    return out


# Match the entity against the document's TEXT (question + its answer), not only against
# table rows: only 183 of the 517 documents have any table at all, so a table-join
# retriever is blind to the other 334. Table entities still count -- they add to the hit
# score -- but they are not the only source.
#
# ILIKE over a small corpus (1914 sub-questions) is fast enough and needs no extra index;
# revisit with a trigram (pg_trgm) index if the corpus grows an order of magnitude.
# CANONICAL entity join. The query's mentions are canonicalised to entity_ids by the SAME
# dictionary matcher used at index time (see entities/dictionary.match_entities), and this
# joins the pre-computed sub_question_entities links. That is the whole fix: a record linked
# to 'himachal_pradesh' matches whether the officer typed "HP" or "Himachal Pradesh", because
# both queries canonicalise to the same id BEFORE they reach this SQL. No ILIKE, no
# per-query text scan -- a straight indexed join on entity_id.
_SQL = """
SELECT sq.sub_question_id,
       sq.doc_key,
       sq.sub_question_local,
       sq.question_text,
       count(DISTINCT sqe.entity_id) AS n_entity_hits
FROM sub_question_entities sqe
JOIN sub_questions sq ON sq.sub_question_id = sqe.sub_question_id
JOIN diaries d        ON d.doc_key = sq.doc_key
WHERE sqe.entity_id = ANY(%(entities)s::text[])
  AND d.active                       -- soft-deleted documents never appear in results
{filters}
GROUP BY sq.sub_question_id, sq.doc_key, sq.sub_question_local, sq.question_text
ORDER BY n_entity_hits DESC, sq.sub_question_id
LIMIT %(k)s
"""


def _metadata_filters(house=None, session=None, nhpc_only=False):
    clauses, params = [], {}
    if house:
        clauses.append("AND d.house = %(house)s")
        params["house"] = house
    if session:
        clauses.append("AND d.session = %(session)s")
        params["session"] = session
    if nhpc_only:
        clauses.append("AND d.is_nhpc_relevant IS TRUE")
    return "\n".join(clauses), params


def search(conn, entities, top_n, house=None, session=None, nhpc_only=False):
    """
    Sub-questions in documents that mention any of `entities`.

    Returns [] when `entities` is empty -- the retriever is INELIGIBLE, not failing.
    """
    if not entities:
        return []
    filters_sql, params = _metadata_filters(house, session, nhpc_only)
    params["entities"] = list(entities)
    params["k"] = int(top_n)

    with conn.cursor() as cur:
        cur.execute(_SQL.format(filters=filters_sql), params)
        rows = cur.fetchall()

    return [{
        "sub_question_id": r[0],
        "doc_key": r[1],
        "sub_question_local": r[2],
        "question_text": r[3],
        "score": float(r[4]),          # number of distinct matching entities
        "rank": i,
        "retriever": "entity",
    } for i, r in enumerate(rows, start=1)]
