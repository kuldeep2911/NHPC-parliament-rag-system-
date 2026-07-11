"""
Keyword retriever — Postgres full-text over sub_questions.question_tsv (GIN indexed).

The generated column is:
    question_tsv tsvector GENERATED ALWAYS AS
        (to_tsvector('english', coalesce(question_text,''))) STORED

so this is an ENGLISH-config index. Postgres ships no Hindi dictionary, so a Devanagari
query mostly falls through to the dense + rerank path -- which is exactly why keyword is
one of THREE retrievers and never the only one. (Hindi FTS is a documented follow-up:
either a Hindi dictionary/unaccent config, or a 'simple' tsvector column alongside.)

LANGUAGE: this retriever NEVER filters by language. It uses an English text-search config
to PARSE the query, which is processing -- it never adds `question_language = ...` to the
WHERE clause. A Hindi query simply gets few keyword hits; it is not excluded from the
English candidates, and the other two retrievers still surface them.
"""

from __future__ import annotations

# websearch_to_tsquery is forgiving: it accepts bare user text, quotes, OR/-, and never
# raises a syntax error the way to_tsquery does on arbitrary input.
_SQL = """
SELECT sq.sub_question_id,
       sq.doc_key,
       sq.sub_question_local,
       sq.question_text,
       ts_rank(sq.question_tsv, websearch_to_tsquery('english', %(q)s)) AS rank_score
FROM sub_questions sq
JOIN diaries d ON d.doc_key = sq.doc_key
WHERE sq.question_tsv @@ websearch_to_tsquery('english', %(q)s)
{filters}
ORDER BY rank_score DESC
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


def search(conn, query_text, top_n, house=None, session=None, nhpc_only=False):
    """Top-N by full-text rank. Returns [] when the query has no indexable English terms
    (e.g. pure Devanagari) -- that is expected, not an error."""
    if not (query_text or "").strip():
        return []
    filters_sql, params = _metadata_filters(house, session, nhpc_only)
    params["q"] = query_text
    params["k"] = int(top_n)

    with conn.cursor() as cur:
        cur.execute(_SQL.format(filters=filters_sql), params)
        rows = cur.fetchall()

    return [{
        "sub_question_id": r[0],
        "doc_key": r[1],
        "sub_question_local": r[2],
        "question_text": r[3],
        "score": float(r[4]),
        "rank": i,
        "retriever": "keyword",
    } for i, r in enumerate(rows, start=1)]
