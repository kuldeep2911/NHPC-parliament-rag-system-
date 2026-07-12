"""
Dense retriever — pgvector ANN over sub_question embeddings.

⚠️ THE ONE RULE IN THIS FILE ⚠️
The HNSW index was built on the HALFVEC CAST of the embedding column:

    CREATE INDEX idx_sub_questions_embedding_hnsw ON sub_questions
      USING hnsw (((embedding)::halfvec(2048)) halfvec_cosine_ops)

(pgvector caps an HNSW index at 2000 dimensions; this model emits 2048, so the column
stays full-fidelity vector(2048) and the index rides the halfvec cast.)

A query therefore MUST order by that exact expression:

    ORDER BY embedding::halfvec(2048) <=> %s::halfvec(2048)        -> Index Scan  ✅
    ORDER BY embedding <=> %s                                       -> Sort/full scan ❌

Both return CORRECT results. Only the first uses the index -- which is precisely why the
wrong form is dangerous: it degrades silently. Verified with EXPLAIN (see
tests/test_dense_uses_index.py, which fails the build if the cast is ever dropped).

There is exactly ONE SQL template in this module (_SQL) so the ordering cannot drift.

LANGUAGE: this retriever NEVER filters by language. A Hindi query must be able to match
an English sub-question and vice versa -- cross-lingual matching is a core capability and
the reranker handles it. `question_language` never appears in a WHERE clause here.
"""

from __future__ import annotations

# The vector dim is fixed by the indexed model (llama-nemotron-embed-1b-v2 -> 2048).
# It is written into the SQL because the index expression itself is dim-qualified.
VECTOR_DIM = 2048

# ---------------------------------------------------------------------------
# The ONLY dense SQL in the system. The halfvec cast appears in BOTH the ORDER BY
# and the distance projection so the planner sees one consistent expression.
#
# Optional metadata filters are appended as AND clauses; none of them is ever a
# LANGUAGE filter (see the module docstring).
# ---------------------------------------------------------------------------
_SQL = """
SELECT sq.sub_question_id,
       sq.doc_key,
       sq.sub_question_local,
       sq.question_text,
       (sq.embedding::halfvec({dim}) <=> %(qvec)s::halfvec({dim})) AS cos_dist
FROM sub_questions sq
JOIN diaries d ON d.doc_key = sq.doc_key
WHERE sq.embedding IS NOT NULL
  AND d.active                       -- soft-deleted documents never appear in results
{filters}
ORDER BY sq.embedding::halfvec({dim}) <=> %(qvec)s::halfvec({dim})
LIMIT %(k)s
"""


def build_sql(filters_sql: str = "") -> str:
    """The dense SQL, with optional extra AND-clauses. Exposed so the EXPLAIN test can
    assert the real query (not a copy) uses the HNSW index."""
    return _SQL.format(dim=VECTOR_DIM, filters=filters_sql)


def _metadata_filters(house=None, session=None, nhpc_only=False):
    """
    Build the optional metadata WHERE clauses + params.

    NOTE: language is deliberately NOT a filter here, and must never become one.
    """
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


def _as_pgvector(vec) -> str:
    """
    Render a vector for pgvector's text input: '[0.1,-0.2,...]'.

    Must coerce each element with float(): pgvector's psycopg adapter hands back numpy
    float32, and str(list_of_np_float32) yields 'np.float32(-0.017…)', which Postgres
    rejects with InvalidTextRepresentation. This bit us in the EXPLAIN test.
    """
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def search(conn, query_vec, top_n, house=None, session=None, nhpc_only=False):
    """
    Top-N nearest sub-questions for an already-embedded query.

    `query_vec` must come from embedder.embed_queries() -- QUERY mode. The passages were
    indexed in PASSAGE mode and this model is asymmetric; using passage mode for the
    query degrades retrieval.

    Returns [{sub_question_id, doc_key, sub_question_local, question_text, score, rank}]
    with rank 1-based and score = cosine similarity (1 - distance), so bigger is better.
    """
    filters_sql, params = _metadata_filters(house, session, nhpc_only)
    sql = build_sql(filters_sql)
    params["qvec"] = _as_pgvector(query_vec)
    params["k"] = int(top_n)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [{
        "sub_question_id": r[0],
        "doc_key": r[1],
        "sub_question_local": r[2],
        "question_text": r[3],
        "score": 1.0 - float(r[4]),         # cosine similarity
        "rank": i,
        "retriever": "dense",
    } for i, r in enumerate(rows, start=1)]


def explain(conn, query_vec, top_n=10):
    """Return the EXPLAIN plan text for the real dense query (used by the index test)."""
    sql = "EXPLAIN (COSTS OFF) " + build_sql("")
    params = {"qvec": _as_pgvector(query_vec), "k": int(top_n)}
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return "\n".join(r[0] for r in cur.fetchall())
