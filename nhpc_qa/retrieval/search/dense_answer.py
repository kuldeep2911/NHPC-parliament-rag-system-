"""
Answer-side dense retriever — pgvector ANN over answer_group embeddings.

EXPERIMENTAL, config-gated. This module is only ever called when USE_ANSWER_EMBEDDINGS is
true (see nodes.hybrid_retrieve). With the flag off it is never imported into the hot path
and behaviour is unchanged.

⚠️ SAME ONE RULE AS dense.py ⚠️
The HNSW index (migration 020) was built on the HALFVEC CAST of answer_groups.embedding:

    CREATE INDEX idx_answer_groups_embedding_hnsw ON answer_groups
      USING hnsw (((embedding)::halfvec(2048)) halfvec_cosine_ops)

so a query MUST order by that exact expression or it silently full-scans:

    ORDER BY embedding::halfvec(2048) <=> %s::halfvec(2048)   -> Index Scan  ✅
    ORDER BY embedding <=> %s                                  -> Seq/Sort   ❌

WHAT THIS RETURNS, AND WHY IT IS STILL QUESTION-KEYED.
The unit we retrieve, rerank, verify and display is ALWAYS the sub_question. An answer
group can be shared by several sub-questions, so an answer-embedding hit is EXPANDED to
every active sub-question that links to that answer_group_id. Each expanded row carries the
sub_question_id, doc_key and question_text — identical shape to dense.search() — so the
fuse/rerank/verify pipeline downstream cannot tell the signal came from the answer side.
The score is the answer's cosine similarity, attributed to each of its sub-questions.

LANGUAGE: never filtered here, exactly as in dense.py. A Hindi query may match an English
answer and vice versa.

There is exactly ONE SQL template (_SQL) so the ordering discipline cannot drift.
"""

from __future__ import annotations

from nhpc_qa.retrieval.search.dense import VECTOR_DIM, _as_pgvector

# ---------------------------------------------------------------------------
# The ONLY answer-side dense SQL. The halfvec cast appears in BOTH the ORDER BY and the
# distance projection, matching dense.py. We first take the top-K nearest ANSWER GROUPS
# (this is the part the HNSW index accelerates), then expand each to its active
# sub-questions. The ORDER BY / LIMIT are on the answer_groups scan so the index is used;
# the join to sub_questions happens around that.
# ---------------------------------------------------------------------------
_SQL = """
WITH nearest AS (
    SELECT ag.answer_group_id,
           ag.doc_key,
           (ag.embedding::halfvec({dim}) <=> %(qvec)s::halfvec({dim})) AS cos_dist
    FROM answer_groups ag
    JOIN diaries d ON d.doc_key = ag.doc_key
    WHERE ag.embedding IS NOT NULL
      AND d.active                        -- soft-deleted documents never appear
    {filters}
    ORDER BY ag.embedding::halfvec({dim}) <=> %(qvec)s::halfvec({dim})
    LIMIT %(k)s
)
SELECT sq.sub_question_id,
       sq.doc_key,
       sq.sub_question_local,
       sq.question_text,
       n.cos_dist,
       n.answer_group_id
FROM nearest n
JOIN sub_questions sq ON sq.answer_group_id = n.answer_group_id
ORDER BY n.cos_dist ASC, sq.sub_question_id
"""


def build_sql(filters_sql: str = "") -> str:
    """The answer-side dense SQL, with optional extra AND-clauses (applied to the
    answer_groups scan). Exposed so the EXPLAIN test asserts the real query uses the index."""
    return _SQL.format(dim=VECTOR_DIM, filters=filters_sql)


def _metadata_filters(house=None, session=None, nhpc_only=False):
    """Optional metadata WHERE clauses + params, applied to the diaries join inside the CTE.
    Language is deliberately NOT a filter, and must never become one."""
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


def search(conn, query_vec, top_n, house=None, session=None, nhpc_only=False):
    """
    Top-N nearest ANSWER groups for an already-embedded query, expanded to their active
    sub-questions.

    `top_n` bounds the number of ANSWER GROUPS scanned (the index-accelerated step); the
    expansion to sub-questions may yield more rows than top_n when a group is shared, which
    is correct — every sub-question sharing a matched answer is a legitimate candidate.

    `query_vec` must come from embedder.embed_queries() — QUERY mode, same asymmetric-model
    reason as dense.search().

    Returns [{sub_question_id, doc_key, sub_question_local, question_text, score, rank,
    retriever='dense_answer', answer_group_id}], rank 1-based over the expanded rows in
    ascending cosine-distance order, score = cosine similarity (1 - distance).
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
        "score": 1.0 - float(r[4]),          # cosine similarity, bigger is better
        "rank": i,
        "retriever": "dense_answer",
        "answer_group_id": r[5],
    } for i, r in enumerate(rows, start=1)]


def explain(conn, query_vec, top_n=10):
    """EXPLAIN plan text for the real answer-side dense query (used by the index test)."""
    sql = "EXPLAIN (COSTS OFF) " + build_sql("")
    params = {"qvec": _as_pgvector(query_vec), "k": int(top_n)}
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return "\n".join(r[0] for r in cur.fetchall())
