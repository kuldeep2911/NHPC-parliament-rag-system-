"""
Answer-side dense retriever — pgvector ANN over answer_group embeddings.
(Answer ki taraf ka dense retriever — answer_group embeddings ke upar pgvector ka
nearest-neighbour search.)

EXPERIMENTAL, config-gated. This module runs ONLY when USE_ANSWER_EMBEDDINGS is true (see
nodes.hybrid_retrieve). With the flag off it is never even imported into the hot path, so
behaviour is byte-for-byte unchanged. (Yeh module SIRF tab chalta hai jab
USE_ANSWER_EMBEDDINGS on ho; flag off hai to yeh import bhi nahi hota, isliye purana
behaviour bilkul waisa ka waisa rehta hai.)

⚠️ SAME ONE RULE AS dense.py ⚠️  (dense.py wala eklauta niyam yahan bhi)
The HNSW index (migration 020) was built on the HALFVEC CAST of answer_groups.embedding
(HNSW index halfvec cast ke upar bana hai):

    CREATE INDEX idx_answer_groups_embedding_hnsw ON answer_groups
      USING hnsw (((embedding)::halfvec(2048)) halfvec_cosine_ops)

so a query MUST order by that exact expression or it silently full-scans. (Query ko usi
exact expression se ORDER BY karna PADEGA, warna woh chupke se poora table scan kar degi —
result sahi aayega par index use nahi hoga.)

    ORDER BY embedding::halfvec(2048) <=> %s::halfvec(2048)   -> Index Scan  ✅
    ORDER BY embedding <=> %s                                  -> Seq/Sort   ❌

WHAT THIS RETURNS, AND WHY IT IS STILL QUESTION-KEYED.  (Yeh kya return karta hai, aur
result phir bhi question ke hisaab se kyun hai.)
The unit we retrieve, rerank, verify and display is ALWAYS the sub_question. An answer
group can be shared by several sub-questions, so an answer-embedding hit is EXPANDED to
every active sub-question that links to that answer_group_id. (Hum hamesha sub_question hi
dikhate hain. Ek answer group ko kai sub-questions share kar sakte hain, isliye ek answer
match ko us group se juDe har active sub-question tak PHAILA dete hain.) Each expanded row
carries the sub_question_id, doc_key and question_text — identical shape to dense.search()
— so the downstream fuse/rerank/verify pipeline cannot even tell the signal came from the
answer side. (Har row ka shape dense.search() jaisa hi hota hai, isliye aage ki pipeline
ko pata bhi nahi chalta ki signal answer se aaya.) The score is the answer's cosine
similarity, attributed to each of its sub-questions.

LANGUAGE: never filtered here, exactly as in dense.py — a Hindi query may match an English
answer and vice versa. (Bhasha se kabhi filter nahi karte; Hindi query English answer se
match kar sakti hai aur ulta bhi.)

There is exactly ONE SQL template (_SQL) so the ordering discipline cannot drift. (Sirf EK
SQL template hai taaki ordering ka niyam kabhi bikhre nahi.)
"""

from __future__ import annotations

from nhpc_qa.retrieval.search.dense import VECTOR_DIM, _as_pgvector

# ---------------------------------------------------------------------------
# The ONLY answer-side dense SQL. The halfvec cast appears in BOTH the ORDER BY and the
# distance projection, matching dense.py. (Eklauti answer-side SQL. halfvec cast ORDER BY
# aur distance dono jagah aata hai — dense.py ke jaisa.)
# We first take the top-K nearest ANSWER GROUPS (this is the index-accelerated step), then
# expand each to its active sub-questions. The ORDER BY / LIMIT sit on the answer_groups
# scan so the HNSW index is used; the join to sub_questions wraps around that. (Pehle top-K
# sabse nazdeek ANSWER GROUPS lete hain — yahi step index se tez hota hai — phir har group
# ko uske active sub-questions me phailate hain. ORDER BY/LIMIT answer_groups par hain taaki
# index lage; sub_questions ka join uske bahar hota hai.)
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
    """
    Return the answer-side dense SQL, with optional extra AND-clauses spliced into the
    answer_groups scan. Exposed as a function (not just used inline) so the EXPLAIN test can
    assert the REAL query — the exact string the retriever runs — uses the index.

    (Answer-side dense SQL banake deta hai, saath me optional AND-clauses jo answer_groups
    scan me lag jaati hain. Isko alag function isliye rakha hai taaki EXPLAIN test asli query
    par check kar sake ki index lag raha hai — kisi copy par nahi.)
    """
    return _SQL.format(dim=VECTOR_DIM, filters=filters_sql)


def _metadata_filters(house=None, session=None, nhpc_only=False):
    """
    Build the optional metadata WHERE clauses + their params, applied to the diaries join
    inside the CTE. Language is deliberately NOT a filter here and must never become one.

    (Optional metadata WHERE-clauses aur unke params banata hai, jo CTE ke andar diaries join
    par lagte hain. Bhasha ko JAAN-BUJHKAR filter nahi banaya — ise kabhi filter mat banana,
    warna cross-lingual matching toot jaayegi.)
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


def search(conn, query_vec, top_n, house=None, session=None, nhpc_only=False):
    """
    Top-N nearest ANSWER groups for an already-embedded query, expanded to their active
    sub-questions. (Ek pehle-se-embed ki hui query ke liye top-N sabse nazdeek ANSWER groups,
    unke active sub-questions me phaila kar.)

    `top_n` bounds the number of ANSWER GROUPS scanned (the index-accelerated step). The
    expansion to sub-questions can yield MORE rows than top_n when a group is shared — and
    that is correct, because every sub-question sharing a matched answer is a legitimate
    candidate. (`top_n` sirf itne answer groups tak scan seemit karta hai; phailne ke baad
    rows top_n se zyada ho sakti hain — yeh sahi hai, kyun ek match hue answer ko share karne
    wala har sub-question ek valid candidate hai.)

    `query_vec` MUST come from embedder.embed_queries() — QUERY mode. The passages were
    indexed in PASSAGE mode and this model is asymmetric, so mixing the two degrades results
    (same reason as dense.search()). (`query_vec` QUERY mode se aana chahiye; model asymmetric
    hai, passage aur query mode mila diye to retrieval kharab ho jaata hai.)

    Returns a list of dicts shaped EXACTLY like dense.search() output plus answer_group_id:
    [{sub_question_id, doc_key, sub_question_local, question_text, score, rank,
    retriever='dense_answer', answer_group_id}]. rank is 1-based over the expanded rows in
    ascending cosine-distance order; score = cosine similarity (1 - distance), so bigger is
    better. (Return dense.search() jaisa hi shape, bas answer_group_id extra. score = 1 -
    distance, yaani zyada = behtar.)
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
        "score": 1.0 - float(r[4]),          # cosine similarity (bigger = better) / cosine similarity, zyada = behtar
        "rank": i,
        "retriever": "dense_answer",
        "answer_group_id": r[5],
    } for i, r in enumerate(rows, start=1)]


def explain(conn, query_vec, top_n=10):
    """
    Return the EXPLAIN plan text for the REAL answer-side dense query — used by the build
    test that fails if the halfvec cast (and hence the HNSW index) is ever dropped.

    (Asli answer-side dense query ka EXPLAIN plan return karta hai. Build test isse check
    karta hai ki halfvec cast — aur isliye HNSW index — kabhi hataya to na ho jaaye.)
    """
    sql = "EXPLAIN (COSTS OFF) " + build_sql("")
    params = {"qvec": _as_pgvector(query_vec), "k": int(top_n)}
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return "\n".join(r[0] for r in cur.fetchall())
