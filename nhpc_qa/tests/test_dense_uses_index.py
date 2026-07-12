"""
BUILD TEST: the dense retriever must use the HNSW index.

The index was built on the HALFVEC CAST of the embedding column, because pgvector caps
an HNSW index at 2000 dimensions and this model emits 2048:

    USING hnsw (((embedding)::halfvec(2048)) halfvec_cosine_ops)

If anyone "simplifies" the cast out of the ORDER BY, the query STILL RETURNS CORRECT
RESULTS -- it just quietly stops using the index and does a full scan. That is exactly
the kind of regression that never gets noticed. This test EXPLAINs the real query
(imported from nhpc_qa.retrieval.search.dense, not a copy) and fails if the plan is not an
Index Scan on idx_sub_questions_embedding_hnsw.

    python -m nhpc_qa.tests.test_dense_uses_index
"""

from __future__ import annotations

import sys

from nhpc_qa.core.db.session import connect
from nhpc_qa.config import Settings, load_dotenv
from nhpc_qa.retrieval.search import dense

INDEX_NAME = "idx_sub_questions_embedding_hnsw"


def _short(plan: str, width=100) -> str:
    """The plan embeds the whole 2048-dim literal; elide it so the output is readable."""
    out = []
    for line in plan.splitlines():
        if len(line) > width:
            line = line[:width] + " …[vector literal elided]"
        out.append(line)
    return "\n".join(out)


def main():
    load_dotenv()
    cfg = Settings()
    failures = []

    with connect(cfg) as conn:
        # a real vector from the corpus — the plan does not depend on its contents
        with conn.cursor() as cur:
            cur.execute("SELECT embedding FROM sub_questions "
                        "WHERE embedding IS NOT NULL LIMIT 1")
            row = cur.fetchone()
            if not row:
                print("SKIP: no embeddings in the database")
                return 0
            qvec = list(row[0])

        # 1. THE REAL QUERY (the one the retriever actually runs)
        plan = dense.explain(conn, qvec, top_n=10)
        used_index = ("Index Scan" in plan and INDEX_NAME in plan)
        print("=" * 74)
        print("EXPLAIN of the REAL dense query (nhpc_qa.retrieval.search.dense.build_sql)")
        print("=" * 74)
        print(_short(plan))
        print()
        if used_index:
            print(f"PASS  uses {INDEX_NAME}")
        else:
            print(f"FAIL  did NOT use {INDEX_NAME} — the halfvec cast was probably lost")
            failures.append("dense query does not use the HNSW index")

        # 2. NEGATIVE CONTROL: the WRONG ordering must NOT use the index. This proves the
        #    test is actually discriminating and not just matching any plan.
        with conn.cursor() as cur:
            cur.execute("""
                EXPLAIN (COSTS OFF)
                SELECT sub_question_id FROM sub_questions
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT 10
            """, (dense._as_pgvector(qvec),))
            bad_plan = "\n".join(r[0] for r in cur.fetchall())
        print("\n" + "=" * 74)
        print("NEGATIVE CONTROL — plain `embedding <=> q` (no halfvec cast)")
        print("=" * 74)
        print(_short(bad_plan))
        if INDEX_NAME in bad_plan:
            print("\nFAIL  the negative control unexpectedly used the index — this test "
                  "cannot distinguish the two forms")
            failures.append("negative control used the index")
        else:
            print("\nPASS  as expected, the un-cast form does a full scan (Sort/Seq Scan) "
                  "— it returns correct rows but ignores the index")

    print("\n" + "=" * 74)
    if failures:
        for f in failures:
            print("FAILURE:", f)
        return 1
    print("ALL CHECKS PASS — dense retrieval is index-backed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
