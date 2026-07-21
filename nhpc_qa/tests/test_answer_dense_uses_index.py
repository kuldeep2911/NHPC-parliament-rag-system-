"""
BUILD TEST: the answer-side dense retriever must use the HNSW index (migration 020).

Same discipline and same failure mode as tests/test_dense_uses_index.py, but for
answer_groups.embedding. The index was built on the halfvec(2048) cast; if the cast is
lost from the ORDER BY the query still returns correct rows but silently full-scans. This
EXPLAINs the REAL query (imported from nhpc_qa.retrieval.search.dense_answer, not a copy)
and fails if the plan is not an Index Scan on idx_answer_groups_embedding_hnsw.

    python -m nhpc_qa.tests.test_answer_dense_uses_index
"""

from __future__ import annotations

from nhpc_qa.core.db.session import connect
from nhpc_qa.config import Settings, load_dotenv
from nhpc_qa.retrieval.search import dense_answer

INDEX_NAME = "idx_answer_groups_embedding_hnsw"


def _short(plan: str, width=100) -> str:
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
        with conn.cursor() as cur:
            cur.execute("SELECT embedding FROM answer_groups "
                        "WHERE embedding IS NOT NULL LIMIT 1")
            row = cur.fetchone()
            if not row:
                print("SKIP: no answer_group embeddings in the database "
                      "(run the answer backfill first)")
                return 0
            qvec = list(row[0])

        # 1. THE REAL QUERY the answer retriever runs.
        plan = dense_answer.explain(conn, qvec, top_n=10)
        used_index = ("Index Scan" in plan and INDEX_NAME in plan)
        print("=" * 74)
        print("EXPLAIN of the REAL answer-dense query "
              "(nhpc_qa.retrieval.search.dense_answer.build_sql)")
        print("=" * 74)
        print(_short(plan))
        print()
        if used_index:
            print(f"PASS  uses {INDEX_NAME}")
        else:
            print(f"FAIL  did NOT use {INDEX_NAME} — the halfvec cast was probably lost")
            failures.append("answer-dense query does not use the HNSW index")

        # 2. NEGATIVE CONTROL: the un-cast ordering must NOT use the index.
        with conn.cursor() as cur:
            cur.execute("""
                EXPLAIN (COSTS OFF)
                SELECT answer_group_id FROM answer_groups
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT 10
            """, (dense_answer._as_pgvector(qvec),))
            bad_plan = "\n".join(r[0] for r in cur.fetchall())
        print("\n" + "=" * 74)
        print("NEGATIVE CONTROL — plain `embedding <=> q` (no halfvec cast)")
        print("=" * 74)
        print(_short(bad_plan))
        if INDEX_NAME in bad_plan:
            print("\nFAIL  the negative control unexpectedly used the index")
            failures.append("negative control used the index")
        else:
            print("\nPASS  as expected, the un-cast form ignores the index (full scan)")

    print("\n" + "=" * 74)
    if failures:
        for f in failures:
            print("FAILURE:", f)
        return 1
    print("ALL CHECKS PASS — answer-side dense retrieval is index-backed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
