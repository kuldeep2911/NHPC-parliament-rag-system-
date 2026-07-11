"""
Browse the Phase-3 database from the terminal.

    python -m phase3.inspect_db                 # overview: tables, counts, indexes
    python -m phase3.inspect_db --schema        # full column definitions
    python -m phase3.inspect_db --doc 8773      # one diary, fully expanded
    python -m phase3.inspect_db --search "solar" --k 5    # keyword search
    python -m phase3.inspect_db --similar 8773_d --k 5    # vector similarity search
    python -m phase3.inspect_db --sql "SELECT ..."        # arbitrary read-only query

Read-only: --sql refuses anything that is not a SELECT/WITH.
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import Phase3Config, load_dotenv
from .db import connect


def _table(cur, sql, params=(), maxw=46):
    cur.execute(sql, params)
    cols = [d.name for d in cur.description]
    rows = cur.fetchall()
    if not rows:
        print("   (no rows)")
        return
    def cell(v):
        s = "" if v is None else str(v)
        s = s.replace("\n", " ")
        return s[:maxw] + ("…" if len(s) > maxw else "")
    widths = [max(len(c), *(len(cell(r[i])) for r in rows)) for i, c in enumerate(cols)]
    print("   " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols)))
    print("   " + "-+-".join("-" * w for w in widths))
    for r in rows:
        print("   " + " | ".join(cell(r[i]).ljust(widths[i]) for i in range(len(cols))))
    print(f"   ({len(rows)} row{'s' if len(rows) != 1 else ''})")


def overview(cur):
    print("=" * 78)
    print("TABLES")
    print("=" * 78)
    _table(cur, """
        SELECT relname AS table_name, n_live_tup AS rows,
               pg_size_pretty(pg_total_relation_size(relid)) AS size
        FROM pg_stat_user_tables ORDER BY relname""")

    print("\n" + "=" * 78)
    print("CORPUS AT A GLANCE")
    print("=" * 78)
    _table(cur, """
        SELECT session, house, count(*) AS diaries,
               sum(CASE WHEN needs_review THEN 1 ELSE 0 END) AS needs_review
        FROM diaries GROUP BY 1,2 ORDER BY 1,2""")

    print("\n  answer types:")
    _table(cur, "SELECT answer_type, count(*) FROM answer_groups GROUP BY 1 ORDER BY 2 DESC")

    print("\n  embeddings:")
    _table(cur, """
        SELECT coalesce(embedding_model,'(none)') AS model,
               count(*) AS rows, min(vector_dims(embedding)) AS dim
        FROM sub_questions GROUP BY 1""")

    print("\n" + "=" * 78)
    print("INDEXES")
    print("=" * 78)
    _table(cur, """
        SELECT tablename, indexname FROM pg_indexes
        WHERE schemaname='public' ORDER BY tablename, indexname""", maxw=40)


def schema(cur):
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema='public' ORDER BY table_name""")
    for (t,) in cur.fetchall():
        print("\n" + "=" * 78)
        print(t.upper())
        print("=" * 78)
        _table(cur, """
            SELECT column_name, data_type, is_nullable AS nullable,
                   coalesce(column_default,'') AS default
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name=%s
            ORDER BY ordinal_position""", (t,), maxw=34)


def one_doc(cur, qid):
    print("=" * 78)
    print(f"DIARY {qid}")
    print("=" * 78)
    cur.execute("""
        SELECT question_id, house, session, session_year, subject, starred,
               reply_format, is_nhpc_relevant, needs_review, extraction_flags,
               page_count, file_sha256, answer_file_path
        FROM diaries WHERE question_id=%s""", (qid,))
    row = cur.fetchone()
    if not row:
        print(f"  no diary with question_id={qid!r}")
        return
    cols = [d.name for d in cur.description]
    for c, v in zip(cols, row):
        print(f"  {c:22}: {v}")

    print("\n  SUB-QUESTIONS (the embedding unit)")
    _table(cur, """
        SELECT sub_question_id, part_label, answer_group_id,
               (embedding IS NOT NULL) AS embedded, embedding_model,
               left(question_text, 60) AS question
        FROM sub_questions WHERE question_id=%s ORDER BY sub_question_id""", (qid,))

    print("\n  ANSWER GROUPS (answer stored once; parts may share)")
    _table(cur, """
        SELECT answer_group_id, answers_parts, answer_type,
               left(answer_text, 54) AS answer_text
        FROM answer_groups WHERE question_id=%s ORDER BY answer_group_id""", (qid,))

    print("\n  TABLES (nested inside their answer group)")
    _table(cur, """
        SELECT t.table_id, t.answer_group_id, count(r.row_id) AS n_rows
        FROM answer_tables t LEFT JOIN answer_table_rows r USING (table_id)
        WHERE t.question_id=%s GROUP BY 1,2 ORDER BY 1""", (qid,))

    cur.execute("""SELECT t.table_id, r.cells FROM answer_tables t
                   JOIN answer_table_rows r USING (table_id)
                   WHERE t.question_id=%s ORDER BY t.table_id, r.row_index""", (qid,))
    rows = cur.fetchall()
    if rows:
        print("\n  TABLE CONTENT")
        cur_t = None
        for tid, cells in rows:
            if tid != cur_t:
                print(f"    [{tid}]")
                cur_t = tid
            print("      " + " | ".join(str(v) for v in cells.values()))

    print("\n  ANNEXURES (path capture only)")
    _table(cur, """
        SELECT ref_label, referenced_in_parts, file_present, match_confidence, file_path
        FROM annexures WHERE question_id=%s""", (qid,))


def search(cur, q, k):
    print("=" * 78)
    print(f"KEYWORD SEARCH (full-text): {q!r}")
    print("=" * 78)
    _table(cur, """
        SELECT sq.sub_question_id, d.house, d.session,
               ts_rank(sq.question_tsv, websearch_to_tsquery('english', %s)) AS rank,
               left(sq.question_text, 56) AS question
        FROM sub_questions sq JOIN diaries d USING (question_id)
        WHERE sq.question_tsv @@ websearch_to_tsquery('english', %s)
        ORDER BY rank DESC LIMIT %s""", (q, q, k))


def similar(cur, sqid, k):
    print("=" * 78)
    print(f"VECTOR SIMILARITY (cosine, via the halfvec HNSW index): {sqid}")
    print("=" * 78)
    cur.execute("SELECT question_text FROM sub_questions WHERE sub_question_id=%s", (sqid,))
    row = cur.fetchone()
    if not row:
        print(f"  no sub_question {sqid!r}")
        return
    print(f"  seed: {row[0][:70]}\n")
    _table(cur, """
        SELECT sq.sub_question_id,
               round((sq.embedding::halfvec(2048) <=>
                     (SELECT embedding::halfvec(2048) FROM sub_questions
                      WHERE sub_question_id=%s))::numeric, 4) AS cos_dist,
               left(sq.question_text, 54) AS question
        FROM sub_questions sq
        WHERE sq.embedding IS NOT NULL
        ORDER BY sq.embedding::halfvec(2048) <=>
                 (SELECT embedding::halfvec(2048) FROM sub_questions
                  WHERE sub_question_id=%s)
        LIMIT %s""", (sqid, sqid, k))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Browse the Phase-3 database")
    ap.add_argument("--schema", action="store_true", help="full column definitions")
    ap.add_argument("--doc", help="expand one diary by question_id, e.g. 8773")
    ap.add_argument("--search", help="full-text search over sub-questions")
    ap.add_argument("--similar", help="vector-similarity search from a sub_question_id")
    ap.add_argument("--sql", help="read-only SELECT")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args(argv)

    load_dotenv()
    cfg = Phase3Config()
    errs = cfg.validate(need_db=True, need_embed=False)
    if errs:
        for e in errs:
            print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 1

    with connect(cfg) as conn, conn.cursor() as cur:
        if args.sql:
            s = args.sql.strip().lstrip("(").lower()
            if not (s.startswith("select") or s.startswith("with")):
                print("refusing: --sql is read-only (SELECT/WITH only)", file=sys.stderr)
                return 1
            _table(cur, args.sql, maxw=60)
        elif args.schema:
            schema(cur)
        elif args.doc:
            one_doc(cur, args.doc)
        elif args.search:
            search(cur, args.search, args.k)
        elif args.similar:
            similar(cur, args.similar, args.k)
        else:
            overview(cur)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
