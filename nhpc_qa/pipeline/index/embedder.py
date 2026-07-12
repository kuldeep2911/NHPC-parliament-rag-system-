"""
Generate embeddings for sub_questions.

    python -m nhpc_qa.pipeline.index.embedder                 # embed rows that have no vector
    python -m nhpc_qa.pipeline.index.embedder --stale         # + rows embedded by a DIFFERENT model
    python -m nhpc_qa.pipeline.index.embedder --force         # re-embed EVERYTHING
    python -m nhpc_qa.pipeline.index.embedder --dry-run       # show what would be embedded
    python -m nhpc_qa.pipeline.index.embedder --limit 10

EMBEDS ONLY sub_question.question_text -- parsed.json declares
embedding_unit = 'sub_question.question_text'. Answers, tables and annexures are the
display payload fetched AFTER a question matches; embedding them would blur the index.

EVERY loaded sub-question is embedded. needs_review does not skip anything.

MODEL CHANGES: vectors from different models are not comparable. Each row records the
embedding_model that produced it; --stale finds rows whose model != the configured one
and re-embeds just those. --force re-embeds all.

FAIL-FAST: the provider rejects any vector whose length != the column dim, so a model
swap cannot silently corrupt the index.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

from nhpc_qa.config import Settings, load_dotenv
from nhpc_qa.core.db.session import connect
from nhpc_qa.core.providers.embeddings import EmbeddingError, get_embedder


def _select(cur, cfg, mode, limit):
    """Rows to embed. mode: missing | stale | all"""
    if mode == "all":
        sql = ("SELECT sub_question_id, question_text FROM sub_questions "
               "ORDER BY sub_question_id")
        params = ()
    elif mode == "stale":
        sql = ("SELECT sub_question_id, question_text FROM sub_questions "
               "WHERE embedding IS NULL OR embedding_model IS DISTINCT FROM %s "
               "ORDER BY sub_question_id")
        params = (cfg.embed_model,)
    else:  # missing
        sql = ("SELECT sub_question_id, question_text FROM sub_questions "
               "WHERE embedding IS NULL ORDER BY sub_question_id")
        params = ()
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql, params)
    return cur.fetchall()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Embed sub_question.question_text")
    ap.add_argument("--force", action="store_true", help="re-embed every row")
    ap.add_argument("--stale", action="store_true",
                    help="also re-embed rows whose embedding_model != the current model")
    ap.add_argument("--dry-run", action="store_true", help="report only, no API calls/writes")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    load_dotenv()
    cfg = Settings()
    errs = cfg.validate(need_db=True, need_embed=not args.dry_run)
    if errs:
        for e in errs:
            print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 1

    mode = "all" if args.force else ("stale" if args.stale else "missing")
    print(f"Phase 3 embedder | backend={cfg.embed_backend} model={cfg.embed_model} "
          f"dim={cfg.embed_dim} input_type={cfg.embed_input_type} mode={mode}"
          f"{' | DRY RUN' if args.dry_run else ''}")

    with connect(cfg) as conn:
        with conn.cursor() as cur:
            rows = _select(cur, cfg, mode, args.limit)
            cur.execute("SELECT count(*) FROM sub_questions")
            total = cur.fetchone()[0]

        print(f"  sub_questions in db : {total}")
        print(f"  to embed ({mode:7})   : {len(rows)}")
        if args.dry_run or not rows:
            if args.dry_run:
                print("\nDRY RUN — no API calls, no writes.")
            elif not rows:
                print("\nnothing to embed (all up to date)")
            return 0

        embedder = get_embedder(cfg)
        done = failed = 0
        t0 = time.time()
        bs = max(1, cfg.embed_batch_size)

        for i in range(0, len(rows), bs):
            batch = rows[i:i + bs]
            ids = [r[0] for r in batch]
            texts = [r[1] or "" for r in batch]
            try:
                # PASSAGE mode: we are INDEXING. Phase-4 search uses query mode.
                vecs = embedder.embed_passages(texts)
            except EmbeddingError as e:
                print(f"  ! batch {i//bs + 1} failed: {e}", file=sys.stderr)
                failed += len(batch)
                continue

            now = datetime.now(timezone.utc)
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.executemany(
                        "UPDATE sub_questions SET embedding = %s, embedding_model = %s, "
                        "embedding_created_at = %s WHERE sub_question_id = %s",
                        [(v, cfg.embed_model, now, sid) for sid, v in zip(ids, vecs)])
            done += len(batch)
            print(f"  [{done:4}/{len(rows)}] embedded  ({cfg.embed_model})", flush=True)

        dt = time.time() - t0
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM sub_questions WHERE embedding IS NOT NULL")
            have = cur.fetchone()[0]
            cur.execute("SELECT embedding_model, count(*) FROM sub_questions "
                        "WHERE embedding IS NOT NULL GROUP BY 1")
            by_model = cur.fetchall()

    print("\n" + "=" * 56)
    print("EMBED SUMMARY")
    print("=" * 56)
    print(f"  embedded this run : {done}")
    print(f"  failed            : {failed}")
    print(f"  with vectors now  : {have} / {total}")
    for m, c in by_model:
        print(f"      {c:5}  {m}")
    print(f"  elapsed           : {dt:.1f}s")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
