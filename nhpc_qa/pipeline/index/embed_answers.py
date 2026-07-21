"""
Generate embeddings for answer_groups.answer_text (the answer-embedding experiment).

    python -m nhpc_qa.pipeline.index.embed_answers                 # embed rows with no vector
    python -m nhpc_qa.pipeline.index.embed_answers --stale         # + rows from a DIFFERENT model
    python -m nhpc_qa.pipeline.index.embed_answers --force         # re-embed EVERYTHING
    python -m nhpc_qa.pipeline.index.embed_answers --dry-run       # report only, no API/writes
    python -m nhpc_qa.pipeline.index.embed_answers --limit 10

MIRRORS pipeline/index/embedder.py (which embeds sub_question.question_text) exactly, so the
two vector sets are comparable: SAME model, SAME PASSAGE mode, SAME fail-fast dim check, SAME
--stale/--force semantics, SAME per-batch transaction.

WHAT IS EMBEDDED: answer_groups.answer_text, for ACTIVE documents only (JOIN diaries.active),
skipping groups whose answer_text is empty/NULL (deferred / not_applicable / orphan groups
have nothing to embed). Deterministic ORDER BY answer_group_id makes the run resumable — a
killed run just re-selects the still-missing rows.

IDEMPOTENT: writes are keyed on answer_group_id; re-running only touches rows still needing a
vector. Additive experiment — this never changes sub_question embeddings or any other table.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

from nhpc_qa.config import Settings, load_dotenv
from nhpc_qa.core.db.session import connect
from nhpc_qa.core.providers.embeddings import EmbeddingError, get_embedder

# Only ACTIVE documents, and only groups with real answer text. An answer that is empty,
# whitespace, or NULL (deferred_to_ministry / not_applicable / a synthesised orphan group)
# has nothing meaningful to embed and would just add a noise vector.
_ACTIVE_NONEMPTY = ("JOIN diaries d ON d.doc_key = ag.doc_key "
                    "WHERE d.active AND coalesce(trim(ag.answer_text), '') <> ''")


def _select(cur, cfg, mode, limit):
    """Rows to embed. mode: missing | stale | all. Active + non-empty answer_text only."""
    base = f"SELECT ag.answer_group_id, ag.answer_text FROM answer_groups ag {_ACTIVE_NONEMPTY}"
    if mode == "all":
        sql, params = base + " ORDER BY ag.answer_group_id", ()
    elif mode == "stale":
        sql = (base + " AND (ag.embedding IS NULL OR ag.embedding_model IS DISTINCT FROM %s)"
                      " ORDER BY ag.answer_group_id")
        params = (cfg.embed_model,)
    else:  # missing
        sql = base + " AND ag.embedding IS NULL ORDER BY ag.answer_group_id"
        params = ()
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql, params)
    return cur.fetchall()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Embed answer_groups.answer_text (experiment)")
    ap.add_argument("--force", action="store_true", help="re-embed every active row")
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
    print(f"answer embedder | backend={cfg.embed_backend} model={cfg.embed_model} "
          f"dim={cfg.embed_dim} mode={mode}{' | DRY RUN' if args.dry_run else ''}")

    with connect(cfg) as conn:
        with conn.cursor() as cur:
            rows = _select(cur, cfg, mode, args.limit)
            cur.execute(f"SELECT count(*) FROM answer_groups ag {_ACTIVE_NONEMPTY}")
            target = cur.fetchone()[0]

        print(f"  active answer groups w/ text : {target}")
        print(f"  to embed ({mode:7})            : {len(rows)}")
        if args.dry_run or not rows:
            print("\nDRY RUN — no API calls, no writes." if args.dry_run
                  else "\nnothing to embed (all up to date)")
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
                # PASSAGE mode: we are INDEXING (same as sub_question embedding). Search uses
                # query mode. Mixing the two degrades this asymmetric model.
                vecs = embedder.embed_passages(texts)
            except EmbeddingError as e:
                print(f"  ! batch {i//bs + 1} failed: {e}", file=sys.stderr)
                failed += len(batch)
                continue

            now = datetime.now(timezone.utc)
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.executemany(
                        "UPDATE answer_groups SET embedding = %s, embedding_model = %s, "
                        "embedding_created_at = %s WHERE answer_group_id = %s",
                        [(v, cfg.embed_model, now, gid) for gid, v in zip(ids, vecs)])
            done += len(batch)
            print(f"  [{done:4}/{len(rows)}] embedded  ({cfg.embed_model})", flush=True)

        dt = time.time() - t0
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM answer_groups WHERE embedding IS NOT NULL")
            have = cur.fetchone()[0]
            cur.execute("SELECT embedding_model, count(*) FROM answer_groups "
                        "WHERE embedding IS NOT NULL GROUP BY 1")
            by_model = cur.fetchall()

    print("\n" + "=" * 56)
    print("ANSWER EMBED SUMMARY")
    print("=" * 56)
    print(f"  embedded this run : {done}")
    print(f"  failed            : {failed}")
    print(f"  with vectors now  : {have} / {target} active target")
    for m, c in by_model:
        print(f"      {c:5}  {m}")
    print(f"  elapsed           : {dt:.1f}s")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
