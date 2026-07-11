# Phase 3 — Postgres + pgvector storage & embeddings

Loads every Phase-2 `parsed.json` into a normalized Postgres database and embeds
`sub_question.question_text` for retrieval. Phase 2 is untouched.

## Policy: everything loads

`needs_review` and `extraction_flags` are **developer warnings**, not correctness gates.
They are stored, indexed and reported — but they never exclude a record from loading,
embedding, or retrieval. **There is no quarantine.** Every valid `parsed.json` becomes an
active, embedded, searchable record.

## Quick start

```bash
# 1. database (Postgres 16 + pgvector, pinned)
cd deploy/postgres && cp .env.postgres.example .env   # set POSTGRES_PASSWORD
docker compose up -d

# 2. set PHASE3_DB_DSN + NVIDIA_EMBED_API_KEY in the project-root .env

# 3. schema
python -m phase3.migrate                 # --status to inspect

# 4. load  (READ-ONLY on parsed.json; idempotent; per-document transaction)
python -m phase3.loader --dry-run        # validate + report, zero writes
python -m phase3.loader                  # load everything
python -m phase3.loader --only 8773      # or a subset; --limit N; --force

# 5. embed
python -m phase3.embed_runner            # rows with no vector
python -m phase3.embed_runner --stale    # + rows embedded by a different model
python -m phase3.embed_runner --force    # re-embed everything
```

Reports (JSON + CSV) land in `phase3/_reports/`.

## Embedding model — read this before changing it

The model the original spec named, `nvidia/llama-3.2-nv-embedqa-1b-v2`, **reached end of
life on 2026-05-18** and now returns **HTTP 410 Gone**. Phase 3 uses its live successor,
`nvidia/llama-nemotron-embed-1b-v2`. Measured, not assumed:

| property | value |
|---|---|
| dimension | **2048** |
| output | L2-normalised → **cosine** is the correct metric |
| Devanagari | embeds correctly |
| `input_type` | `passage` (indexing) / `query` (Phase-4 search) |

**Vectors from different models are not comparable.** Every row records the
`embedding_model` that produced it; `--stale` finds rows whose model differs from the
configured one and re-embeds only those. A vector whose length ≠ the column dim is
rejected before any write (fail-fast), so a model swap can never silently corrupt the
index.

### The 2048-dim / HNSW catch

pgvector caps an **HNSW index** at 2000 dimensions, but this model emits 2048. So:

- the column stays **`vector(2048)`** — full fidelity, nothing discarded, exact rescoring
  stays possible;
- the HNSW index is built on the **`halfvec` cast** (limit 4000 dims), which costs
  negligible recall and halves index size.

**Phase 4 must search with the same expression**, or the index won't be used:

```sql
SELECT sub_question_id
FROM sub_questions
ORDER BY embedding::halfvec(2048) <=> %(query_vec)s::halfvec(2048)
LIMIT 20;
```

Set `PHASE3_VECTOR_INDEX=none` to rely on exact scan instead (viable at this corpus size).

## Embedding backends (config-only switch)

| `EMBED_BACKEND` | what | when |
|---|---|---|
| `nvidia_nim_api` | NVIDIA-hosted NIM. Text **leaves the network**. | dev / now |
| `nvidia_selfhosted` | On-prem NIM container or HF weights. Nothing leaves. | server |

Same input/output contract, so switching changes no code. For the server, run the NIM
container and set `EMBED_SELFHOSTED_URL`; the sentence-transformers path is documented in
`embeddings.py` (needs the GPU box to validate — re-measure dim there; the fail-fast check
will catch a mismatch).

## Schema

`diaries` → `answer_groups` → `sub_questions` (the embedding unit), plus `answer_tables` /
`answer_table_rows` (tables live **inside** their answer group), `annexures`,
`diary_level_tables`. Deterministic Phase-2 IDs (`8773_a`, `8773_g3`, `8773_g3_t1`,
`8773_g3_t1_r1`) are the natural primary keys — verified globally unique with zero
dangling links — which is what makes UPSERT idempotent: **re-running updates in place and
never duplicates**.

`diaries.raw_json` keeps the full document for audit/reprocessing.

## Scope note

At the time of writing, **2 of 15 sessions are parsed** (73 documents). The loader picks up
the rest automatically as Phase 2 processes them — no schema change needed.
