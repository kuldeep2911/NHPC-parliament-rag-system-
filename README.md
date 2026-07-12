# NHPC Parliamentary Q&A

An officer drafting a reply to Parliament asks a question; the system returns the **past
parliamentary questions NHPC has already answered**, with those answers and the source
files. Results are **retrieved, not generated** (drafting is optional and off by default).

New source files are picked up **automatically** — drop a question folder into the source
tree and it becomes searchable on its own, with no command run by hand.

---

## Quick start

```bash
pip install -e .                 # installs the `nhpc` command

cd deploy/postgres && cp .env.postgres.example .env   # set POSTGRES_PASSWORD
docker compose up -d                                  # Postgres 16 + pgvector
cd ../..

cp .env.example .env             # set PHASE3_DB_DSN + the model API keys
nhpc migrate                     # create the schema

nhpc run                         # crawl -> parse -> index (the whole corpus)
nhpc serve                       # officer UI at http://127.0.0.1:8099
nhpc watch                       # keep the corpus in sync with the source tree
```

## The one command

```bash
nhpc run                         # full pipeline: crawl -> parse -> index (+ embeddings)
nhpc run --stages index          # only the DB load + embedding generation
nhpc run --stages parse,index    # a subset (always run in canonical order)
nhpc run --from parse            # from a stage to the end
nhpc run --only 8773 --force     # one question folder, re-processed
nhpc run --dry-run               # validate and report, write nothing

nhpc serve                       # API + officer UI
nhpc watch                       # source watcher (incremental sync)
nhpc query "electricity dues owed by J&K"
nhpc migrate --status
nhpc inspect --doc 8773
nhpc purge --older-than 30d      # PERMANENT removal — separate and deliberate
```

Every stage is **independently runnable** and **idempotent** (deterministic ids, upsert by
primary key), so a crash mid-run is fixed by re-running the same command: each stage picks
up where it left off, and redoing finished work is a no-op rather than a duplicate.

## Structure

```
nhpc_qa/
  config/        ONE Settings object (every env var name unchanged)
  core/
    providers/   embedder, reranker, llm, parser — one registry, config-only backends
    db/          session + migrations
    trace/       Postgres trace + Langfuse (config-gated, off by default)
    logging/     structured, text or JSON
  pipeline/
    crawl/       source tree -> organized/          (READ-ONLY on source)
    parse/       organized/ -> parsed.json          (Docling + LLM span extraction)
    index/       parsed.json -> Postgres + pgvector
    orchestrator.py
  retrieval/     LangGraph: hybrid search -> RRF -> rerank -> assemble -> [draft]
  api/           /query, /file, /feedback + officer UI + RBAC + audit
  watcher/       durable queue, settling, soft-delete, purge
  cli.py
migrations/      one lineage: 001 .. 004
```

The dependency direction is `config -> core -> pipeline -> retrieval -> api`, and
`nhpc_qa/tests/test_layering.py` fails the build if anything ever points left.

## Incremental sync (`nhpc watch`)

### Adding

A new file or folder appearing in the source is **not processed immediately**. The
affected **question folder** is queued with a quiet period (`WATCH_SETTLE_SECONDS`,
default 10s), and every further event **pushes the deadline out**. Only once the folder has
stopped changing is it processed.

That settling matters: a question folder is copied in file by file. Parsing it mid-copy
would read a reply whose annexure has not landed yet and record *"referenced but
unavailable"* as fact. Copying a 40-file session folder produces **one** job, not forty.

Then only the affected slice runs through crawl → parse → index, and the document is
immediately retrievable.

The queue is a **Postgres table, not an in-memory list**. If the watcher is restarted mid-
processing, nothing is lost: pending events are still queued, and a job left claimed by the
killed process is released on startup. Re-processing is idempotent, so "at least once"
delivery is harmless.

### Deleting — soft by default, never a silent hard delete

A file disappearing from the source **does not delete anything**. The record is marked
inactive: it **drops out of retrieval immediately**, but the row, its answers, its tables
and its 2048-dim vectors all remain.

A disappearance is an **ambiguous signal** — a folder moved, a share reorganised, a mount
blipping, an officer tidying up all look identical from here. Acting irreversibly on an
ambiguous signal is how data is lost, and the embeddings alone cost real time and money to
rebuild. The system's discipline everywhere is *flag, don't act*; delete is the one place
where getting that wrong cannot be undone.

If the folder **reappears**, the record **reactivates** — matched on the deterministic
`doc_key`, so there is no re-parse and no re-embed.

**Hard removal is a separate, deliberate command** and is never reached from a filesystem
event:

```bash
nhpc purge --dry-run             # what WOULD go
nhpc purge --older-than 30d      # asks for confirmation; cascades; no undo
```

Every add, soft-delete, reactivation and purge is written to `sync_log`, so *"why did this
record vanish from search?"* is always answerable.

## Running as a service (on-prem)

```bash
sudo cp deploy/systemd/nhpc-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nhpc-api nhpc-watch
journalctl -u nhpc-watch -f
```

The units are hardened with `ProtectSystem=strict`: the watcher is granted write access
only to `organized/`, and **the source tree is deliberately not writable** — "read-only on
source" is enforced by the OS, not just by convention in the code.

⚠️ The API trusts `X-User-Id` / `X-User-Role`, which the **authenticating reverse proxy** in
front of it sets. It binds to `127.0.0.1` only. Exposed directly, those headers would be
self-asserted and the RBAC worthless.

## Things that will bite you if you change them

**Dense search must order by the halfvec cast.** pgvector caps an HNSW index at 2000
dimensions and the embedding model emits 2048, so the index was built on
`((embedding)::halfvec(2048))`. A query that drops the cast **still returns correct
results** — it just silently stops using the index and does a full scan.
`nhpc_qa/tests/test_dense_uses_index.py` EXPLAINs the real query and fails if the plan is
not an index scan (with a negative control proving the test discriminates).

**`doc_key` is the identity, never `question_id`.** A diary number is reused across sessions
for a *different* question — diary 1894 is "hydro disasters" (2023 Lok Sabha) and
"renewable energy" (2025 Rajya Sabha). Keying on the number silently destroys one of them.

**Queries are embedded in QUERY mode**, passages in PASSAGE mode. The model is asymmetric.

**Language never filters retrieval.** It is detected for processing only. A Devanagari query
must be able to match an English answer — and does.

## Model notes

The models the original spec named are **gone**: `llama-3.2-nv-embedqa-1b-v2` (embedding)
and `nv-rerankqa-mistral-4b-v3` (reranker) return 410/404. In use:

| | model | note |
|---|---|---|
| embedding | `nvidia/llama-nemotron-embed-1b-v2` | dim 2048, L2-normalised → cosine |
| reranker | `nvidia/llama-nemotron-rerank-1b-v2` | strongly multilingual |
| extraction LLM | Gemini 2.5 Flash (build) / Qwen3 14B via Ollama (deploy) | |

Every backend swaps to a self-hosted/on-prem one by **config alone** — that is what the
provider seam is for.
