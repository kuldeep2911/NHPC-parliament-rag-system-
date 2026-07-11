# Phase 4 — hybrid retrieval, reranking, feedback

Officers ask a question; the system returns the **past parliamentary questions NHPC has
already answered**, with those answers and the source files. Results are **retrieved, not
generated** (drafting is optional and off by default).

```bash
python -m phase4.api.app                 # officer UI + API  (http://127.0.0.1:8099)
python -m phase4.graph.run "electricity dues owed by J&K"   # CLI
python -m phase4.tests.test_dense_uses_index               # build test (see below)
python -m phase4.scripts.measure_rrf                       # re-measure the RRF scale
```

## Three constraints that are structurally enforced

**1. Dense search MUST order by the halfvec cast.** pgvector caps an HNSW index at 2000
dimensions and the model emits 2048, so the index was built on `((embedding)::halfvec(2048))`.
A query that omits the cast **still returns correct results** — it just silently stops
using the index and does a full scan. `phase4/retrieval/dense.py` therefore contains
exactly **one** SQL template, and `tests/test_dense_uses_index.py` EXPLAINs the real query
and fails if the plan is not `Index Scan using idx_sub_questions_embedding_hnsw`. It also
runs a negative control (the un-cast form → `Sort`/`Seq Scan`) to prove the test
discriminates.

**2. `doc_key` is the identity, never `question_id`.** A diary number is reused across
sessions for a *different* question — diary 1894 is "hydro disasters" (2023 Lok Sabha) and
"renewable energy" (2025 Rajya Sabha). Fuse, dedup, assemble, feedback and audit all key on
`doc_key` = `<session>/<house>/<question_id>`.

**3. The query is embedded in QUERY mode.** Passages were indexed with
`input_type=passage`; the model is asymmetric. `embed_queries()`, never `embed_passages()`.

## Language never filters retrieval

Language is detected in node 1 for **processing only** (reporting, and the generation
prompt). It is never passed to a retriever and `question_language` appears in **no WHERE
clause** — `grep -rn "language" phase4/retrieval/` returns only comments. Cross-lingual
matching is a core capability: a Devanagari query retrieves the English document `8773_d`
at rank 1 (rerank logit +3.41), the same top hit as its English equivalent.

## The graph

```
query_process → retrieve (dense ∥ keyword ∥ entity) → fuse (RRF)
                                                        │
                              weak? → widen (ONCE) ─────┘
                                                        ▼
                                       rerank → assemble → [generate] → END
```

LangGraph is the **conductor only**. Every model call goes through the existing provider
interfaces (`phase3.embeddings`, `phase4.rerank.providers`, `phase2.providers`) and every
DB call through psycopg. No LangChain retriever, vectorstore, or LLM wrapper.

### WIDEN — what it actually does

RRF scores are **not** on a 0–1 scale. Measured on this corpus (`scripts/measure_rrf.py`):

| queries | top_score | #1−#2 gap |
|---|---|---|
| strong / entity | 0.027 – 0.085 | **≥ 0.0017** |
| vague / nonsense | 0.028 – 0.032 | **0.0004** – 0.0054 |

`top_score` **does not separate good from bad** — the ranges overlap. The **gap** does. So
`WIDEN_DELTA=0.0015` does the real work and `WIDEN_TAU=0.0150` is a low floor.

When it fires (once, and it logs why), the retry **materially broadens**: top-N × 3, the
entity retriever relaxes from *filter* to *boost-only*, and metadata filters are dropped.
Observed: 47 → 127 candidates.

## Reranker — the spec's model does not exist

`nvidia/nv-rerankqa-mistral-4b-v3` returns **404**, and `llama-3.2-nv-rerankqa-1b-v2` is
**EOL (410)** — the same fate as the embedding model. Rerankers are **not listed by
`GET /v1/models`**; they live on `ai.api.nvidia.com` under a per-model path.

In use: **`nvidia/llama-nemotron-rerank-1b-v2`**, which is strongly multilingual (it ranked
a Hindi passage above its English equivalent for a Hindi-relevant query).

The reranker is an **optional layer**: if it fails, the officer still gets the RRF ordering,
flagged `rerank_failed`.

## Security

- `/file` takes **`doc_key` + `file_kind`** (+ `ref_label`) — **never a path**. The path is
  looked up in the DB and then `realpath()`'d and proven inside the `organized/` root.
  All traversal attempts (`../../../../etc/passwd`, `../../.env`, …) return **403**.
- **RBAC fails closed**: no role, or an unknown role → 403.
- **Every query and every file open is audited, including denials.**
- Annexures are honest: a referenced-but-missing file is *"referenced but unavailable"*,
  not a dead button.

⚠️ Identity comes from `X-User-Id` / `X-User-Role`, which the **authenticating reverse
proxy** in front of this service sets. Without such a proxy those headers are self-asserted
and the RBAC is worthless — keep the bind address on `127.0.0.1` (the default).

## Feedback — captured, never acted on

A 👍/👎 (optionally per result, with a reason) is stored and **joined to the exact retrieval
decision**: which retrievers surfaced that document, at what rank, its RRF score, and how
far the reranker moved it. So a 👎 resolves to *"dense had it at rank 14, keyword missed it,
the reranker promoted it 9 places"* — actionable.

**A vote is updatable** (👎 → 👍 UPSERTs, it does not throw or duplicate), and feedback
**never mutates live rankings** — live self-mutation would be unstable and unauditable for
government data. `store.export_feedback()` emits it join-ready as a future labelled test
set. *(No evaluation layer is built; this only captures cleanly.)*

## Optional layers (both OFF by default)

- **`GENERATION_ENABLED=1`** — drafts an answer **strictly from the retrieved past answers**,
  cites `[session house diary]` for every claim, and is labelled *"DRAFT FOR OFFICER
  REVIEW"*. It refuses to invent: asked for a figure that lives only in an annexure table,
  it answered *"the specific amount is not provided in the context"*. When off, the node is
  not even wired into the graph. When on and failing, the officer still gets their results.
- **`LANGFUSE_ENABLED=1`** — one trace per query (trace id = `run_id`) with a span per node.
  When off, the SDK is **never imported**. The durable Postgres trace (`query_runs` /
  `query_results`) is the system of record either way.

## Note

Port **8080** is often reserved on Windows (WinError 10013). The default here is **8099**.
