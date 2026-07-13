# NHPC Parliamentary Q&A — Engineering Handoff

**Read this end to end before changing anything.** It explains what the system is, how each
stage works, why the non-obvious decisions were made, and — in the last section — the
traps that will silently corrupt results if you undo them.

Status: working, on `main`, 12,975 lines of Python. Corpus loaded: **517 documents, 1,914
sub-questions, all embedded.** Retrieval, reranking, feedback, RBAC/audit and automatic
incremental sync are all live. Answer *generation* exists but is **off by default**.

---

## 1. What problem this solves

An NHPC officer is handed a fresh parliamentary question and must draft a reply. Somewhere
in years of past sessions, NHPC has probably answered something very close to it — and the
reply that was actually filed is the best possible starting point, because it was cleared
and submitted to Parliament.

Finding it means remembering which session, opening a folder of PDFs and DOCXs, and
reading. This system replaces that with a search box.

**The core promise: results are RETRIEVED, not GENERATED.** Every answer shown is a real
answer NHPC actually gave, with the source file one click away. The system does not
paraphrase, summarise, or invent. An officer must be able to trust that what they are
reading is what was filed. That constraint shapes every design decision below.

---

## 2. The shape of the data

A parliamentary question is not one question. Diary no. 8773 is a *document* containing
sub-questions (a), (b), (c)… and NHPC's answers — and **one answer often covers several
sub-questions at once** ("(a) and (b): NHPC is executing…"). Some answers are tables. Some
reference annexures that may or may not exist on disk.

Four things a reply can say, and the UI badges each one, because they mean very different
things to an officer:

| `answer_type` | count | meaning |
|---|---|---|
| `substantive` | 965 | NHPC actually answered |
| `deferred_to_ministry` | 221 | "May be replied by Ministry of Power" |
| `nil` | 96 | "Information may be treated as Nil" |
| `not_applicable` | 64 | "Does not pertain to NHPC" |

Corpus today: **15 sessions**, **517 documents** (269 Rajya Sabha, 234 Lok Sabha, 14 Vidhan
Sabha), **1,914 sub-questions**, 1,346 answer groups, 475 answer tables, 176 annexures.
Questions and answers appear in **English and Hindi, often mixed in the same document.**

### `doc_key` — the identity, and the bug that taught us

> **`doc_key = '<session>/<house>/<question_id>'`. Never key on `question_id` alone.**

A diary number is **reused across sessions for a completely different question**. Diary 1894
is "hydro disasters" in 2023 Lok Sabha and "renewable energy" in 2025 Rajya Sabha. Of 517
documents, only **508 have distinct diary numbers — 9 collide.**

The first schema used `question_id` as the primary key. The upsert silently *overwrote* the
colliding documents: 9 diaries, 25 sub-questions and 15 answer groups were destroyed. It
was caught only because a row count came out 1,889 instead of 1,914. Migration `002`
introduced `doc_key`, which is now the identity from parse through to the UI.

---

## 3. Directory structure

```
nhpc_qa/
  config/        ONE Settings object. 78 env vars, one place.
  core/
    providers/   embedder, reranker, LLM, parser — the swap seam (see §9)
    db/          connection + migration runner
    trace/       Postgres query trace + Langfuse (config-gated, OFF)
    logging/      structured, text or JSON
  pipeline/
    crawl/       source tree  -> organized/        (READ-ONLY on source)
    parse/       organized/   -> parsed.json       (Docling + LLM span extraction)
    index/       parsed.json  -> Postgres + pgvector
    orchestrator.py
  retrieval/
    search/      dense.py, keyword.py, entity.py, fuse.py
    graph/       LangGraph nodes — the query pipeline
    feedback/    thumbs + durable run trace
    generation/  optional draft (OFF by default)
  api/           /query /file /feedback + officer UI + RBAC + audit
  watcher/       durable queue, settling, soft-delete, purge
  cli.py         nhpc run | serve | watch | query | migrate | inspect | purge
  tests/         test_dense_uses_index.py, test_layering.py
migrations/      one lineage: 001..004
deploy/          postgres + langfuse compose, systemd units
```

**The dependency direction is `config -> core -> pipeline -> retrieval -> api`.**
`tests/test_layering.py` fails the build if anything ever points left. It earned its keep
immediately: it caught config modules sitting inside `pipeline/`, which made the bottom
layer import upward.

---

## 4. The ingest pipeline (`nhpc run`)

Three stages, each **independently runnable and idempotent** (deterministic ids, upsert by
key), so a crash mid-run is fixed by re-running the same command.

```
source tree  ──crawl──>  organized/  ──parse──>  parsed.json  ──index──>  Postgres + pgvector
```

### 4.1 Crawl — normalise the mess

Real files arrive as `LS 8773 dated 12.03.2020 (reply).pdf`, `Annexure-I.docx`, scanned
images, nested folders. The crawler derives `session` and `house` **from the path below
`--source`** and produces a clean `organized/<session>/<house>/<question_id>/` per document.

> ⚠️ **The crawler must run from the source ROOT**, not a session subfolder. It infers
> session/house from the relative path, so scoping it to a subfolder silently produces
> nothing.

It is **strictly read-only on the source tree**. The systemd unit enforces this at the OS
level (`ProtectSystem=strict`, and `organized/` is the *only* writable path) — read-only-on-
source is not merely a convention in the code.

### 4.2 Parse — the hard part

Docling converts PDF/DOCX (with OCR for scans) into text + tables. Then an LLM finds the
questions and answers. **This is where most of the engineering pain lived**, and the final
design is deliberately narrow:

> **The LLM returns only LINE NUMBERS, never text.**
>
> ```json
> { "question_lines": [12, 14], "answer_lines": [31, 36], "part_label": "(a)" }
> ```
>
> We number every line of the document, ask the model which line spans are the question and
> which are the answer, and then **slice the text ourselves**.

This is the single most important decision in the parser. The model **cannot invent,
paraphrase, drop, or reorder a word**, because it never emits any words — only integers.
Two sub-questions sharing an answer is expressed as *identical `answer_lines`*, which falls
out for free.

**Why not regex?** The first version keyed on English opener phrases ("whether the
Government…"). It was tuned on the 2020 session and **broke on 2021**, where questions
opened with "the contingent liability…". 7 of 19 files were wrong. There are 20+ sessions
and each formats slightly differently. Per-session rules do not scale — that is precisely
the judgement an LLM is for.

**Verification is structural and universal**, in `parse/spans.py`. We reject (never patch)
on: spans out of range, inverted spans, a question starting on an answer marker, overlapping
questions, duplicate part labels, and — the strongest one — **the coverage invariant: every
`Comment:` / `Answer:` block in the document must be claimed by some question.** An unclaimed
answer block means we dropped something, so we reject and fall back.

> **A hard-won warning, written into the code as a comment so it does not come back:**
> *"an answer must follow its question" is NOT an invariant.* I added that check; the LLM
> refused it three times running — **because the LLM was right and I was wrong.** Diary 5341
> answers question (a) with a block printed *before* the follow-up. The check was removed.

**The recurring lesson of this project: the LLM was right; the bugs were in my
post-processing.** A deterministic count gatekeeper, a validation-order bug, a lossy
text-equality regrouping, and a phrase-list table matcher each corrupted correct model
output. Give the model clean input, have it return *indices*, verify with universal
structural invariants, and reject rather than emit something wrong.

**Two subtle parse bugs worth knowing about**, both fixed:
- *Grouping merged answers by TEXT.* Diary 11800 has three separate "Does not pertain to
  NHPC." blocks. Text-equality collapsed them into one group. Grouping now keys on **span
  identity (location)**, never on the answer string.
- *Collapsed table rows.* `_stitch_wrapped_rows` treated any non-numeric first cell as a
  continuation of the previous row, so text-label tables collapsed into a single row (diary
  8773, 6 tables). A continuation has an **empty** first cell.

### 4.3 Index — load and embed

`parsed.json` → Postgres. **Everything loads. There is no quarantine.** `needs_review` and
`extraction_flags` are **developer warnings, not correctness gates** — a flagged document is
still fully retrievable, and the UI shows the flag rather than hiding the record.

Embeddings are generated for **sub-question text only** (that is what an officer searches
against), with `input_type=passage`.

---

## 5. The database

16 tables. The ones that matter:

| table | rows | purpose |
|---|---|---|
| `diaries` | 517 | one document. PK `doc_key`. Has `active` / `deleted_at` (soft delete) |
| `sub_questions` | 1,914 | one (a)/(b)/(c). **Holds the `embedding vector(2048)` and `question_tsv`** |
| `answer_groups` | 1,346 | an answer + which parts it covers |
| `answer_tables` / `answer_table_rows` | 475 / 3,963 | tabular answers |
| `annexures` | 176 | referenced files, with `file_present` |
| `query_runs` / `query_results` | | durable trace of every search (so a later 👎 is debuggable) |
| `feedback` | | thumbs, upsert-on-revote |
| `query_audit` / `file_access_audit` | | every query and every file open, **including denials** |
| `sync_queue` / `sync_log` | | the watcher's durable queue and its history |

### The halfvec trap — read this before touching dense search

> **pgvector caps an HNSW index at 2000 dimensions. Our embedding model emits 2048.**

The column stays full-fidelity `vector(2048)`; the index is built on the **halfvec cast**:

```sql
CREATE INDEX idx_sub_questions_embedding_hnsw ON sub_questions
  USING hnsw (((embedding)::halfvec(2048)) halfvec_cosine_ops);
```

Therefore **every query must order by that exact expression**:

```sql
ORDER BY sq.embedding::halfvec(2048) <=> %(qvec)s::halfvec(2048)   -- Index Scan  ✅
ORDER BY sq.embedding <=> %(qvec)s                                  -- full scan   ❌
```

**Both return correct results.** Only the first uses the index — which is exactly what makes
the wrong form dangerous: it degrades *silently*, and you will only notice as the corpus
grows. There is exactly **one** dense SQL template in the codebase so the ordering cannot
drift, and `tests/test_dense_uses_index.py` **EXPLAINs the real query** and fails if the plan
is not an index scan (with a negative control proving the test can actually discriminate).

---

## 6. Phase 4 — retrieval and reranking (the heart of the system)

The query pipeline is a **LangGraph** graph. LangGraph is the **conductor only** — it owns no
model call and no DB call. Every node is a plain `(state, deps) -> patch` function, unit-
testable without a graph. There is no LangChain retriever and no LangChain vectorstore.

```
        query
          │
   ┌──────▼───────┐
   │ 1 QUERY_     │  detect language · embed (QUERY mode) · extract entities
   │   PROCESS    │
   └──────┬───────┘
   ┌──────▼───────────────────────────────┐
   │ 2 HYBRID_RETRIEVE                    │
   │   ┌────────┐ ┌─────────┐ ┌────────┐  │
   │   │ dense  │ │ keyword │ │ entity │  │   top-30 each
   │   └────────┘ └─────────┘ └────────┘  │
   └──────┬───────────────────────────────┘
   ┌──────▼───────┐
   │ 3 FUSE (RRF) │  weighted reciprocal rank fusion, dedup by doc_key
   └──────┬───────┘
          │  weak result?  ──yes──> widen ONCE, back to node 2
   ┌──────▼───────┐
   │ 4 RERANK     │  cross-encoder over candidates -> top 5
   └──────┬───────┘
   ┌──────▼───────┐
   │ 5 ASSEMBLE   │  join answers, tables, files. Display payload only.
   └──────┬───────┘
   ┌──────▼───────┐
   │ 6 DRAFT      │  OPTIONAL, OFF BY DEFAULT
   └──────────────┘
```

### Node 1 — query processing

**Language detection is for PROCESSING ONLY.** Devanagari → `hi`, else `en`. It is used to
report back to the caller and to pick the draft language. It is **never** passed to a
retriever.

> ### ⚠️ LANGUAGE NEVER FILTERS THE CANDIDATE SET ⚠️
>
> `question_language` appears in **no WHERE clause anywhere in retrieval**. A Hindi query
> must be able to match an English answer — cross-lingual matching is a **core capability**,
> not an edge case, and the multilingual reranker handles it. This is verified: a Devanagari
> query returns results and the reranker has ranked a Hindi passage above its English
> equivalent when the Hindi one was more relevant.

**Query embedding mode is mandatory.** Passages were indexed with `input_type=passage`; the
query is embedded with `input_type=query`. **The model is asymmetric** — embedding the query
as a passage measurably degrades retrieval. Call `embed_queries()`, never `embed_passages()`.

### Node 2 — the three retrievers

| retriever | how it works | catches |
|---|---|---|
| **dense** | pgvector HNSW cosine over `sub_questions.embedding` | paraphrase, cross-lingual, "electricity dues" ≈ "outstanding payments" |
| **keyword** | Postgres FTS, `websearch_to_tsquery` over `question_tsv` | exact terms, acronyms, project names, numbers |
| **entity** | matches the query against a vocabulary of **1,239** known NHPC entities (projects, states, subsidiaries), built from the corpus itself | "Subansiri", "Arunachal Pradesh" |

Each returns **top-30** (`RETRIEVE_*_TOP_N`). All three exclude soft-deleted documents
(`AND d.active`). None of them filters by language.

They are complementary and that is the entire point of hybrid search: dense finds
*meaning* and misses rare literal tokens; keyword finds *tokens* and misses paraphrase.

### Node 3 — Reciprocal Rank Fusion

```
score(d) = Σ over retrievers r that surfaced d of   weight_r / (rrf_k + rank_r(d))
```

Defaults: `RRF_K=60`, weights **dense 1.0, keyword 0.7, entity 0.5**.

RRF fuses on **rank, not score** — which is the reason to use it. A cosine distance and an
FTS `ts_rank` are not comparable quantities; their ranks are. RRF needs no score
normalisation and no calibration between retrievers.

> **RRF SCORES ARE NOT ON A 0–1 SCALE.** This trips everyone up. With `k=60`, the best
> possible contribution from a single retriever at rank 1 is `weight/61`:
>
> | retriever | max contribution |
> |---|---|
> | dense | 1.0/61 = **0.01639** |
> | keyword | 0.7/61 = **0.01148** |
> | entity | 0.5/61 = **0.00820** |
>
> Theoretical max (rank 1 in all three) = **0.03607**. A lone dense hit at rank 1 = 0.01639.
> Any threshold you set must live in *that* range — not near 1.0.

**Dedup is by `doc_key`**, never `question_id` (§2). Within one document we keep the single
best-scoring sub-question, so **one document occupies one result slot** rather than a
document with eight parts flooding the page.

**Eligible vs fired** — a deliberate distinction:
- **eligible** = retrievers that *could* run. The entity retriever is **ineligible** when the
  query names no known entity.
- **fired** = eligible retrievers that actually returned ≥1 candidate.
- `agreement(d)` = how many *fired* retrievers found `d`, out of `fired`.

A query with no project name can only ever reach 2 retrievers. Scoring its agreement out of
a flat 3 would make every such query look weak. Both counts are tracked and both are shown.

### The widen branch — one retry, and it must actually widen

If the fused set looks weak, the graph **widens and retries once**:

- `top_score < WIDEN_TAU` (**0.0150**), or
- `score_gap < WIDEN_DELTA` (**0.0015**) — nothing clearly separated itself.

Note these thresholds sit inside the RRF range above, not on a 0–1 scale. They were set from
**measured** behaviour on this corpus (`scripts/measure_rrf.py`): strong queries score
0.027–0.085 with a #1–#2 gap ≥ 0.0017; a vague query's gap collapses to ~0.0004.

> **The gap discriminates, not the top score.** A confident result *separates itself* from
> the runner-up. That is the more reliable weakness signal, and it is why both are checked.

A widened pass must **materially broaden** the candidate set — re-running with identical
parameters would return identical results and make the retry pointless. So it:
1. multiplies every retriever's top-N by `WIDEN_TOP_N_FACTOR` (**3**);
2. relaxes entity from **filter** to **boost-only** — entity hits still lift a document's
   rank, but no longer restrict what dense/keyword may return;
3. **drops the metadata filters** (house / session / nhpc_only).

The widen reason is **logged**, so the branch is tunable rather than a black box.

### Node 4 — reranking

The three retrievers are **bi-encoders and lexical matchers**: they compare a query vector to
a passage vector computed *independently*, so no retriever ever looks at the query and the
passage *together*. That is what makes them fast enough to scan the corpus — and it is also
their ceiling.

The reranker is a **cross-encoder**: it feeds `(query, passage)` through the model **jointly**
and scores the pair. It is far too expensive to run over 1,914 sub-questions, but it is
excellent over the ~60 candidates RRF hands it. Retrieval is *recall*; reranking is
*precision*.

Model: **`nvidia/llama-nemotron-rerank-1b-v2`** — strongly multilingual, which is what
carries the Hindi-query→English-answer case.

It emits a **logit, not a probability**: real observed values run from **+31.9** (a very
strong match) to **−10.2** (a weak one). Do not read it as 0–1. It is a *heuristic for
triage*, and the UI says so explicitly.

The final cut is **`RETRIEVE_FINAL_TOP_K=5`**.

> **The reranker is an OPTIONAL layer, and a failure degrades rather than breaks.** If the
> model is unreachable, we log it, set `rerank_failed`, and **return the RRF ordering**.
> Losing precision beats losing the results. The UI then *tells the officer* the ranking is
> fusion-only — hiding that would make the results look more trustworthy than they are.

`rerank_movement` records how far the cross-encoder moved each document (`+10` = promoted ten
places). It is shown because a big movement is exactly the case worth eyeballing.

**One real gotcha:** two documents can receive an *identical* logit (we have seen −10.2422
twice). A rank-order "regression" after a refactor turned out to be a **genuine tie**, not a
bug. Check `query_results` before assuming you broke something.

### Node 5 — assemble

Joins each surviving sub-question to its answer group, tables, reply file and annexures.
**Display payload only — nothing is generated here.**

Annexures are **honest**: `file_present = false` renders as *"referenced but unavailable"*,
not as a dead button. The record says an annexure exists; we could not find the file; we say
exactly that.

### Node 6 — draft (OFF)

`GENERATION_ENABLED=false`. If switched on it drafts a reply *grounded in the retrieved
answers*, in the query's language. It is deliberately off: the product promise is retrieval,
and a fabricated sentence in a parliamentary reply is a serious failure.

### A real trace

Query: *"electricity dues owed by Jammu and Kashmir"* → 5 results in ~1.6 s.
Entity `Jammu` extracted; `dense+entity` fired of 3 eligible.

| # | rrf | rerank | moved | found by | agreement | answer_type |
|---|---|---|---|---|---|---|
| 1 | 0.071201 | **13.66** | 0 | dense+entity | 1.0 | substantive |
| 2 | 0.052595 | −4.55 | +1 | dense+entity | 1.0 | deferred_to_ministry |
| 3 | 0.062552 | −7.39 | −1 | dense | 0.5 | deferred_to_ministry |
| 4 | 0.013699 | −8.53 | **+10** | dense | 0.5 | substantive |
| 5 | 0.012658 | −10.24 | +13 | dense | 0.5 | nil |

Read row 4: RRF ranked it ~14th; the cross-encoder promoted it **ten places**. That is the
reranker earning its cost — and precisely why the pipeline has both stages.

---

## 7. The API and the officer UI

Three endpoints. `POST /query`, `GET /file`, `POST /feedback`.

**Identity** comes from `X-User-Id` / `X-User-Role`, set by the **authenticating reverse
proxy** in front of the service. The API binds to `127.0.0.1` only — exposed directly, those
headers would be self-asserted and the RBAC worthless. This trust boundary is explicit, not
assumed. (The UI currently sends a fixed `officer1`/`officer`; a login system replaces it.)

**RBAC** on `/query` and `/file`. **Every query and every file open is audited — including
denials.**

> ### `/file` takes an ID, never a path
> The request is `doc_key` + `file_kind` (+ `ref_label`). The path is looked up **server-side**
> and proven to be inside `organized/`. A client never names a file, so a client can never
> traverse the filesystem.
>
> This caused a real bug worth knowing: file buttons were `<a href>` links, and **a browser
> navigation cannot send custom headers**, so `/file` correctly rejected them with 403 ("no
> role supplied"). The fix was *not* to weaken RBAC or to put the role in the query string
> (trivially forged, and it leaks into logs and browser history) — it was to `fetch()` the
> file **with** the headers and hand the bytes to the browser as a blob URL.

**Feedback** is **capture only — it never mutates ranking.** A repeat vote **updates** the
previous one (upsert), because officers change their minds. Every run is written to
`query_runs`/`query_results`, so a 👎 three weeks later is still debuggable.

The UI is one self-contained `static/index.html`: sticky search bar, in-memory recent
searches, answer-type badges, the honest annexure state, and a per-result heuristics line —
under a standing disclaimer that **scores are heuristics for triage, not a guarantee of
correctness**.

---

## 8. Automatic incremental sync (`nhpc watch`)

Drop a question folder into the source tree and it becomes searchable on its own.

**Adding.** A new file is **not processed immediately.** The affected *question folder* is
queued with a quiet period (`WATCH_SETTLE_SECONDS`, default 10s), and every further event
**pushes the deadline out**. Only once the folder stops changing is it processed.

That settling is not fussiness. A question folder is copied in **file by file**. Parsing it
mid-copy would read a reply whose annexure has not landed yet and record *"referenced but
unavailable"* **as fact**. Copying a 40-file session folder produces **one** job, not forty.

The queue is a **Postgres table, not an in-memory list.** Restart the watcher mid-processing
and nothing is lost: pending events are still queued, and a job left claimed by a killed
worker is released on startup (`FOR UPDATE SKIP LOCKED`). Re-processing is idempotent, so
at-least-once delivery is harmless.

### Deleting — soft by default, never a silent hard delete

> A file disappearing from the source **deletes nothing.** The record is marked inactive: it
> **drops out of retrieval immediately**, but the row, its answers, its tables and its
> 2048-dim vectors all remain.

A disappearance is an **ambiguous signal** — a folder moved, a share reorganised, a mount
blipped, an officer tidied up: all four look *identical* from here. Acting irreversibly on an
ambiguous signal is how data is lost, and the embeddings alone cost real time and money to
rebuild. The discipline everywhere in this system is **flag, don't act**; delete is the one
place where getting that wrong cannot be undone.

If the folder **reappears**, the record **reactivates** — matched on the deterministic
`doc_key`, so **no re-parse and no re-embed** (measured: 17.6 s vs 57.6 s).

**Hard removal is a separate, deliberate command** and is never reached from a filesystem
event:

```bash
nhpc purge --dry-run              # what WOULD go
nhpc purge --older-than 30d       # asks for confirmation; cascades; no undo
```

Every add, soft-delete, reactivation and purge is written to `sync_log`, so *"why did this
record vanish from search?"* is always answerable.

---

## 9. Models and the provider seam

**Every model the original spec named is dead.** `llama-3.2-nv-embedqa-1b-v2` → 410.
`nv-rerankqa-mistral-4b-v3` → 404. `llama-3.2-nv-rerankqa-1b-v2` → 410. They were probed and
replaced:

| role | model | notes |
|---|---|---|
| embedding | `nvidia/llama-nemotron-embed-1b-v2` | **dim 2048**, L2-normalised → cosine. Asymmetric (query vs passage mode) |
| reranker | `nvidia/llama-nemotron-rerank-1b-v2` | cross-encoder, strongly multilingual, emits a logit |
| extraction LLM | Gemini 2.5 Flash (build) / **Qwen3 14B via Ollama (deploy)** | |

> ### The air-gap constraint
> These are **hosted** APIs. Every call sends document text off-network. That is acceptable
> for **build and test only — never for real NHPC data.**
>
> This is why `core/providers/` exists. Every backend sits behind an interface and is
> selected by **config alone**, so moving to a self-hosted model on-prem is an env change,
> not a code change. All keys and URLs come from the environment; **none is ever hardcoded**.
> `.env` is gitignored and the source corpus is never committed.

The UI likewise loads **no webfonts** — an outbound request from an air-gapped network would
hang, fall back silently, and leak a call from a network that must not make one.

---

## 10. Running it

```bash
pip install -e .                     # installs the `nhpc` command

cd deploy/postgres && docker compose up -d      # Postgres 16 + pgvector
cp .env.example .env                            # DSN + model API keys
nhpc migrate                                    # schema (001..004)

nhpc run                             # crawl -> parse -> index (the whole corpus)
nhpc serve                           # officer UI at http://127.0.0.1:8099
nhpc watch                           # keep the corpus in sync automatically

nhpc run --only 8773 --force         # reprocess one document
nhpc run --stages index              # just the DB load + embeddings
nhpc run --dry-run                   # validate, write nothing
nhpc query "electricity dues owed by J&K"
nhpc inspect --doc 8773
```

Port **8099**, not 8080: on Windows 8080 often falls inside a reserved TCP exclusion range
(Hyper-V/WinNAT) and fails to bind with **WinError 10013 even though nothing is listening**.
The error handler distinguishes that from 10048 (genuinely in use), because the two need
opposite responses.

Production: `deploy/systemd/` (`nhpc-api`, `nhpc-watch`, and an **optional, disabled**
`nhpc-purge.timer`). Hardened with `ProtectSystem=strict`.

---

## 11. Things that will bite you

1. **Drop the halfvec cast in dense SQL** → still correct, silently stops using the index.
   `test_dense_uses_index.py` exists to catch exactly this.
2. **Key on `question_id` instead of `doc_key`** → silently destroys the 9 colliding
   documents. This already happened once.
3. **Filter retrieval by language** → breaks cross-lingual search, which is a core
   capability. Language is for processing only.
4. **Embed the query in passage mode** → the model is asymmetric; retrieval measurably
   degrades.
5. **Read RRF scores as 0–1** → they max out around 0.036. Any threshold must be set in that
   range.
6. **Treat `needs_review` as a gate** → it is a *developer warning*. Everything loads;
   everything is retrievable.
7. **Hard-delete on a filesystem event** → a disappearance is ambiguous. Soft-delete, always.
8. **Trust `X-User-Role` from an exposed port** → it is only meaningful behind the
   authenticating proxy. Bind to loopback.
9. **Add an "answer must follow its question" invariant** → it is false. Diary 5341.
10. **Assume a rank change is a regression** → check `query_results` first; identical rerank
    logits produce genuine ties.

---

## 12. Open items

- **Rotate the two NVIDIA API keys** (embedding + reranking). They were pasted into a chat
  transcript during development. They live only in the gitignored `.env`, but they are
  exposed and should be rolled.
- **Delete the deprecation shims** at `phase2/`, `phase3/`, `phase4/` once the restructure is
  confirmed settled.
- **Generation and Langfuse are OFF by default** and await a decision. Both are one config
  flag away.
- **Authentication.** The UI sends a fixed `officer1`/`officer`. A real login must replace it
  and set the identity headers at the proxy.
