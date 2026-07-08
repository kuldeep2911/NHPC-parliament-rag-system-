# NHPC Parliament Data — Phase 2 (Parsing + Q&A Extraction)

Turns each Phase-1 question folder into a validated, structured `parsed.json`
of Question–Answer pairs, handling prose layouts (Case A/B/C) and the three
table types. **Read-only on all Phase-1 inputs; idempotent; resumable.**

This phase does NOT build embeddings, a database, or retrieval — only parsing.

## Install

```bash
pip install -r phase2/requirements.txt
# optional, for the 12 legacy .doc answers:  install libreoffice (soffice on PATH)
```

Docling (primary extractor) pulls torch + rapidocr; first run downloads its
layout/table models once.

## Run

```bash
python -m phase2.pipeline                       # parse everything into parsed.json
python -m phase2.pipeline --limit 20            # first 20 unparsed folders
python -m phase2.pipeline --only 2025-feb/rajya_sabha/s-2299   # one folder/subtree
python -m phase2.pipeline --force               # re-parse even if parsed.json exists
python -m phase2.pipeline --dry-run             # analyze + report, write nothing
python -m phase2.pipeline --no-docling          # force lightweight fallbacks
python -m phase2.pipeline --backend local       # on-prem model (default; deterministic if no server)
python -m phase2.pipeline --backend nvidia      # air-gapped NVIDIA (stub until wired on-site)
python -m phase2.pipeline --no-trace            # disable the trace layer
```

## Backends — independent parser + LLM

Parser and LLM are selected separately so Nemotron-parse and Ollama-LLM run together:

```bash
NHPC_PARSER_BACKEND=nemotron   # nemotron (NVIDIA NIMs) | docling (CPU fallback)
NHPC_LLM_BACKEND=ollama        # ollama (llama3.2:3b) | deterministic (no-network)
NHPC_LLM_MODEL=llama3.2:3b     # single config value for the Ollama model
NHPC_OLLAMA_BASE_URL=http://localhost:11434/v1
```

CLI: `--parser-backend nemotron --llm-backend ollama`. The legacy `--backend nvidia`
still works (derives `parser=nemotron, llm=ollama`). Config is validated on startup
(fail-fast if the NVIDIA key or Ollama URL for a selected backend is missing).

## Document schema (diary-as-container)

Each parsed.json models ONE diary as a container of sub-questions (not a flat pair):
`diary_numbers[]` (array — a file may cover several), `starred`, `subject`,
`reply_format` (interleaved | questions_then_answers | covering_letter | unknown),
`is_nhpc_relevant`, and `sub_questions[]` where each has `part_label` (a/b/c…),
`question_text`, `answer_text`, `answer_type` (substantive | deferred_to_ministry |
nil | not_applicable), `answer_blocks[]`, per-sub-part `tables[]`, `annexure_refs[]`.
Plus `diary_level_tables[]`, `annexures_referenced[]`, `annexure_content_present`.
The folder `question_id` and the document `diary_numbers` are kept SEPARATE — a
difference is expected, not an error. Legacy `pairs[]` is retained for compatibility.

## (legacy) Backends — one switch: `local` ⇄ `nvidia`

Every model call (LLM structuring, OCR, VLM) goes through one provider interface
(`providers.py`). Switching backends changes ONLY `NHPC_BACKEND` (or `--backend`);
nothing downstream changes.

| Backend | Status | Config (env) |
|---|---|---|
| `local` | **functional, only tested path** | `NHPC_LLM_BASE_URL` (OpenAI-compatible on-prem server), `NHPC_LLM_MODEL` (default `llama3.2:3b`), `NHPC_VISION_MODEL` (VLM/OCR). No base_url ⇒ deterministic no-network LLM so the pipeline always runs. Tables via Docling/TableFormer. |
| `nvidia` | **NeMo Retriever NIMs** (self-hosted, air-gapped) | Three separate NIM microservices, each `POST {url}/v1/infer`: `NVIDIA_OCR_URL` (nemoretriever-ocr), `NVIDIA_PAGE_ELEMENTS_URL` (nemoretriever-page-elements), `NVIDIA_TABLE_STRUCTURE_URL` (nemoretriever-table-structure). Optional `NVIDIA_TOKEN`. **Tables are extracted via the NIMs** (page-elements → table-structure → OCR), replacing Docling/TableFormer. LLM is NOT served by these NIMs — keep it on `local` Ollama or point `NVIDIA_MODEL` at a separate on-prem LLM NIM. Never a cloud URL. |

### NVIDIA NIM table pipeline (`backend=nvidia`)

For each PDF page: (1) render to PNG, (2) **page-elements NIM** locates `table` regions, (3) for each region the **table-structure NIM** returns row/column boxes, (4) the **OCR NIM** reads each lattice cell → a clean grid (`RawTable source="nim"`, flag `nim_table_extracted`). Body text still comes from the digital parse. If a NIM is unreachable the page's tables are flagged (`nim_*_failed`) and the run continues — never crashes. Validated end-to-end against the documented `/v1/infer` schemas via a mock NIM; real GPU NIMs are a drop-in (set the URLs).

No model name or key is hardcoded in logic; all come from config/env.

## Per-page routing (combined-image PDFs)

PDFs are routed **per page**, not per file (`routing.py`): each page is classified
`digital` → `parse_document`, `scanned` → `ocr_image`, or `image_based` →
`parse_visual` (VLM), by extractable-text length + image-coverage. Pages reassemble
in reading order; every decision + heuristic value is recorded in `parsed.json`
under `page_routing[]` and in the trace.

## Trace / observability

Every run gets a `run_id`; every step (routing, extraction) writes a structured
row keyed by `run_id`/`doc_run_id` (`trace/`). Extraction rows store the exact
prompt, the **raw model output**, the parsed result, and the **model name +
backend** — so local-vs-nvidia results are comparable and a later 👎 traces to the
failing step. Sink: **Postgres** when `NHPC_TRACE_DSN` is set (tables `runs`,
`run_steps`, created idempotently), else append-only **JSONL** under
`organized/_reports/trace/`. ⚠️ The trace holds document content — treat it with
the same access controls + retention as the source data.

- **Idempotent / resumable:** folders with a `parsed.json` are skipped unless
  `--force`. A crash on one document is caught, logged, and the run continues.
- **UTF-8 throughout;** Devanagari in filenames and content is preserved (never
  translated).

## Modules (independently testable)

| Module | Responsibility |
|---|---|
| `config.py` | All tunables + env; API keys read from env only, never stored. |
| `ir.py` | Intermediate representation (Blocks, RawTable, Document) + output objects (Pair, TableOut) + language detection. |
| `reader.py` | File router. Docling primary; pdfplumber / python-docx / openpyxl / text fallbacks; libreoffice for DOC/RTF. Scanned-vs-digital by text length, not extension. |
| `layout.py` | Classifies `mostly_prose` / `prose_with_tables` / `qa_table`, detects Case A/B/C, flags email-wrapper replies. |
| `tables.py` | Table geometry cleaning (drops padding cols, merges split cols + multi-row headers, stitches wrapped rows) and column-role tagging. |
| `extract.py` | Two extraction paths (prose / qa-table) behind the swappable provider; schema validation; emits extraction trace steps. |
| `providers.py` | The single model seam: `get_provider(cfg)` → `LocalProvider` / `NvidiaProvider`, each with `complete_json` / `ocr_image` / `parse_visual`. |
| `routing.py` | Per-page PDF classification (digital / scanned / image_based) + heuristics. |
| `trace/` | Run/step trace to Postgres or JSONL; `run_id` + `doc_run_id` join keys. |
| `llm.py` | Back-compat shim: `get_backend(cfg)` delegates to `providers.get_provider`. |
| `pipeline.py` | Orchestration, idempotency, per-doc error isolation, reports, trace wiring. |

## LLM extraction path

`extract_qa()` calls the provider through one interface. With a real local LLM
(`NHPC_LLM_BASE_URL` set) the model does the pairing and its exact prompt + raw
output are traced; with no server it uses a deterministic no-network extractor so
the pipeline is always runnable/testable. All outputs are strict-JSON,
schema-validated; invalid → one stricter retry → route to review. Never crashes.

```bash
export NHPC_BACKEND=local
export NHPC_LLM_BASE_URL=http://localhost:11434/v1   # Ollama/vLLM/LM Studio
export NHPC_LLM_MODEL=llama3.2:3b                     # single place to set the model
python -m phase2.pipeline --force
```

## Table handling (tables are first-class, never flattened)

- **Type-1** (table inside prose): attached to the answer, `answer_is_table=false`,
  `table_role=supporting`.
- **Type-2** (table IS the answer): `answer_is_table=true`, `table_role=answer_data`;
  every row emitted with an `nl_rendering` so each row is independently embeddable.
- **Type-3** (whole reply is a Q&A table): one pair per row read from the
  question/answer columns; underlying table also emitted (`table_role=qa_pairs`).
  Multi-column / sub-row answers (e.g. per-year breakups) are stitched into the
  parent question and flagged.

Column roles tagged: `qno, question, answer, project_name, location, status,
date_timeline, capacity, cost, percentage_complete, other`. Uncertain geometry →
low confidence + flag, never a confident-but-wrong table.

## Output — `parsed.json` per question folder

Written alongside `metadata.json` (Phase-1 files never modified). See the top-level
task spec for the full schema; key fields: `layout_structure`, `layout_case_detected`,
`qa_table`, `pairs[]` (with per-pair `tables[]`), `tables_index`, `extraction_flags`,
`needs_review`.

## Reports (`organized/_reports/`)

- `parse_report.json` — summary (counts by status / parser / language / layout /
  case; table & OCR & conversion counts) + per-document records + errors.
- `parse_report.csv` — one row per question folder.
- `review_queue.csv` — every folder needing human review, with the reason.

A document is routed to review if it: has no answer file, is unreadable, failed
schema validation, is low-confidence, required OCR or format conversion, came from
a Phase-1-flagged folder, had inferred (not header-derived) QA-table columns, had
risky table flags, is an email-wrapper, or mismatches metadata question numbers.

## Known data realities (surfaced, not hidden)

- Corpus is **almost entirely digital PDFs** (only ~3 scanned → OCR via rapidocr).
- **~7% contain Devanagari** (mixed en+hi); detected and preserved, never translated.
- **~42% of replies contain a table.**
- Some "answers" are **Zimbra/forwarded emails** whose real reply is an attachment
  Phase-1 didn't extract → flagged `email_wrapper_reply_may_be_attachment`.
- **12 legacy `.doc`** answers need libreoffice; without it they are flagged.
