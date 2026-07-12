"""
Trace / observability layer.

Every document processed gets a run_id. Every pipeline step (routing, extraction)
writes a structured JSON trace row keyed by run_id. Two sinks behind one interface:

  * PostgresSink  — writes to tables `runs` and `run_steps` in an existing Postgres
    (DSN from cfg.trace_dsn / env NHPC_TRACE_DSN). Tables are created idempotently
    and do NOT disturb any existing schema.
  * JsonlSink     — append-only JSONL under organized/_reports/trace/ when no DSN is
    configured or psycopg is unavailable, so the pipeline always runs.

⚠️ This trace stores DOCUMENT CONTENT (raw text in, raw model output). It therefore
inherits the SAME access controls + retention policy as the source data — it is not
a casual debug log. Treat _reports/trace/ and the Postgres tables accordingly.

run_id design: "<YYYYmmddTHHMMSSZ>_<short-uuid>" at the run level, and each document
gets a doc_run_id "<run_id>::<question_path>" so a later 👎 in the query phase can be
joined back to the exact document + step that produced a bad answer.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import uuid


def new_run_id() -> str:
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# sinks
# ---------------------------------------------------------------------------

class JsonlSink:
    """Append-only JSONL sink. One runs file + one run_steps file per run."""

    def __init__(self, trace_dir: str, run_id: str):
        os.makedirs(trace_dir, exist_ok=True)
        self.run_id = run_id
        self.runs_path = os.path.join(trace_dir, "runs.jsonl")
        self.steps_path = os.path.join(trace_dir, f"run_steps_{run_id}.jsonl")

    def start_run(self, meta: dict):
        self._append(self.runs_path, {"run_id": self.run_id, "event": "start",
                                      **meta})

    def end_run(self, meta: dict):
        self._append(self.runs_path, {"run_id": self.run_id, "event": "end",
                                      **meta})

    def write_step(self, row: dict):
        self._append(self.steps_path, row)

    def _append(self, path, obj):
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    @property
    def kind(self):
        return f"jsonl:{os.path.basename(self.steps_path)}"


class PostgresSink:
    """
    Postgres sink. Creates `runs` and `run_steps` idempotently. Uses psycopg (v3)
    if importable, else psycopg2. Never disturbs existing schema (CREATE IF NOT
    EXISTS only, no drops/alters of anything else).
    """

    def __init__(self, dsn: str, run_id: str):
        self.run_id = run_id
        self.dsn = dsn
        self._conn = None
        self._driver = None
        self._connect()
        self._ensure_schema()

    def _connect(self):
        try:
            import psycopg  # v3
            self._conn = psycopg.connect(self.dsn, autocommit=True)
            self._driver = "psycopg"
        except Exception:
            import psycopg2
            self._conn = psycopg2.connect(self.dsn)
            self._conn.autocommit = True
            self._driver = "psycopg2"

    def _ensure_schema(self):
        ddl = [
            """CREATE TABLE IF NOT EXISTS runs (
                   run_id        TEXT PRIMARY KEY,
                   started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                   ended_at      TIMESTAMPTZ,
                   backend       TEXT,
                   config        JSONB,
                   summary       JSONB
               )""",
            """CREATE TABLE IF NOT EXISTS run_steps (
                   id            BIGSERIAL PRIMARY KEY,
                   run_id        TEXT NOT NULL,
                   doc_run_id    TEXT,
                   question_path TEXT,
                   question_id   TEXT,
                   step          TEXT NOT NULL,          -- routing | extraction
                   ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
                   backend       TEXT,
                   model_name    TEXT,
                   duration_ms   INTEGER,
                   payload       JSONB                    -- raw in/out, prompt, parsed
               )""",
            "CREATE INDEX IF NOT EXISTS idx_run_steps_run_id ON run_steps(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_run_steps_doc ON run_steps(doc_run_id)",
        ]
        with self._conn.cursor() as cur:
            for stmt in ddl:
                cur.execute(stmt)

    def start_run(self, meta: dict):
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO runs (run_id, backend, config) VALUES (%s, %s, %s) "
                "ON CONFLICT (run_id) DO NOTHING",
                (self.run_id, meta.get("backend"),
                 json.dumps(meta.get("config", {}), ensure_ascii=False)))

    def end_run(self, meta: dict):
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE runs SET ended_at = now(), summary = %s WHERE run_id = %s",
                (json.dumps(meta.get("summary", {}), ensure_ascii=False), self.run_id))

    def write_step(self, row: dict):
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO run_steps (run_id, doc_run_id, question_path, "
                "question_id, step, backend, model_name, duration_ms, payload) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (row.get("run_id"), row.get("doc_run_id"), row.get("question_path"),
                 row.get("question_id"), row.get("step"), row.get("backend"),
                 row.get("model_name"), row.get("duration_ms"),
                 json.dumps(row.get("payload", {}), ensure_ascii=False)))

    @property
    def kind(self):
        return f"postgres:{self._driver}"


# ---------------------------------------------------------------------------
# tracer facade
# ---------------------------------------------------------------------------

class RunTracer:
    """Run-level tracer. Spawn a DocTracer per document via .for_doc()."""

    def __init__(self, cfg, run_id: str, root: str):
        self.cfg = cfg
        self.run_id = run_id
        self.enabled = cfg.trace_enabled
        self.sink = None
        self.status = "disabled"
        # OPTIONAL Langfuse mirror. Constructed always, but a hard no-op unless
        # cfg.langfuse_enabled -- when off it imports nothing and does nothing, so
        # the durable sink below is entirely unaffected. Independent of trace_enabled.
        self.langfuse = _make_langfuse(cfg)
        if not self.enabled:
            return
        # choose sink
        if cfg.trace_dsn:
            try:
                self.sink = PostgresSink(cfg.trace_dsn, run_id)
                self.status = self.sink.kind
                return
            except Exception as e:
                # degrade to JSONL rather than fail the run
                self._degraded = f"postgres unavailable ({type(e).__name__}); using jsonl"
        trace_dir = os.path.join(root, cfg.reports_subdir, "trace")
        self.sink = JsonlSink(trace_dir, run_id)
        self.status = self.sink.kind

    def start(self, backend_name: str):
        self.backend_name = backend_name
        if self.sink:
            self.sink.start_run({"backend": backend_name,
                                 "config": _safe_config(self.cfg)})
        if self.langfuse and self.langfuse.enabled:
            self.langfuse.start_run(self.run_id,
                                    {"backend": backend_name,
                                     "config": _safe_config(self.cfg)})

    def finish(self, summary: dict):
        if self.sink:
            try:
                self.sink.end_run({"backend": getattr(self, "backend_name", None),
                                   "summary": summary})
            except Exception:
                pass
        if self.langfuse and self.langfuse.enabled:
            self.langfuse.finish()

    def for_doc(self, question_path: str, question_id, backend_name: str):
        return DocTracer(self, question_path, question_id, backend_name)


class DocTracer:
    """Per-document tracer. All steps carry the run_id + doc_run_id join keys."""

    def __init__(self, run_tracer: RunTracer, question_path: str, question_id,
                 backend_name: str):
        self.rt = run_tracer
        self.run_id = run_tracer.run_id
        self.doc_run_id = f"{run_tracer.run_id}::{question_path}"
        self.question_path = question_path
        self.question_id = question_id
        self.backend_name = backend_name

    def step(self, step: str, payload: dict, model_name: str = None,
             duration_ms: int = None):
        rt = self.rt
        lf = getattr(rt, "langfuse", None) if rt else None
        # Nothing to do only if BOTH the durable sink and the Langfuse mirror are off.
        if not rt or (not rt.sink and not (lf and lf.enabled)):
            return
        row = {
            "run_id": self.run_id,
            "doc_run_id": self.doc_run_id,
            "question_path": self.question_path,
            "question_id": self.question_id,
            "step": step,
            "backend": self.backend_name,
            "model_name": model_name,
            "duration_ms": duration_ms,
            "payload": payload,
        }
        if rt.sink:
            try:
                rt.sink.write_step(row)
            except Exception:
                pass  # tracing must never crash the run
        if lf and lf.enabled:
            lf.log_step(row)  # already swallows its own exceptions


def _make_langfuse(cfg):
    """
    Build the optional Langfuse mirror. Returns a LangfuseTracer (which is itself a
    no-op when cfg.langfuse_enabled is false), or None if even importing the small
    wrapper fails. Importing langfuse_client does NOT import the langfuse SDK -- that
    only happens inside LangfuseTracer when enabled -- so this is safe when off.
    """
    try:
        from nhpc_qa.core.trace.langfuse_client import LangfuseTracer
        return LangfuseTracer(cfg)
    except Exception:
        return None


def _safe_config(cfg) -> dict:
    """Config snapshot with no secrets (env var NAMES only, never values)."""
    return {
        "backend": cfg.backend,
        "llm_model": cfg.llm_model,
        "vision_model": cfg.vision_model,
        "llm_base_url_set": bool(cfg.llm_base_url),
        "nvidia_base_url": cfg.nvidia_base_url,
        "nvidia_model": cfg.nvidia_model,
        "prefer_docling": cfg.prefer_docling,
        "enable_ocr": cfg.enable_ocr,
    }
