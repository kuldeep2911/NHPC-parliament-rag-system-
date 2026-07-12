"""
Langfuse mirror — an OPTIONAL, config-gated UI layer over the durable trace sink.

The Postgres/JSONL sink in trace/__init__.py stays the system-of-record. This module
adds a browsable Langfuse view on top and is DORMANT unless cfg.langfuse_enabled:

  * disabled (default / local): the langfuse SDK is NEVER imported, nothing connects,
    every method is a no-op returning None, and there is no latency. A missing or
    uninstalled SDK, or an unreachable server, cannot affect a run because the import
    only happens inside enable().
  * enabled (on the on-prem server): initialise from cfg and mirror each trace step as
    a Langfuse span, grouped under one trace per run_id so Langfuse cross-references
    the Postgres run_steps one-to-one. EVERY Langfuse call is wrapped in try/except
    that logs once and continues — a tracing failure must never crash or block a
    document.

⚠️ On-prem this points at a SELF-HOSTED Langfuse (single-server deployment). Do NOT
use Langfuse cloud: traces carry document content, same as the Postgres sink.

Design note: the langfuse package is an OPTIONAL dependency. It is imported lazily and
guarded, so it is not required for the pipeline to run.
"""

from __future__ import annotations

import logging

log = logging.getLogger("nhpc.langfuse")


class LangfuseTracer:
    """
    One instance per run. `.enabled` is false unless config asked for it AND init
    succeeded, so callers can gate with a single boolean and the disabled path costs
    nothing.

    Lifecycle mirrors RunTracer:
        lf = LangfuseTracer(cfg); lf.start_run(run_id, meta)
        lf.log_step(step_row)          # once per DocTracer.step()
        lf.finish()                    # flush
    """

    def __init__(self, cfg):
        self.enabled = False
        self._client = None
        self._traces = {}          # run_id -> trace handle (lazily created)
        self.status = "disabled"
        # Read the flag WITHOUT importing anything. If off, we are done -- the SDK is
        # never touched.
        if not getattr(cfg, "langfuse_enabled", False):
            return
        self._enable(cfg)

    # -- init (only when enabled) --------------------------------------------
    def _enable(self, cfg):
        public_key, secret_key = cfg.langfuse_keys()
        host = getattr(cfg, "langfuse_host", "") or ""
        # Validate ONLY here (enabled path). Missing config -> log + stay disabled;
        # never crash the run.
        missing = [name for name, val in (
            ("$" + cfg.langfuse_public_key_env, public_key),
            ("$" + cfg.langfuse_secret_key_env, secret_key),
            ("LANGFUSE_HOST", host)) if not val]
        if missing:
            log.error("LANGFUSE_ENABLED=true but missing %s; Langfuse disabled "
                      "for this run (Postgres/JSONL trace is unaffected).",
                      ", ".join(missing))
            return
        try:
            from langfuse import Langfuse  # lazy: only imported when enabled
        except Exception as e:              # not installed / import error
            log.error("LANGFUSE_ENABLED=true but the langfuse SDK is unavailable "
                      "(%s: %s); Langfuse disabled for this run.",
                      type(e).__name__, e)
            return
        try:
            self._client = Langfuse(public_key=public_key, secret_key=secret_key,
                                    host=host)
            self.enabled = True
            self.status = f"langfuse:{host}"
            log.info("Langfuse enabled -> %s", host)
        except Exception as e:
            log.error("Langfuse init failed (%s: %s); disabled for this run.",
                      type(e).__name__, e)
            self._client = None
            self.enabled = False

    # -- run / step / finish (all no-ops when disabled) ----------------------
    def start_run(self, run_id: str, meta: dict):
        if not self.enabled:
            return
        try:
            # one trace per run_id, so run_id is the Langfuse trace id and the
            # Postgres `runs` row and the Langfuse trace share the same key.
            self._traces[run_id] = self._client.trace(
                id=run_id, name="nhpc-phase2-run",
                metadata=_scrub(meta))
        except Exception as e:
            self._warn_once("start_run", e)

    def log_step(self, row: dict):
        """Mirror one DocTracer.step() row as a Langfuse span/generation."""
        if not self.enabled:
            return
        try:
            run_id = row.get("run_id")
            trace = self._traces.get(run_id) or self._ensure_trace(run_id)
            if trace is None:
                return
            step = row.get("step") or "step"
            payload = row.get("payload") or {}
            model = row.get("model_name")
            duration_ms = row.get("duration_ms")
            common = dict(
                name=f"{step}:{row.get('question_path', '')}"[:200],
                metadata=_scrub({
                    "doc_run_id": row.get("doc_run_id"),
                    "question_path": row.get("question_path"),
                    "question_id": row.get("question_id"),
                    "backend": row.get("backend"),
                    "duration_ms": duration_ms,
                    **{k: v for k, v in payload.items()
                       if k not in ("system_prompt", "user_prompt",
                                    "raw_model_output")},
                }),
            )
            # An LLM call -> a Langfuse "generation" (captures prompt/output/model);
            # anything else -> a plain span.
            is_model_call = bool(model) and step in ("extraction", "llm_crosscheck")
            if is_model_call:
                trace.generation(
                    model=model,
                    input=_truncate(payload.get("user_prompt")
                                    or payload.get("system_prompt")),
                    output=_truncate(payload.get("raw_model_output")),
                    **common)
            else:
                trace.span(**common)
        except Exception as e:
            self._warn_once("log_step", e)

    def finish(self):
        if not self.enabled:
            return
        try:
            self._client.flush()
        except Exception as e:
            self._warn_once("finish", e)

    # -- helpers --------------------------------------------------------------
    def _ensure_trace(self, run_id):
        if not run_id:
            return None
        try:
            t = self._client.trace(id=run_id, name="nhpc-phase2-run")
            self._traces[run_id] = t
            return t
        except Exception as e:
            self._warn_once("ensure_trace", e)
            return None

    _warned = set()

    def _warn_once(self, where: str, exc: Exception):
        key = (where, type(exc).__name__)
        if key not in LangfuseTracer._warned:
            LangfuseTracer._warned.add(key)
            log.warning("Langfuse %s failed (%s: %s); continuing without it.",
                        where, type(exc).__name__, exc)


def _truncate(val, limit=12000):
    if val is None:
        return None
    s = val if isinstance(val, str) else _json(val)
    return s[:limit]


def _json(val):
    import json
    try:
        return json.dumps(val, ensure_ascii=False, default=str)
    except Exception:
        return str(val)


def _scrub(meta: dict) -> dict:
    """Drop obviously secret-looking keys; values are already env-var-name-only in
    the trace layer, but be defensive since this leaves the process."""
    if not isinstance(meta, dict):
        return {}
    out = {}
    for k, v in meta.items():
        if any(s in str(k).lower() for s in ("secret", "api_key", "password", "token")):
            continue
        out[k] = v
    return out
