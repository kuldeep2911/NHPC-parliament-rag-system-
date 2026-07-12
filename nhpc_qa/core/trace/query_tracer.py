"""
Langfuse monitoring for the query graph — config-gated, OFF by default.

Same pattern as Phase 2 (phase2/trace/langfuse_client.py), and it REUSES that client
rather than adding a second one:

  * disabled (default): the langfuse SDK is never imported, nothing connects, zero
    latency, and a tracing failure can never break a query.
  * enabled: one trace per query (trace id = run_id) with a nested span per graph node
    (query_process -> retrieve -> fuse -> rerank -> assemble -> [generate]), so a trace
    lines up 1:1 with the durable query_runs / query_results rows on the same run_id.
    Feedback attaches to the same trace id.

The durable Postgres trace stays the system of record; Langfuse is the developer view on
top of it. Every Langfuse call is wrapped and swallowed -- an observability failure must
never cost an officer their results.
"""

from __future__ import annotations

import logging

log = logging.getLogger("nhpc.phase4.tracing")


class QueryTracer:
    """No-op unless cfg.langfuse_enabled. Safe to call unconditionally."""

    def __init__(self, cfg):
        self.enabled = False
        self._lf = None
        if not getattr(cfg, "langfuse_enabled", False):
            return
        try:
            # importing this does NOT import the langfuse SDK; LangfuseTracer only
            # imports it inside its own enable path
            from nhpc_qa.core.trace.langfuse_client import LangfuseTracer
            self._lf = LangfuseTracer(cfg)
            self.enabled = bool(self._lf.enabled)
        except Exception as e:      # noqa: BLE001
            log.warning("Langfuse unavailable (%s: %s); continuing without it",
                        type(e).__name__, e)

    def start(self, state, cfg):
        if not self.enabled:
            return
        try:
            self._lf.start_run(state["run_id"], {
                "query": state.get("query"),
                "user_role": state.get("user_role"),
                "rerank_enabled": cfg.rerank_enabled,
                "generation_enabled": cfg.generation_enabled,
            })
        except Exception as e:      # noqa: BLE001
            log.warning("langfuse start failed: %s", e)

    def node(self, state, node_name, payload, model_name=None, duration_ms=None):
        """One span per graph node."""
        if not self.enabled:
            return
        try:
            self._lf.log_step({
                "run_id": state["run_id"],
                "doc_run_id": state["run_id"],
                "question_path": node_name,
                "question_id": None,
                "step": node_name,
                "backend": "phase4",
                "model_name": model_name,
                "duration_ms": duration_ms,
                "payload": payload,
            })
        except Exception as e:      # noqa: BLE001
            log.warning("langfuse span failed: %s", e)

    def finish(self):
        if not self.enabled:
            return
        try:
            self._lf.finish()
        except Exception as e:      # noqa: BLE001
            log.warning("langfuse flush failed: %s", e)
