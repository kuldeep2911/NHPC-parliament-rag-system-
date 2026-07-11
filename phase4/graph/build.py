"""
LangGraph wiring. THE CONDUCTOR ONLY.

This module decides the ORDER things happen in and nothing else. Every model call
(embedding, rerank, generation) and every DB call happens inside a node, through the
existing provider interfaces. LangGraph/LangChain never wraps or owns a pgvector query
or a NIM call -- there is no LangChain retriever, no vectorstore adapter, no LLM wrapper.

    query_process -> hybrid_retrieve -> fuse
                                         |
                          weak?  -> widen_retrieve (ONCE) -> fuse
                                         |
                                      rerank -> assemble -> [generate] -> END

The WIDEN edge is capped at one retry and only fires when the fused set looks weak
(top RRF below tau, or the #1-#2 gap below delta -- both CONFIG, both measured against
the real RRF scale; see phase4/config.py). Every WIDEN decision is logged with the
scores and thresholds so the branch is tunable, not a black box.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from phase4.graph import nodes
from phase4.graph.state import QueryState
from phase4.retrieval.fuse import should_widen

log = logging.getLogger("nhpc.phase4.graph")


def _span_payload(node, state):
    """What each node contributes to its Langfuse span. Never the raw vectors."""
    stats = state.get("fuse_stats") or {}
    if node == "query_process":
        return {"language": state.get("language"), "entities": state.get("entities"),
                "_note": "language is PROCESSING ONLY — it never filters retrieval"}
    if node == "retrieve":
        r = state.get("retrieved") or {}
        return {"widened": bool(state.get("widened")),
                "counts": {k: len(v or []) for k, v in r.items()}}
    if node == "fuse":
        return {k: stats.get(k) for k in
                ("top_score", "score_gap", "n_candidates", "eligible", "fired")}
    if node == "rerank":
        rr = state.get("reranked") or []
        return {"kept": len(rr), "failed": bool(state.get("rerank_failed")),
                "top_logit": rr[0].get("rerank_logit") if rr else None}
    if node == "assemble":
        return {"results": [r["doc_key"] for r in (state.get("results") or [])]}
    return {}


def build_graph(deps):
    """
    Compile the query graph. `deps` carries the already-constructed providers and the DB
    connection:
        {"cfg":…, "conn":…, "embedder":…, "reranker":…|None, "llm":…|None,
         "entity_vocab":[…], "tracer":…|None}
    """
    cfg = deps["cfg"]

    g = StateGraph(QueryState)

    # Optional Langfuse mirror. A no-op unless LANGFUSE_ENABLED -- when off the SDK is
    # never imported and this costs one boolean check per node. The durable Postgres
    # trace (query_runs / query_results) is written regardless and remains the system of
    # record; Langfuse is the developer view on top.
    tracer = deps.get("tracer")

    def traced(name, fn):
        """Run a node, then mirror it as a Langfuse span. Tracing never breaks a query."""
        def _wrapped(state):
            patch = fn(state)
            if tracer is not None and tracer.enabled:
                merged = {**state, **(patch or {})}
                tracer.node(state, name, _span_payload(name, merged),
                            duration_ms=(merged.get("timings_ms") or {}).get(name))
            return patch
        return _wrapped

    g.add_node("query_process", traced("query_process", lambda s: nodes.query_process(s, deps)))
    g.add_node("retrieve", traced("retrieve", lambda s: nodes.hybrid_retrieve(s, deps)))
    g.add_node("fuse", traced("fuse", lambda s: nodes.fuse_results(s, deps)))
    g.add_node("rerank", traced("rerank", lambda s: nodes.rerank(s, deps)))
    g.add_node("assemble", traced("assemble", lambda s: nodes.assemble(s, deps)))

    # The widen node ONLY flips the flag; hybrid_retrieve reads it and broadens
    # (bigger top-N, entity relaxed to boost-only, metadata filters dropped).
    #
    # It RECOMPUTES the reason rather than reading one stashed by the conditional edge:
    # a LangGraph edge function's return value routes, and any mutation it makes to its
    # local state copy is DISCARDED. Stashing the reason there silently produced
    # "WIDEN fired: None". Recomputing here is cheap and cannot desync.
    def widen(state):
        stats = state.get("fuse_stats") or {}
        _do, reason = should_widen(stats, cfg, False)
        log.info("WIDEN fired: %s | top_score=%.5f gap=%.5f | tau=%.5f delta=%.5f "
                 "| top_n x%d, entity->boost-only, metadata filters dropped",
                 reason, stats.get("top_score", 0.0), stats.get("score_gap", 0.0),
                 cfg.widen_tau, cfg.widen_delta, cfg.widen_top_n_factor)
        return {"widened": True, "widen_reason": reason}

    g.add_node("widen", widen)

    g.set_entry_point("query_process")
    g.add_edge("query_process", "retrieve")
    g.add_edge("retrieve", "fuse")

    def after_fuse(state):
        """
        Weak result set -> widen ONCE. Otherwise straight to rerank.

        This function ROUTES ONLY. Anything it writes to `state` is discarded by
        LangGraph, so the reason is recomputed in the widen node (see above).
        """
        do, _reason = should_widen(state.get("fuse_stats") or {}, cfg,
                                   bool(state.get("widened")))
        return "widen" if do else "rerank"

    g.add_conditional_edges("fuse", after_fuse, {"widen": "widen", "rerank": "rerank"})
    g.add_edge("widen", "retrieve")        # re-retrieve, now broadened; fuse again
    g.add_edge("rerank", "assemble")

    # NODE 6 — generation is OPTIONAL and OFF by default. When off it is not even wired
    # in, so it cannot affect the live path.
    if cfg.generation_enabled:
        from phase4.generation.draft import generate_draft   # local: no import at all when off
        g.add_node("generate", lambda s: generate_draft(s, deps))
        g.add_edge("assemble", "generate")
        g.add_edge("generate", END)
    else:
        g.add_edge("assemble", END)

    return g.compile()
