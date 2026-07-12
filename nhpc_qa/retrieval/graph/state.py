"""The state carried through the LangGraph query pipeline."""

from __future__ import annotations

from typing import Any, TypedDict


class QueryState(TypedDict, total=False):
    # --- input ---------------------------------------------------------------
    run_id: str
    query: str
    user_id: str
    user_role: str
    # optional metadata narrowing, supplied by the caller (NEVER a language filter)
    house: str | None
    session: str | None
    nhpc_only: bool

    # --- node 1: query_process ----------------------------------------------
    # language drives PROCESSING ONLY. It never scopes the candidate set -- a Hindi
    # query must be able to match an English sub-question and vice versa.
    language: str                # 'hi' | 'en'  (informational + generation prompt)
    query_vec: list              # QUERY-mode embedding (passages were indexed as passage)
    entities: list               # entities recognised in the query ([] -> entity retriever
                                 # is INELIGIBLE, not "failed")

    # --- node 2/3: retrieve + fuse ------------------------------------------
    retrieved: dict              # {"dense":[...], "keyword":[...], "entity":[...]}
    fused: list                  # RRF-ordered, deduped by doc_key
    fuse_stats: dict             # top_score, score_gap, eligible/fired, n_candidates
    widened: bool                # has the WIDEN retry already been used? (capped at 1)
    widen_reason: str | None     # logged so the branch is tunable, not a black box

    # --- node 4: rerank ------------------------------------------------------
    reranked: list               # top-K after the cross-encoder
    rerank_failed: bool          # optional layer: a failure degrades, never breaks

    # --- node 5: assemble ----------------------------------------------------
    results: list                # the display payload the officer sees

    # --- node 6: generate (OPTIONAL, off by default) -------------------------
    draft: dict | None           # {"text":..., "citations":[...]} or None

    # --- observability -------------------------------------------------------
    timings_ms: dict[str, Any]
    errors: list
