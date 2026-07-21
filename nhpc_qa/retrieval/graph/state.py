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
    query_canon: str             # query with entity mentions canonicalised (HP->Himachal Pradesh)
    entity_ids: list             # canonical entity ids matched in the query
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

    # --- node 4b: verify (sigmoid filter + LLM similarity) -------------------
    # These MUST be declared here: LangGraph merges only keys present in this TypedDict,
    # so a returned key that is not listed is silently dropped -- which is exactly how
    # verify_meta went missing from the API response.
    sigmoid_dropped: int         # candidates cut by the sigmoid recall-filter
    verify_meta: dict            # {enabled, unavailable, checked, kept, ms, reason}
    verification_unavailable: bool  # the LLM verify pass could not run -> results unverified
    generic_query_suspected: bool   # plateau guard emptied the set: query is a stock
    #                                 fragment ("the reasons therefor") — UI says "too
    #                                 generic", not "no results". (Was silently dropped
    #                                 before this declaration — same lesson as verify_meta.)

    # --- node 5: assemble ----------------------------------------------------
    results: list                # the display payload the officer sees

    # --- node 6: generate (OPTIONAL, off by default) -------------------------
    draft: dict | None           # {"text":..., "citations":[...]} or None

    # --- observability -------------------------------------------------------
    timings_ms: dict[str, Any]
    errors: list
