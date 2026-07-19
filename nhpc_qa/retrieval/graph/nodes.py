"""
LangGraph nodes. Each is a PLAIN FUNCTION (state, deps) -> state-patch, so every one is
unit-testable without a graph.

LangGraph is the CONDUCTOR ONLY. Nodes call the existing provider interfaces directly
(nhpc_qa.core.providers.embeddings.get_embedder, nhpc_qa.core.providers.rerank.get_reranker,
nhpc_qa.core.providers.models.get_llm) and psycopg directly. There is no LangChain retriever, no
LangChain vectorstore, nothing wrapping or owning a model/DB call.

⚠️ LANGUAGE NEVER FILTERS THE CANDIDATE SET (Change 6) ⚠️
`state["language"]` is computed in node 1 and used ONLY for:
    - reporting it back to the caller, and
    - the generation prompt (node 6, optional), so a Hindi question gets a Hindi draft.
It is NEVER passed to dense/keyword/entity search, and `question_language` appears in no
WHERE clause anywhere in phase4/retrieval/. Cross-lingual matching (Hindi query ->
English answer) is a core capability -- the reranker handles it, and it measurably does:
it ranked a Hindi passage above its English equivalent for a Hindi-relevant query.
Grep proof: `grep -rn "language" phase4/retrieval/` shows only comments, never SQL.
"""

from __future__ import annotations

import logging
import re
import time

from nhpc_qa.retrieval.search import dense, entity, fuse   # keyword/BM25 removed from fusion

log = logging.getLogger("nhpc.phase4.graph")

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def _timed(state, name, t0):
    state.setdefault("timings_ms", {})[name] = int((time.time() - t0) * 1000)


# ---------------------------------------------------------------------------
# NODE 1 — QUERY_PROCESS
# ---------------------------------------------------------------------------
def query_process(state, deps):
    """
    Detect language, embed the query in QUERY mode, extract entities.

    QUERY MODE IS MANDATORY: the sub-questions were indexed with input_type='passage'
    and this embedding model is ASYMMETRIC. Embedding the query as a passage measurably
    degrades retrieval, so we call embed_queries(), never embed_passages().
    """
    t0 = time.time()
    q = (state.get("query") or "").strip()

    # PROCESSING ONLY -- see the module docstring. Never a retrieval filter.
    language = "hi" if _DEVANAGARI.search(q) else "en"

    # Canonicalise the query's entity mentions against the SAME dictionary used at index
    # time. Two things use the result:
    #   1. entity_ids -> the entity retriever (a record linked to himachal_pradesh matches
    #      whether the query said "HP" or "Himachal Pradesh").
    #   2. a CANONICALISED query STRING -> dense embedding + reranking. "projects in HP" is
    #      rewritten to "projects in Himachal Pradesh" before it is embedded, so the dense
    #      neighbourhood and the reranker score are the SAME for both surface forms. Without
    #      this the entity signal agreed but dense/rerank still read the raw text ("HP"
    #      embeds only 0.63 like "Himachal Pradesh"), and the final SETS diverged. Measured.
    # It is deterministic (dictionary lookup, no LLM) and never filters by language.
    from nhpc_qa.entities import dictionary as _edict
    matched = _edict.match_entities(q, deps.get("alias_map") or {})
    entity_ids = [m["entity_id"] for m in matched]
    entities = [m["canonical"] for m in matched]
    q_canon = _edict.canonicalise_text(q, matched)

    query_vec = deps["embedder"].embed_queries([q_canon])[0]

    _timed(state, "query_process", t0)
    log.info("query_process: lang=%s entities=%s canon=%r", language, entities,
             q_canon if q_canon != q else None)
    return {"language": language, "query_vec": query_vec, "query_canon": q_canon,
            "entities": entities, "entity_ids": entity_ids,
            "widened": state.get("widened", False)}


# ---------------------------------------------------------------------------
# NODE 2 — HYBRID_RETRIEVE  (three retrievers)
# ---------------------------------------------------------------------------
def hybrid_retrieve(state, deps):
    """
    Run dense + keyword + entity.

    WIDENING (Change 4). Re-running this node with identical parameters would return
    identical results and make the retry pointless, so a widened pass MATERIALLY
    broadens the candidate set in three concrete ways:

        1. every retriever's top-N is multiplied by WIDEN_TOP_N_FACTOR (default 3);
        2. the entity retriever relaxes from FILTER to BOOST-ONLY -- entity results are
           still fused (so they still lift a document's rank) but they no longer
           restrict what dense/keyword may return, because...
        3. ...the metadata filters (house / session / is_nhpc_relevant) are DROPPED.

    (1) and (3) widen dense+keyword; (2) is what stops the entity list from acting as a
    de-facto gate on the fused set. Capped at ONE retry by the graph.

    NOTE: no language argument is passed to any retriever. Ever.
    """
    t0 = time.time()
    cfg = deps["cfg"]
    conn = deps["conn"]
    widened = bool(state.get("widened"))

    factor = cfg.widen_top_n_factor if widened else 1
    # On a widened pass the metadata filters are dropped (point 3 above).
    house = None if widened else state.get("house")
    session = None if widened else state.get("session")
    nhpc_only = False if widened else bool(state.get("nhpc_only"))

    # CANONICAL entity ids (not raw text). See query_process.
    entity_ids = state.get("entity_ids") or []

    # RETRIEVAL IS DENSE + ENTITY. BM25/keyword was REMOVED from fusion: the gating test
    # (scratchpad/step0_bm25off.py) proved it was inert -- turning it off produced
    # byte-identical top-5 results, because the cross-encoder reranker dominates the fused
    # pool. Adjacent-project precision (Teesta-VI vs Teesta-V) is carried by the entity
    # retriever, which the same test confirmed. The tsvector column/index remain in the
    # schema (migration 001) but are no longer read here -- disabled, not dropped, so this
    # is reversible.
    retrieved = {
        "dense": dense.search(conn, state["query_vec"], cfg.dense_top_n * factor,
                              house=house, session=session, nhpc_only=nhpc_only),
        # entity is BOOST-ONLY on a widened pass: it contributes RRF rank but its metadata
        # filters are gone, so it cannot narrow the candidate pool.
        "entity": entity.search(conn, entity_ids, cfg.entity_top_n * factor,
                                house=house, session=session, nhpc_only=nhpc_only),
    }

    _timed(state, "retrieve_widened" if widened else "retrieve", t0)
    log.info("hybrid_retrieve(widened=%s, factor=%d): dense=%d entity=%d",
             widened, factor, len(retrieved["dense"]), len(retrieved["entity"]))
    return {"retrieved": retrieved}


# ---------------------------------------------------------------------------
# NODE 3 — FUSE (RRF)
# ---------------------------------------------------------------------------
def fuse_results(state, deps):
    """
    Weighted RRF, deduped by doc_key.

    ELIGIBLE vs FIRED. Retrieval is now DENSE + ENTITY (BM25 removed). Dense is ALWAYS
    eligible; the entity retriever is eligible only when the query canonicalises to a known
    entity. So a query naming no entity reaches 1 retriever, one naming an entity reaches 2 --
    and agreement is scored out of what was ELIGIBLE, not a flat count, so a no-entity query
    is not made to look like it had poor agreement.
    """
    t0 = time.time()
    cfg = deps["cfg"]
    retrieved = state.get("retrieved") or {}

    eligible = {"dense"}
    if state.get("entity_ids"):
        eligible.add("entity")
    fired = {r for r in eligible if retrieved.get(r)}

    fused, stats = fuse.fuse(retrieved, cfg, eligible, fired)
    _timed(state, "fuse", t0)
    log.info("fuse: candidates=%d top=%.5f gap=%.5f eligible=%s fired=%s",
             stats["n_candidates"], stats["top_score"], stats["score_gap"],
             stats["eligible"], stats["fired"])
    return {"fused": fused, "fuse_stats": stats}


# ---------------------------------------------------------------------------
# NODE 4 — RERANK  (optional layer: a failure degrades, never breaks)
# ---------------------------------------------------------------------------
def rerank(state, deps):
    """
    Cross-encoder over the fused candidates -> top-K.

    The reranker is an OPTIONAL layer. If it fails (network, model gone), the officer
    still gets the RRF ordering -- we log, flag it, and carry on. Losing precision beats
    losing the results.
    """
    t0 = time.time()
    cfg = deps["cfg"]
    fused = state.get("fused") or []
    k = cfg.final_top_k

    if not fused:
        _timed(state, "rerank", t0)
        return {"reranked": [], "rerank_failed": False}

    if not cfg.rerank_enabled or deps.get("reranker") is None:
        out = [dict(c, rerank_logit=None, rerank_movement=0) for c in fused[:k]]
        _timed(state, "rerank", t0)
        return {"reranked": out, "rerank_failed": False}

    passages = [c["question_text"] or "" for c in fused]
    # Rerank against the CANONICALISED query (same rewrite dense used), so "HP" and
    # "Himachal Pradesh" get the same cross-encoder scores. Falls back to the raw query if
    # canonicalisation produced nothing (no entity mentioned).
    rerank_q = state.get("query_canon") or state["query"]
    try:
        ranking = deps["reranker"].rerank(rerank_q, passages)
    except Exception as e:                       # noqa: BLE001 -- optional layer
        log.warning("rerank failed (%s: %s) — falling back to RRF order", type(e).__name__, e)
        out = [dict(c, rerank_logit=None, rerank_movement=0) for c in fused[:k]]
        _timed(state, "rerank", t0)
        return {"reranked": out, "rerank_failed": True,
                "errors": (state.get("errors") or []) + [f"rerank: {type(e).__name__}"]}

    out = []
    for new_pos, (idx, logit) in enumerate(ranking[:k], start=1):
        c = dict(fused[idx])
        c["rerank_logit"] = round(float(logit), 4)
        # how far the cross-encoder moved it: + means promoted
        c["rerank_movement"] = (idx + 1) - new_pos
        out.append(c)

    _timed(state, "rerank", t0)
    log.info("rerank: %d -> %d (top logit %.3f)", len(fused), len(out),
             out[0]["rerank_logit"] if out else float("nan"))
    return {"reranked": out, "rerank_failed": False}


# ---------------------------------------------------------------------------
# NODE 4b — VERIFY  (sigmoid filter, then batched LLM similarity check)
# ---------------------------------------------------------------------------
def verify(state, deps):
    """
    Trim the reranked set to genuine matches: a cheap sigmoid recall-filter, then an LLM
    precision pass. Output is VARIABLE length (0..safety_max) -- the fixed-5 cap is gone.

    Calibration proved the sigmoid cannot separate matches from noise on its own (a real
    match scored 0.003; boilerplate scored 0.9999), so the threshold is deliberately low
    and the LLM does the discriminating. See nhpc_qa/retrieval/verify.py.

    RESILIENT: if the LLM verify pass fails, the sigmoid set is returned UNVERIFIED with
    verification_unavailable=True. The officer never loses results to an LLM outage.
    """
    from nhpc_qa.retrieval import verify as V

    t0 = time.time()
    cfg = deps["cfg"]
    reranked = state.get("reranked") or []

    kept, sig_dropped = V.sigmoid_filter(reranked, cfg.similarity_threshold,
                                         cfg.safety_max_results)
    log.info("sigmoid_filter: %d candidates, %d passed >= %.3f, %d dropped",
             len(reranked), len(kept), cfg.similarity_threshold, sig_dropped)

    verify_meta = {"enabled": bool(cfg.llm_verify_enabled), "unavailable": False,
                   "ms": 0, "checked": 0, "kept": len(kept)}

    if cfg.llm_verify_enabled and kept:
        llm = deps.get("llm")
        if llm is None:
            from nhpc_qa.core.providers import get_llm
            try:
                llm = get_llm(cfg)
                deps["llm"] = llm
            except Exception as e:      # noqa: BLE001 -- resilient: no LLM -> unverified
                log.warning("verify: no LLM (%s) — returning sigmoid set unverified",
                            type(e).__name__)
                llm = None
        if llm is not None:
            kept, meta = V.llm_verify(cfg, llm, state["query"], kept)
            verify_meta.update(meta)
        else:
            verify_meta.update({"unavailable": True, "reason": "no LLM configured"})
            for c in kept:
                c["verify_verdict"] = "unverified"
    else:
        for c in kept:
            c["verify_verdict"] = "disabled" if not cfg.llm_verify_enabled else "similar"

    _timed(state, "verify", t0)
    return {"reranked": kept,
            "sigmoid_dropped": sig_dropped,
            "verify_meta": verify_meta,
            "verification_unavailable": verify_meta["unavailable"]}


# ---------------------------------------------------------------------------
# NODE 5 — ASSEMBLE  (display payload; nothing is generated here)
# ---------------------------------------------------------------------------
_ASSEMBLE_SQL = """
SELECT sq.sub_question_id, sq.sub_question_local, sq.doc_key, sq.part_label,
       sq.question_text, sq.question_language,
       d.question_id, d.session, d.session_year, d.house, d.subject, d.starred,
       d.is_nhpc_relevant, d.answer_file_path, d.needs_review,
       d.reply_date,
       ag.answer_group_id, ag.answer_text, ag.answer_type, ag.answer_language,
       ag.answers_parts
FROM sub_questions sq
JOIN diaries d        ON d.doc_key = sq.doc_key
JOIN answer_groups ag ON ag.answer_group_id = sq.answer_group_id
WHERE sq.sub_question_id = ANY(%(ids)s)
  AND d.active   -- never assemble a soft-deleted document
"""

_ANNEX_SQL = """
SELECT doc_key, ref_label, file_present, file_path, match_confidence, referenced_in_parts
FROM annexures WHERE doc_key = ANY(%(keys)s)
ORDER BY doc_key, ref_label
"""


def assemble(state, deps):
    """
    Join each surviving sub-question (BY doc_key) to its answer group, the reply file and
    its annexures. Display payload only -- nothing is generated here.

    Annexures are honest: file_present=false becomes 'referenced but unavailable' rather
    than a dead button.
    """
    t0 = time.time()
    conn = deps["conn"]
    picks = state.get("reranked") or []
    if not picks:
        _timed(state, "assemble", t0)
        return {"results": []}

    ids = [p["sub_question_id"] for p in picks]
    keys = [p["doc_key"] for p in picks]

    with conn.cursor() as cur:
        cur.execute(_ASSEMBLE_SQL, {"ids": ids})
        cols = [c.name for c in cur.description]
        by_id = {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}
        cur.execute(_ANNEX_SQL, {"keys": keys})
        annex_cols = [c.name for c in cur.description]
        annex_by_doc = {}
        for r in cur.fetchall():
            a = dict(zip(annex_cols, r))
            annex_by_doc.setdefault(a["doc_key"], []).append(a)

    stats = state.get("fuse_stats") or {}
    results = []
    for rank, p in enumerate(picks, start=1):
        row = by_id.get(p["sub_question_id"])
        if not row:
            continue
        annexes = annex_by_doc.get(p["doc_key"], [])
        results.append({
            "rank": rank,
            # IDENTITY: doc_key, never question_id (diary numbers repeat across sessions)
            "doc_key": p["doc_key"],
            "sub_question_id": p["sub_question_id"],
            "sub_question_local": row["sub_question_local"],
            "part_label": row["part_label"],
            "question_text": row["question_text"],
            "question_language": row["question_language"],

            "diary_number": row["question_id"],
            "session": row["session"],
            "session_year": row["session_year"],
            "house": row["house"],
            "subject": row["subject"],
            "starred": row["starred"],
            "is_nhpc_relevant": row["is_nhpc_relevant"],
            "needs_review": row["needs_review"],       # developer warning, not a gate

            # RECENCY. reply_date is the date the question was to be answered in
            # Parliament -- what "the most recent reply" means to an officer. NULL is
            # shown as "date unknown" and sorts LAST; it is never silently treated as
            # old or as new.
            "reply_date": row["reply_date"].isoformat() if row["reply_date"] else None,

            "answer_text": row["answer_text"],
            "answer_type": row["answer_type"],
            "answer_language": row["answer_language"],
            "answer_covers_parts": row["answers_parts"],

            # file buttons — served by ID, never by path (see phase4/security/paths.py)
            "reply_file": {
                "available": bool(row["answer_file_path"]),
                "file_kind": "reply",
            },
            "annexures": [{
                "ref_label": a["ref_label"],
                "file_kind": "annexure",
                # honest: referenced but the file was never found
                "available": bool(a["file_present"]),
                "status": ("available" if a["file_present"]
                           else "referenced but unavailable"),
                "match_confidence": a["match_confidence"],
            } for a in annexes],

            # RELEVANCE — sigmoid(logit), a readable 0-1 the officer can actually use.
            # Labelled a heuristic, not a probability (see verify.py). None when the
            # reranker degraded to RRF order and there is no logit to transform.
            "relevance": p.get("relevance"),
            # what the LLM verify pass decided: similar | unverified | disabled
            "verify_verdict": p.get("verify_verdict"),
            "verify_reason": p.get("verify_reason"),

            # CONFIDENCE SIGNALS — HEURISTICS, NOT CORRECTNESS GUARANTEES
            "signals": {
                "_note": "heuristics for triage, not a correctness guarantee",
                "relevance": p.get("relevance"),        # sigmoid(logit), 0-1
                "rrf_score": round(p.get("rrf_score", 0.0), 6),
                "rerank_logit": p.get("rerank_logit"),
                "rerank_movement": p.get("rerank_movement"),
                "retrievers": p.get("retrievers", []),
                "retriever_ranks": p.get("ranks", {}),
                "agreement": p.get("agreement"),
                "retrievers_eligible": stats.get("n_eligible"),
                "retrievers_fired": stats.get("n_fired"),
            },
        })

    # -----------------------------------------------------------------------
    # DISPLAY ORDER — relevance SELECTS, date ORDERS.
    # -----------------------------------------------------------------------
    # This runs LAST, over the already-retrieved, already-reranked top-K. It does not touch
    # retrieval: the same K documents are shown either way, only their order changes.
    #
    # WHY NOT SORT THE WHOLE CANDIDATE SET BY DATE. Because "most recent" is not "most
    # relevant". Sorting the candidate pool by date would put the newest questions on the
    # page whether or not they have anything to do with what the officer asked -- a recent
    # irrelevant reply outranking last year's exact precedent. So the cross-encoder decides
    # WHICH replies are relevant, and the date decides in WHAT ORDER those relevant replies
    # are read.
    #
    # relevance_rank is preserved on every result, so the UI can always show what the
    # ranker actually thought.
    if getattr(cfg_sort := deps["cfg"], "result_sort", "date") == "date" and results:
        for r in results:
            r["relevance_rank"] = r["rank"]

        # NULLS LAST, always. An undated document must never float to the top of a
        # "most recent first" list -- it is unknown, not new. Ties (same date) keep the
        # reranker's order, which is why relevance_rank is the second key.
        results.sort(key=lambda r: (r["reply_date"] is None,               # dated first
                                    _neg_date(r["reply_date"]),            # newest first
                                    r["relevance_rank"]))                  # then relevance
        for i, r in enumerate(results, start=1):
            r["rank"] = i
        log.info("display order: by reply_date DESC (relevance chose the set) — %d dated, "
                 "%d undated", sum(1 for r in results if r["reply_date"]),
                 sum(1 for r in results if not r["reply_date"]))
    else:
        for r in results:
            r["relevance_rank"] = r["rank"]

    _timed(state, "assemble", t0)
    return {"results": results}


def _neg_date(iso: str | None):
    """Sort key for DESCENDING date. ISO strings sort lexicographically, so negating means
    inverting the string order -- easiest done by returning a tuple that reverses it."""
    if not iso:
        return ()                      # never reached for the sort (nulls are keyed first)
    # '2026-04-02' -> (-2026, -4, -2): newest first, with no string trickery.
    y, m, d = iso.split("-")
    return (-int(y), -int(m), -int(d))
