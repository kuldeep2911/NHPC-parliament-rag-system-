"""
Reciprocal Rank Fusion over the three retrievers.

    score(d) = SUM over retrievers r that surfaced d of   weight_r / (rrf_k + rank_r(d))

rank is 1-based; a retriever that did not surface d contributes nothing. Weights and
rrf_k are CONFIG (RRF_W_DENSE / RRF_W_KEYWORD / RRF_W_ENTITY / RRF_K).

DEDUP BY doc_key (never question_id). A diary number is reused across sessions for a
DIFFERENT question -- 9 of the 517 documents share a number with another. doc_key
('<session>/<house>/<question_id>') is the identity end to end. Within one doc_key we
keep the single best-scoring sub-question, so one document occupies one result slot.

RRF SCORES ARE NOT ON A 0-1 SCALE. With rrf_k=60 the best possible contribution from one
retriever at rank 1 is weight/(60+1):
    dense   1.0/61 = 0.01639
    keyword 0.7/61 = 0.01148
    entity  0.5/61 = 0.00820
so the theoretical maximum (rank 1 in all three) is 0.03607, and a lone dense hit at
rank 1 is 0.01639. The WIDEN thresholds (tau/delta) must be set against THAT range --
see scripts/measure_rrf.py, which reports the observed distribution on this corpus.

RETRIEVER AGREEMENT (a HEURISTIC, not a correctness signal):
    eligible = retrievers that COULD run for this query
               (entity is INELIGIBLE when the query names no known entity)
    fired    = eligible retrievers that actually returned >= 1 candidate
    agreement(d) = how many of the FIRED retrievers surfaced d, out of `fired`
A query with no project name can reach at most 2 retrievers; scoring it out of 3 would
make every such query look like it had poor agreement. That is why eligible and fired
are tracked separately and both are reported.
"""

from __future__ import annotations


def fuse(result_lists, cfg, eligible, fired):
    """
    result_lists: {"dense": [...], "keyword": [...], "entity": [...]} -- each a list of
                  dicts from the retrievers, already rank-ordered (rank is 1-based).
    eligible:     set of retriever names that COULD run for this query
    fired:        set of retriever names that returned >= 1 candidate

    Returns (fused, stats):
      fused  = [{doc_key, sub_question_id, sub_question_local, question_text,
                 rrf_score, ranks: {retriever: rank}, agreement, retrievers: [...]}]
               sorted by rrf_score desc
      stats  = {top_score, score_gap, n_eligible, n_fired, eligible, fired}
    """
    weights = {
        "dense": cfg.rrf_weight_dense,
        "keyword": cfg.rrf_weight_keyword,
        "entity": cfg.rrf_weight_entity,
    }
    k = cfg.rrf_k

    # doc_key -> accumulator. DEDUP HAPPENS HERE, on doc_key.
    acc = {}
    for retriever, items in (result_lists or {}).items():
        w = weights.get(retriever, 0.0)
        for it in items or []:
            dk = it["doc_key"]
            contrib = w / (k + it["rank"])
            a = acc.get(dk)
            if a is None:
                a = acc[dk] = {
                    "doc_key": dk,
                    "sub_question_id": it["sub_question_id"],
                    "sub_question_local": it.get("sub_question_local"),
                    "question_text": it.get("question_text"),
                    "rrf_score": 0.0,
                    "ranks": {},
                    "_best_contrib": -1.0,
                }
            a["rrf_score"] += contrib
            # keep the rank from each retriever that surfaced this document
            prev = a["ranks"].get(retriever)
            if prev is None or it["rank"] < prev:
                a["ranks"][retriever] = it["rank"]
            # within a doc_key, the representative sub-question is the one with the
            # strongest single contribution (so one document = one result slot)
            if contrib > a["_best_contrib"]:
                a["_best_contrib"] = contrib
                a["sub_question_id"] = it["sub_question_id"]
                a["sub_question_local"] = it.get("sub_question_local")
                a["question_text"] = it.get("question_text")

    n_fired = max(1, len(fired))          # avoid /0; a query always fires >=1 retriever
    fused = []
    for a in acc.values():
        a.pop("_best_contrib", None)
        surfaced_by = [r for r in a["ranks"] if r in fired]
        a["retrievers"] = sorted(a["ranks"].keys())
        # HEURISTIC: fraction of the retrievers that actually ran which found this doc
        a["agreement"] = round(len(surfaced_by) / n_fired, 3)
        fused.append(a)

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)

    top = fused[0]["rrf_score"] if fused else 0.0
    second = fused[1]["rrf_score"] if len(fused) > 1 else 0.0
    stats = {
        "top_score": round(top, 6),
        "score_gap": round(top - second, 6),
        "n_eligible": len(eligible),
        "n_fired": len(fired),
        "eligible": sorted(eligible),
        "fired": sorted(fired),
        "n_candidates": len(fused),
    }
    return fused, stats


def should_widen(stats, cfg, already_widened):
    """
    Should the graph widen the search and retry ONCE?

    True when the fused set looks weak: the top RRF score is below tau, OR the #1-#2 gap
    is below delta (nothing clearly separated itself). Capped at one retry.

    Returns (bool, reason) -- the reason is LOGGED so the branch is tunable rather than a
    black box (see Change 3).
    """
    if not cfg.widen_enabled or already_widened:
        return False, None
    if not stats["n_candidates"]:
        return True, "no candidates at all"
    if stats["top_score"] < cfg.widen_tau:
        return True, (f"top_score {stats['top_score']:.5f} < tau {cfg.widen_tau:.5f}")
    if stats["score_gap"] < cfg.widen_delta:
        return True, (f"score_gap {stats['score_gap']:.5f} < delta {cfg.widen_delta:.5f}")
    return False, None
