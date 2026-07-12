"""
Query trace + officer feedback.

TRACE: every query writes one `query_runs` row and one `query_results` row per shown
result, recording WHICH retrievers surfaced it, at what rank, its RRF score, and how far
the reranker moved it. That is what makes a later 👎 debuggable to root cause rather than
just "the answer was bad".

FEEDBACK: captured ONLY. It never feeds back into ranking. Live self-mutation would make
retrieval unstable and unauditable, which is unacceptable for government data. The table
is shaped so it can later be exported as a labelled evaluation set.

CHANGE 1 — a vote is UPDATABLE. An officer must be able to change their mind (👎 -> 👍),
so record_feedback UPSERTs on the uniqueness constraint and updates verdict/reason/
timestamp. It never raises on a repeat vote.
"""

from __future__ import annotations

import json
import logging

log = logging.getLogger("nhpc.phase4.feedback")


# ---------------------------------------------------------------------------
# trace
# ---------------------------------------------------------------------------

def save_run(conn, cfg, state):
    """
    Persist one query run + its results. Idempotent on run_id (a replay updates).

    Best-effort: if the trace write fails the officer still gets their results, but we
    log loudly -- an untraceable query cannot be debugged from a 👎 later.
    """
    stats = state.get("fuse_stats") or {}
    results = state.get("results") or []
    try:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO query_runs (
                        run_id, query_text, query_language, entities,
                        user_id, user_role,
                        retrievers_eligible, retrievers_fired, rrf_weights, rrf_k,
                        widened, widen_reason, rerank_enabled, rerank_failed,
                        generation_enabled, top_score, score_gap, n_candidates,
                        timings_ms, errors)
                    VALUES (%s,%s,%s,%s, %s,%s, %s,%s,%s,%s, %s,%s,%s,%s,
                            %s,%s,%s,%s, %s,%s)
                    ON CONFLICT (run_id) DO UPDATE SET
                        query_text = EXCLUDED.query_text,
                        top_score = EXCLUDED.top_score,
                        score_gap = EXCLUDED.score_gap,
                        timings_ms = EXCLUDED.timings_ms
                """, (
                    state["run_id"], state.get("query"), state.get("language"),
                    list(state.get("entities") or []),
                    state.get("user_id"), state.get("user_role"),
                    stats.get("eligible") or [], stats.get("fired") or [],
                    json.dumps({"dense": cfg.rrf_weight_dense,
                                "keyword": cfg.rrf_weight_keyword,
                                "entity": cfg.rrf_weight_entity}),
                    cfg.rrf_k,
                    bool(state.get("widened")), state.get("widen_reason"),
                    bool(cfg.rerank_enabled), bool(state.get("rerank_failed")),
                    bool(cfg.generation_enabled),
                    stats.get("top_score"), stats.get("score_gap"),
                    stats.get("n_candidates"),
                    json.dumps(state.get("timings_ms") or {}),
                    list(state.get("errors") or []),
                ))

                for r in results:
                    s = r["signals"]
                    ranks = s.get("retriever_ranks") or {}
                    cur.execute("""
                        INSERT INTO query_results (
                            run_id, doc_key, sub_question_id, rank,
                            dense_rank, keyword_rank, entity_rank,
                            retrievers, agreement, rrf_score, rerank_logit,
                            rerank_movement)
                        VALUES (%s,%s,%s,%s, %s,%s,%s, %s,%s,%s,%s,%s)
                        ON CONFLICT (run_id, doc_key) DO UPDATE SET
                            rank = EXCLUDED.rank,
                            rrf_score = EXCLUDED.rrf_score,
                            rerank_logit = EXCLUDED.rerank_logit,
                            rerank_movement = EXCLUDED.rerank_movement
                    """, (
                        state["run_id"], r["doc_key"], r["sub_question_id"], r["rank"],
                        ranks.get("dense"), ranks.get("keyword"), ranks.get("entity"),
                        list(s.get("retrievers") or []), s.get("agreement"),
                        s.get("rrf_score"), s.get("rerank_logit"),
                        s.get("rerank_movement"),
                    ))
    except Exception as e:                      # noqa: BLE001
        log.error("TRACE WRITE FAILED for run %s: %s: %s",
                  state.get("run_id"), type(e).__name__, e)


# ---------------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------------

def record_feedback(conn, run_id, user_id, verdict, doc_key=None, reason=None,
                    user_role=None):
    """
    Capture one officer verdict. UPSERT -- a revised vote OVERWRITES the previous one
    (Change 1): an officer who first hits 👎 and then decides the result was useful must
    be able to change it, and the row must reflect the latest opinion.

    doc_key=None means feedback on the QUERY as a whole ("none of these helped").

    Raises ValueError on a bad verdict or an unknown run_id -- those are caller bugs, not
    something to swallow. Feedback NEVER changes any ranking.
    """
    if verdict not in ("up", "down"):
        raise ValueError(f"verdict must be 'up' or 'down', got {verdict!r}")

    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM query_runs WHERE run_id = %s", (run_id,))
        if not cur.fetchone():
            raise ValueError(f"unknown run_id {run_id!r} — cannot attach feedback")

        if doc_key is None:
            # whole-query feedback: the partial index uq_feedback_query enforces one
            # per (run_id, user_id)
            cur.execute("""
                INSERT INTO feedback (run_id, doc_key, verdict, reason, user_id, user_role)
                VALUES (%s, NULL, %s, %s, %s, %s)
                ON CONFLICT (run_id, user_id) WHERE doc_key IS NULL
                DO UPDATE SET verdict = EXCLUDED.verdict,
                              reason  = EXCLUDED.reason,
                              updated_at = now()
                RETURNING id
            """, (run_id, verdict, reason, user_id, user_role))
        else:
            cur.execute("SELECT 1 FROM query_results WHERE run_id=%s AND doc_key=%s",
                        (run_id, doc_key))
            if not cur.fetchone():
                raise ValueError(
                    f"{doc_key!r} was not shown for run {run_id!r} — refusing to attach "
                    f"feedback to a result the officer never saw")
            cur.execute("""
                INSERT INTO feedback (run_id, doc_key, verdict, reason, user_id, user_role)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, doc_key, user_id) WHERE doc_key IS NOT NULL
                DO UPDATE SET verdict = EXCLUDED.verdict,
                              reason  = EXCLUDED.reason,
                              updated_at = now()
                RETURNING id
            """, (run_id, doc_key, verdict, reason, user_id, user_role))
        fid = cur.fetchone()[0]
    return fid


# ---------------------------------------------------------------------------
# eval-ready export (no evaluation layer is built -- this just captures cleanly)
# ---------------------------------------------------------------------------

_EXPORT_SQL = """
SELECT f.run_id,
       r.query_text,
       r.query_language,
       f.doc_key,
       f.verdict,
       f.reason,
       f.user_id,
       f.user_role,
       f.updated_at,
       -- the retrieval decision this verdict is about (debuggable to root cause)
       qr.rank,
       qr.retrievers,
       qr.dense_rank, qr.keyword_rank, qr.entity_rank,
       qr.rrf_score, qr.rerank_logit, qr.rerank_movement,
       r.widened, r.top_score, r.score_gap,
       sq.question_text
FROM feedback f
JOIN query_runs r      ON r.run_id = f.run_id
LEFT JOIN query_results qr ON qr.run_id = f.run_id AND qr.doc_key = f.doc_key
LEFT JOIN sub_questions sq ON sq.sub_question_id = qr.sub_question_id
ORDER BY f.updated_at DESC
"""


def export_feedback(conn):
    """
    Every verdict joined to the exact retrieval decision that produced it.

    This is the future ground-truth source: a 👍 row says "for THIS query, THIS document
    was useful, and here is how retrieval found it". Captured cleanly now so that a
    labelled test set exists whenever an evaluation layer is built.
    """
    with conn.cursor() as cur:
        cur.execute(_EXPORT_SQL)
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
