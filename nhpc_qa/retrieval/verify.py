"""
Sigmoid relevance filter + batched LLM similarity verification.

Pipeline position: rerank -> [sigmoid_filter -> llm_verify] -> date-order -> display. These
are the two middle stages; they don't touch retrieval, RRF, doc_key, or the SQL.

Two stages because the sigmoid is the weak one. Calibration showed the sigmoid of a genuine
match and of boilerplate noise overlap completely (a real match scored 0.003; "details
thereof" scored 0.9999 against 46 docs). So the sigmoid filter is only a cheap recall
pre-filter that bounds how many candidates the LLM reads; the LLM verify pass does the real
"is this the same question" judgement, which a scalar cannot.

    sigmoid filter :  keep sigmoid >= SIMILARITY_THRESHOLD  (low on purpose)
    llm verify     :  one batched call -> drop the ones it says are not the same question

Resilient: the LLM is in the live path, so on failure we return the sigmoid set unverified
with verification_unavailable=True — never an error, never an empty result from the model
being unreachable.
"""

from __future__ import annotations

import json
import math
import re
import time

from nhpc_qa.core.logging import get_logger

log = get_logger("nhpc.verify")


def sigmoid(x: float) -> float:
    """Numerically stable logistic. Overflow-safe for the large-magnitude logits the
    reranker produces (seen: +32, -18)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def sigmoid_filter(reranked: list, threshold: float, safety_max: int):
    """
    Attach `relevance` (sigmoid of the logit) to every candidate and keep those >= threshold.
    Returns (kept, dropped_count).

    A candidate with a null logit (reranker degraded to RRF order) is kept with
    relevance=None — we can't score it, and dropping it would turn a reranker outage into
    silent data loss. `safety_max` bounds the survivors against a degenerate query; applied to
    the relevance-sorted list, it only ever trims the weakest.
    """
    kept = []
    dropped = 0
    for c in reranked:
        lg = c.get("rerank_logit")
        c["relevance"] = round(sigmoid(lg), 6) if lg is not None else None
        if lg is None or c["relevance"] >= threshold:
            kept.append(c)
        else:
            dropped += 1
    if len(kept) > safety_max:
        dropped += len(kept) - safety_max
        kept = kept[:safety_max]
    return kept, dropped


def plateau_guard(kept: list, cfg):
    """
    Deterministic generic-query detection from the shape of the reranked scores.

    A real question peaks: a few strong matches, then a drop. A generic fragment ("the
    reasons therefor") plateaus — the stock phrase appears in hundreds of sub-questions, so
    dozens of candidates tie near sigmoid 1.0 with nothing to prefer. The guard requires both
    a wide plateau (>= plateau_min_n above 0.9) and a saturated top (the plateau_min_n-th best
    sigmoid still >= plateau_min_sigmoid); a real topic fails the second test as its tail
    decays.

    Returns (is_plateau, info); this only measures, the caller decides.
    """
    if not getattr(cfg, "plateau_guard_enabled", False):
        return False, {}
    n_min = max(4, int(getattr(cfg, "plateau_min_n", 12)))
    sig_min = float(getattr(cfg, "plateau_min_sigmoid", 0.97))
    sigs = sorted((c.get("relevance") for c in kept
                   if c.get("relevance") is not None), reverse=True)
    wide = sum(1 for s in sigs if s > 0.9)
    if wide < n_min or len(sigs) < n_min:
        return False, {"wide": wide}
    nth = sigs[n_min - 1]
    if nth >= sig_min:
        return True, {"wide": wide, "nth_sigmoid": round(nth, 4), "n_min": n_min}
    return False, {"wide": wide, "nth_sigmoid": round(nth, 4)}


# ---------------------------------------------------------------------------
# LLM verification
# ---------------------------------------------------------------------------
_SYSTEM = """You check whether past parliamentary questions are the SAME question as a new one.

You are given a NEW question and a numbered list of PAST questions. For each past question,
decide whether it asks about the SAME underlying matter as the new question -- such that the
past answer is a relevant precedent for the new one.

Judge the UNDERLYING ASK, not the wording:
- Different words for the same matter -> similar. ("dues owed by DISCOMs" vs "outstanding
  payments from power distribution companies" are the same question.)
- Same broad topic but a DIFFERENT specific ask -> not_similar. (Both about NHPC dams, but
  one asks about seismic safety and the other about power generation -> not_similar.)
- Generic boilerplate that matches everything ("details thereof", "steps taken") is NOT a
  real match to a specific question -> not_similar.
- If the NEW question is ITSELF such a generic fragment — a stock sub-part with no specific
  subject of its own ("the reasons therefor", "if so the details thereof", "steps taken by
  the government") — then NO past question is a genuine match, even one with identical
  wording: matching boilerplate to boilerplate retrieves nothing an officer can use. Mark
  ALL not_similar.
- The NEW question may be a SHORT TOPIC QUERY rather than a full parliamentary question
  ("Kishanganga project status", "Salal power generation"). Treat it as the underlying ask:
  a past question that directly asks about that topic IS similar. Do not reject a real
  topical match just because the new query is terse.

Return STRICT JSON only, an array with one entry per past question, in order:
[{"id": <the number>, "verdict": "similar" | "not_similar", "reason": "<one short clause>"}]

Every id from the list must appear exactly once. No prose outside the JSON."""


def _build_prompt(query: str, candidates: list) -> str:
    lines = [f"NEW QUESTION:\n{query}\n", "PAST QUESTIONS:"]
    for i, c in enumerate(candidates):
        qt = " ".join((c.get("question_text") or "").split())
        lines.append(f"{i}. {qt}")
    lines.append("\nReturn the JSON array, one entry per past question, ids 0.."
                 f"{len(candidates) - 1}.")
    return "\n".join(lines)


_JSON_RE = re.compile(r"\[.*\]", re.S)


def _parse_verdicts(raw: str, n: int):
    """The model's JSON array -> {id: (verdict, reason)}, or None if unusable."""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S)
    obj = None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = _JSON_RE.search(s)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    if not isinstance(obj, list):
        return None
    out = {}
    for item in obj:
        if not isinstance(item, dict) or "id" not in item:
            continue
        try:
            i = int(item["id"])
        except (TypeError, ValueError):
            continue
        verdict = str(item.get("verdict", "")).lower()
        if verdict not in ("similar", "not_similar"):
            continue
        out[i] = (verdict, str(item.get("reason", ""))[:200])
    return out or None


# Max candidates per call. A larger batch asks for a huge JSON array that overflows the token
# budget and truncates mid-array -> unparseable -> the whole batch falls back to "keep
# everything" (which once let "details thereof" through with 46 results). So we chunk.
_BATCH = 12


def _qhash(query: str) -> str:
    import hashlib
    return hashlib.sha256((query or "").strip().lower().encode()).hexdigest()


def _cache_lookup(conn, qhash, candidates):
    """Return {sub_question_id: (verdict, reason)} already cached for this canonical query."""
    if conn is None:
        return {}
    ids = [c.get("sub_question_id") for c in candidates if c.get("sub_question_id")]
    if not ids:
        return {}
    try:
        with conn.cursor() as cur:
            cur.execute("""SELECT sub_question_id, verdict, reason FROM verify_cache
                           WHERE query_hash=%s AND sub_question_id = ANY(%s)""", (qhash, ids))
            return {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    except Exception:      # noqa: BLE001 -- cache miss on error, never fatal
        return {}


def _cache_store(conn, qhash, candidate, verdict, reason):
    if conn is None or not candidate.get("sub_question_id"):
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO verify_cache
                             (query_hash, doc_key, sub_question_id, verdict, reason)
                           VALUES (%s,%s,%s,%s,%s)
                           ON CONFLICT (query_hash, sub_question_id) DO UPDATE
                             SET verdict=EXCLUDED.verdict, reason=EXCLUDED.reason""",
                        (qhash, candidate.get("doc_key"), candidate["sub_question_id"],
                         verdict, (reason or "")[:300]))
        conn.commit()
    except Exception:      # noqa: BLE001
        pass


def llm_verify(cfg, llm, query: str, candidates: list, conn=None):
    """
    Batched verification with a deterministic per-canonical-query cache.

      verified : candidates the LLM judged similar, annotated with verify_verdict/reason,
                 in relevance order.
      meta     : {"unavailable", "ms", "checked", "kept", "reason"}.

    The cache is what makes synonym-equivalent queries return the same final set: the LLM is
    non-deterministic, so caching by (query_hash, sub_question_id) means the same canonical
    query always reuses the same verdict and skips the call. Only cache misses hit the LLM.

    Chunked so a large set doesn't force one enormous JSON array. A chunk that fails to parse
    after a retry makes the whole verification unavailable (fail soft, flagged). Never raises.
    """
    t0 = time.time()
    if not candidates:
        return [], {"unavailable": False, "ms": 0, "checked": 0, "kept": 0}

    qhash = _qhash(query)
    cached = _cache_lookup(conn, qhash, candidates)
    to_call = [c for c in candidates if c.get("sub_question_id") not in cached]

    verified = []
    # apply cached verdicts first (deterministic, no API)
    for c in candidates:
        if c.get("sub_question_id") in cached:
            verdict, reason = cached[c["sub_question_id"]]
            c["verify_verdict"] = verdict
            c["verify_reason"] = reason
            if verdict == "similar":
                verified.append(c)

    for start in range(0, len(to_call), _BATCH):
        chunk = to_call[start:start + _BATCH]
        try:
            verdicts = _verify_chunk(cfg, llm, query, chunk)
        except Exception as e:      # noqa: BLE001 -- LLM down/timeout mid-batch
            log.warning("llm_verify: call failed (%s) — returning unverified",
                        type(e).__name__)
            return _passthrough(candidates, t0, f"{type(e).__name__}")
        if verdicts is None:
            # A chunk we could not verify. Do NOT silently keep or drop it -- return the
            # WHOLE set unverified and flagged, so the UI tells the officer verification did
            # not complete. Half-verified is a lie either way.
            log.warning("llm_verify: a chunk failed to parse — returning unverified")
            return _passthrough(candidates, t0, "unparseable model output")
        for i, c in enumerate(chunk):
            verdict, reason = verdicts.get(i, ("similar", "no verdict returned — kept"))
            c["verify_verdict"] = verdict
            c["verify_reason"] = reason
            _cache_store(conn, qhash, c, verdict, reason)   # deterministic for next time
            if verdict == "similar":
                verified.append(c)
            else:
                log.info("llm_verify: dropped %s — %s", c.get("doc_key"), reason[:80])

    # verified was filled cache-first then LLM-second; restore the original relevance order
    # so the result order does not depend on what happened to be cached.
    order = {id(c): n for n, c in enumerate(candidates)}
    verified.sort(key=lambda c: order.get(id(c), 1e9))

    ms = int((time.time() - t0) * 1000)
    log.info("llm_verify: checked %d (%d cached, %d called), kept %d, %d ms",
             len(candidates), len(cached), len(to_call), len(verified), ms)
    return verified, {"unavailable": False, "ms": ms, "cached": len(cached),
                      "checked": len(candidates), "kept": len(verified)}


def _verify_chunk(cfg, llm, query, chunk):
    """One batched call over up to _BATCH candidates. Returns {id: (verdict, reason)} or
    None if it could not be parsed after a stricter retry."""
    prompt = _build_prompt(query, chunk)
    for attempt in (1, 2):
        try:
            sys_prompt = _SYSTEM if attempt == 1 else (
                _SYSTEM + "\n\nYour previous reply was not valid JSON. Return ONLY the JSON "
                          "array, nothing else.")
            raw = llm.complete_text(sys_prompt, prompt,
                                    max_tokens=cfg.llm_verify_max_tokens, temperature=0.0)
        except Exception as e:      # noqa: BLE001 -- the LLM is optional in the live path
            log.warning("llm_verify: call failed (%s: %s)", type(e).__name__, e)
            raise                    # let llm_verify's caller turn this into _passthrough
        v = _parse_verdicts(raw, len(chunk))
        if v is not None:
            return v
        log.warning("llm_verify: unparseable output on attempt %d", attempt)
    return None


def _passthrough(candidates, t0, reason):
    """Return everything unverified, flagged. The resilient fallback."""
    for c in candidates:
        c["verify_verdict"] = "unverified"
        c["verify_reason"] = None
    return list(candidates), {
        "unavailable": True, "ms": int((time.time() - t0) * 1000),
        "checked": len(candidates), "kept": len(candidates), "reason": reason}
