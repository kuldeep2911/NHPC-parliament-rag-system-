"""
Sigmoid relevance filter + batched LLM similarity verification.

Pipeline position:  rerank -> [sigmoid_filter -> llm_verify] -> date-order -> display.
This module is the two middle stages. It runs AFTER rerank and BEFORE the display-date
ordering; it does not touch retrieval, RRF, doc_key, the halfvec SQL, or query-mode
embedding.

WHY TWO STAGES, AND WHY THE SIGMOID IS THE WEAK ONE.

Calibration (scratchpad/calibrate2.py, on llama-nemotron-rerank-1b-v2) measured the
reranker logit of every candidate across 12 labelled queries and proved the obvious design
impossible: the sigmoid of a genuine match and the sigmoid of boilerplate noise OVERLAP
COMPLETELY. The lowest real match scored 0.003; the phrase "details thereof" scored 0.9999
against 46 documents. No single sigmoid value separates them.

So the sigmoid filter is NOT the precision gate -- it is a cheap, permissive RECALL filter
that only bounds how many candidates the LLM has to read. The LLM verify pass is what
actually judges "is this the same question", which a scalar score cannot.

    sigmoid filter :  keep sigmoid >= SIMILARITY_THRESHOLD  (default 0.05, low on purpose)
    llm verify     :  ONE batched call -> drop the ones it says are not the same question

RESILIENCE. The LLM is in the live path now, so it WILL sometimes be slow or down. When it
fails, we return the sigmoid-filtered set UNVERIFIED with verification_unavailable=True --
never an error, never an empty result caused by the model being unreachable. Retrieval must
not depend on the LLM being up.
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
    Attach `relevance` (sigmoid of the logit) to every candidate and keep those at or above
    the threshold. Returns (kept, dropped_count).

    A candidate with a null logit (the reranker degraded to RRF order) is KEPT with
    relevance=None: we cannot score it, and dropping it would turn a reranker outage into
    silent data loss. It simply is not sigmoid-filtered.

    `safety_max` bounds the survivors -- a guard against a degenerate query, not a relevance
    cap. It applies to the already-sorted-by-relevance list, so it only ever trims the
    weakest.
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


# The most candidates to judge in ONE call. A batch of 46 asks the model for a 46-element
# JSON array, which overflowed the token budget and truncated mid-array -> unparseable ->
# the whole batch fell back to "unverified, keep everything". Measured: exactly what let
# "details thereof" through with 46 results. Batches of this size are a boilerplate query
# anyway (a specific question does not have 46 genuine matches), but the verifier must
# stay CORRECT for them, not just fail soft. So we chunk.
_BATCH = 12


def llm_verify(cfg, llm, query: str, candidates: list):
    """
    Batched verification. Returns (verified, meta).

      verified : the candidates the LLM judged similar, each annotated with verify_verdict
                 and verify_reason. Order preserved (relevance order).
      meta     : {"unavailable": bool, "ms": int, "checked": int, "kept": int,
                  "reason": <why unavailable, if so>}

    Chunked at _BATCH so a large candidate set does not force one enormous JSON array that
    truncates and fails to parse. Each chunk is one call; a chunk that fails to parse after
    a retry makes THE WHOLE verification unavailable (fail soft, flagged) rather than
    silently dropping that chunk's results.

    NEVER raises. On any failure the sigmoid set is returned unverified with
    meta["unavailable"]=True, so an LLM outage never costs the officer their results.
    """
    t0 = time.time()
    if not candidates:
        return [], {"unavailable": False, "ms": 0, "checked": 0, "kept": 0}

    verified = []
    for start in range(0, len(candidates), _BATCH):
        chunk = candidates[start:start + _BATCH]
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
            if verdict == "similar":
                verified.append(c)
            else:
                log.info("llm_verify: dropped %s — %s", c.get("doc_key"), reason[:80])

    ms = int((time.time() - t0) * 1000)
    log.info("llm_verify: checked %d, kept %d, %d ms (%d chunk(s))",
             len(candidates), len(verified), ms,
             (len(candidates) + _BATCH - 1) // _BATCH)
    return verified, {"unavailable": False, "ms": ms,
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
