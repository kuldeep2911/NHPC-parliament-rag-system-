"""
NODE 6 — OPTIONAL answer drafting. OFF by default (GENERATION_ENABLED=false).

WHAT THIS IS: an LLM synthesises a draft answer STRICTLY from the retrieved past answers,
citing session/house/year/file for every claim. It is explicitly marked as a DRAFT FOR
OFFICER REVIEW, never as an authoritative reply.

WHAT THIS IS NOT: a source of new facts. The model is instructed to use ONLY the retrieved
context and to say so plainly when the context does not answer the question. Nothing here
may introduce a figure, a date, or a project that is not in the retrieved answers -- the
officer is accountable for what goes back to Parliament.

RESILIENCE: this is an OPTIONAL layer. If the LLM is unreachable, slow, or returns
nonsense, the officer still gets their retrieved results -- the failure is caught, logged
and recorded in state["errors"], never raised. Losing the draft beats losing the results.

PHASE COUPLING (Change 5): all generation logic lives HERE, in phase4. Only the generic
transport (complete_text) lives with the other providers in phase2/providers.py. phase2
imports nothing from phase4, so there is no cycle.
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger("nhpc.phase4.generation")

_SYSTEM = """You draft answers for NHPC officers replying to Parliament.

You are given PAST parliamentary questions and the answers NHPC already gave. Draft a
reply to the officer's new question using ONLY those past answers.

ABSOLUTE RULES — you are drafting for Parliament, and an invented fact is a serious error:
1. Use ONLY facts that appear in the CONTEXT below. Never add a figure, date, capacity,
   project or policy that is not there. Do not use outside knowledge.
2. CITE every claim inline as [session house diary] exactly as given in the context,
   e.g. [2020-feb-mar lok_sabha 8773].
3. If the context does not answer the question, SAY SO plainly and state what IS covered.
   A short honest answer is correct; a padded one is not.
4. Do not speculate, and do not soften an absence of information into a guess.
5. If the question is in Hindi, answer in Hindi. Otherwise answer in English.

Write plain prose for a senior officer. No preamble, no headings, no restating the
question."""


def _context(results, limit=6):
    """Render the retrieved answers as the ONLY facts the model may use."""
    blocks = []
    for r in results[:limit]:
        if not (r.get("answer_text") or "").strip():
            continue
        cite = f"{r['session']} {r['house']} {r['diary_number']}"
        blocks.append(
            f"[{cite}]\n"
            f"  PAST QUESTION: {r['question_text']}\n"
            f"  NHPC'S ANSWER ({r['answer_type']}): {r['answer_text']}\n"
            f"  SOURCE FILE: {'reply on record' if r['reply_file']['available'] else 'not on file'}")
    return "\n\n".join(blocks)


def generate_draft(state, deps):
    """
    Draft a cited answer from the retrieved results.

    Returns {"draft": {...}} or {"draft": None} + an error note. NEVER raises: a
    generation failure must not cost the officer their retrieval results.
    """
    t0 = time.time()
    cfg = deps["cfg"]
    results = state.get("results") or []

    if not results:
        return {"draft": None}

    ctx = _context(results)
    if not ctx.strip():
        return {"draft": None}

    llm = deps.get("llm")
    if llm is None:
        # Built LAZILY so nothing is constructed when generation is off.
        #
        # get_llm() expects a PHASE-2 Config (it calls resolve_backends(), which only
        # exists there). Phase4Config extends Phase3Config, which does not have it, so
        # we build the phase-2 config here rather than bolt phase-2 backend-resolution
        # onto the phase-4 config. This also keeps the phases decoupled: phase4 reuses
        # phase2's provider seam without phase2 knowing phase4 exists.
        from phase2.config import Config as Phase2Config
        from phase2.providers import get_llm
        try:
            llm = get_llm(Phase2Config())
            deps["llm"] = llm
        except Exception as e:              # noqa: BLE001
            log.warning("generation: no LLM available (%s: %s)", type(e).__name__, e)
            return {"draft": None,
                    "errors": (state.get("errors") or []) + [f"generation: {type(e).__name__}"]}

    user = (f"OFFICER'S QUESTION:\n{state['query']}\n\n"
            f"CONTEXT — the only facts you may use:\n\n{ctx}")

    try:
        text = llm.complete_text(_SYSTEM, user,
                                 max_tokens=cfg.generation_max_tokens,
                                 temperature=0.0)
    except Exception as e:                  # noqa: BLE001 -- OPTIONAL layer, degrade
        log.warning("generation failed (%s: %s) — returning retrieval results only",
                    type(e).__name__, e)
        return {"draft": None,
                "errors": (state.get("errors") or []) + [f"generation: {type(e).__name__}"],
                "timings_ms": {**(state.get("timings_ms") or {}),
                               "generate": int((time.time() - t0) * 1000)}}

    draft = {
        # unmissable: this is not an answer, it is a draft
        "status": "DRAFT FOR OFFICER REVIEW — not an approved reply",
        "text": text.strip(),
        # exactly the documents the model was allowed to use, so a citation can be checked
        "grounded_in": [{
            "citation": f"{r['session']} {r['house']} {r['diary_number']}",
            "doc_key": r["doc_key"],
            "answer_type": r["answer_type"],
        } for r in results[:6] if (r.get("answer_text") or "").strip()],
        "model": getattr(llm, "name", cfg.embed_model),
        "_warning": ("Generated from the retrieved past answers only. Verify every claim "
                     "against the cited source files before use."),
    }
    return {"draft": draft,
            "timings_ms": {**(state.get("timings_ms") or {}),
                           "generate": int((time.time() - t0) * 1000)}}
