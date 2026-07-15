"""
DRAFT-ASSIST — a grounded, cited draft reply built from the retrieved past answers.

This is what the officer actually uses. It is NOT the old graph node (draft.py): that one
runs INSIDE the LangGraph and would make every /query wait for the LLM. This runs from its
own endpoint, after the results are already on screen, so retrieval never slows down.

WHAT IT PRODUCES
    draft_answer  point-wise when the question has sub-parts, in NHPC's own register
    key_points    the facts the reply must contain, each cited
    gaps          what the past answers DO NOT cover -- stated, never invented
    citations     every point traces to a real past reply the officer can open

THE THREE THINGS THAT MAKE THIS SAFE

1. STRICT GROUNDING. The model may use only the retrieved answers. It is told, plainly,
   that a fabricated figure in a parliamentary reply is a serious error.

2. ANSWER-TYPE AWARENESS, COMPUTED IN CODE. We do not ask the model to notice the pattern;
   we count it ourselves and tell it. If NHPC has always said "May be replied by Ministry of
   Power" on a topic, the draft must mirror that -- NOT invent a substantive NHPC position.
   This is the single most dangerous failure mode of the whole feature: a confident,
   well-written NHPC position on a subject where NHPC has never taken one, going to
   Parliament.

3. CITATIONS ARE VERIFIED, NOT TRUSTED. Every citation the model emits is checked against
   the doc_keys actually fed in. An invented one is STRIPPED and its claim marked uncited.
   Same discipline as the span extractor: the model is trusted to FIND and to WRITE, never
   to be right about its own output.

RESILIENCE: an LLM failure returns draft=None with a reason. The officer keeps their
results. Losing the draft beats losing the results.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter

from nhpc_qa.core.logging import get_logger

log = get_logger("nhpc.draft")


_SYSTEM = """You help NHPC officers draft replies to Parliament.

You are given the officer's NEW question and the PAST parliamentary questions NHPC has
already answered, with the answers it gave. Draft a reply to the new question.

═══ ABSOLUTE RULES — an invented fact in a parliamentary reply is a serious error ═══

1. GROUND EVERYTHING. Use ONLY facts that appear in the PAST ANSWERS below. Never add a
   figure, capacity, date, project name, status or policy that is not there. Do not use
   outside knowledge, however confident you are.

2. CITE EVERY CLAIM. Each point carries the citation of the past reply it came from,
   exactly as given: "2023-jul-aug lok_sabha 8779". Never cite a reply that is not listed.

3. NEVER FILL A GAP. If the past answers do not cover part of the new question, say so in
   `gaps`. Do NOT write a plausible-sounding answer for it. An honest gap is useful; an
   invented answer is dangerous.

4. MIRROR THE HISTORICAL PATTERN. The ANSWER PATTERN section tells you what NHPC has
   actually done on this subject. Follow it:
   - deferred_to_ministry -> draft the deferral ("May be replied by Ministry of Power. As
     far as NHPC is concerned, ...") -- do NOT invent an NHPC position.
   - not_applicable -> say it does not pertain to NHPC.
   - nil -> say the information may be treated as Nil, with NHPC's standard qualification.
   - substantive -> draft substantively FROM the past answers.
   Where NHPC has historically NOT taken a position, you must NOT take one for it.

5. MATCH THE STYLE. Write in the register of the past answers: formal government reply,
   third person, "NHPC" / "NHPC Ltd.", the same standard phrasings and qualifications.
   Do not editorialise. No preamble, no headings, no restating the question.

6. MATCH THE LANGUAGE. If the officer's question is in Hindi, draft in Hindi. If the past
   answers are in Hindi, keep their terminology. Never force-translate.

7. STRUCTURE. If the new question has sub-parts ((a), (b), (c)...), answer point-wise with
   one entry per part. Otherwise write a single part with label "".

Return STRICT JSON only, no prose outside it:

{
  "language": "en" | "hi",
  "pattern": "substantive" | "deferred_to_ministry" | "nil" | "not_applicable" | "mixed",
  "parts": [
    {"label": "(a)", "text": "<the drafted reply for this part>",
     "cites": ["2023-jul-aug lok_sabha 8779"]}
  ],
  "key_points": [
    {"point": "<a fact the reply must contain — a figure, project, status, caveat>",
     "cites": ["2023-jul-aug lok_sabha 8779"]}
  ],
  "gaps": [
    {"part": "(c)", "reason": "<what the past answers do not cover>"}
  ]
}"""


def _answer_pattern(results):
    """
    What has NHPC ACTUALLY done on this subject? Counted in code, never inferred by the
    model.

    This is the guardrail that stops the most dangerous failure: a fluent, confident NHPC
    position on a topic where NHPC has consistently deferred to the Ministry. The model is
    good at writing; it is not the right thing to trust with "should we take a position at
    all". So we count, and we tell it.
    """
    types = [r.get("answer_type") for r in results if r.get("answer_type")]
    if not types:
        return "mixed", Counter(), ""

    counts = Counter(types)
    top, n = counts.most_common(1)[0]
    dominant = top if n >= max(2, len(types) * 0.5) else "mixed"

    breakdown = ", ".join(f"{n} {t}" for t, n in counts.most_common())
    if dominant == "deferred_to_ministry":
        guidance = ("NHPC has historically DEFERRED this subject to the Ministry. Draft the "
                    "deferral. Do NOT invent a substantive NHPC position.")
    elif dominant == "not_applicable":
        guidance = ("This subject has historically NOT PERTAINED to NHPC. Say so. Do NOT "
                    "manufacture NHPC content for it.")
    elif dominant == "nil":
        guidance = ("NHPC has historically returned NIL information on this subject. Draft "
                    "that, with NHPC's standard qualification.")
    elif dominant == "substantive":
        guidance = ("NHPC has historically answered this SUBSTANTIVELY. Draft substantively, "
                    "strictly from the facts in the past answers.")
    else:
        guidance = ("The past answers are MIXED. Follow the pattern of the most relevant "
                    "ones; where NHPC deferred, mirror the deferral rather than inventing a "
                    "position.")
    return dominant, counts, f"{breakdown}. {guidance}"


def _cite(r):
    """The citation string. Stable, and exactly what the model is told to echo back."""
    return f"{r['session']} {r['house']} {r['diary_number']}"


def _context(results, k):
    """Render the retrieved answers as the ONLY facts the model may use."""
    blocks = []
    for r in results[:k]:
        if not (r.get("answer_text") or "").strip():
            continue          # nothing to ground on; a citation to it would be empty
        parts = r.get("answer_covers_parts") or []
        blocks.append(
            f"[{_cite(r)}]"
            f"{'  answered ' + r['reply_date'] if r.get('reply_date') else ''}\n"
            f"  PAST QUESTION {r.get('part_label') or ''}: {r['question_text']}\n"
            f"  NHPC'S ANSWER [{r['answer_type']}]"
            f"{' (covers parts ' + ', '.join(parts) + ')' if parts else ''}: "
            f"{r['answer_text']}")
    return "\n\n".join(blocks)


_JSON_RE = re.compile(r"\{.*\}", re.S)


def _parse_json(raw: str):
    """The model's JSON, tolerantly. A thinking model may wrap it in prose or a fence."""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = _JSON_RE.search(s)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def _verify_cites(obj, allowed: dict):
    """
    ⚠️ CITATIONS ARE VERIFIED, NEVER TRUSTED. ⚠️

    Every citation the model emits is checked against the replies actually fed to it. An
    invented one is STRIPPED, and the point it supported is marked uncited so the officer
    can see that it has no source.

    A fabricated citation is worse than no citation: it looks verifiable. An officer
    checking a draft sees "[2019-feb lok_sabha 4412]" and reasonably assumes someone could
    open it.

    Returns the number of citations dropped.
    """
    dropped = 0

    def clean(lst):
        nonlocal dropped
        out = []
        for c in (lst or []):
            key = str(c).strip()
            if key in allowed:
                if key not in out:
                    out.append(key)
            else:
                dropped += 1
                log.warning("draft: dropped an INVENTED citation %r", key[:60])
        return out

    for p in obj.get("parts") or []:
        p["cites"] = clean(p.get("cites"))
        p["uncited"] = not p["cites"]
    for kp in obj.get("key_points") or []:
        kp["cites"] = clean(kp.get("cites"))
        kp["uncited"] = not kp["cites"]
    return dropped


def _empty_parts_become_gaps(obj):
    """
    An EMPTY drafted part is a GAP, not a part. Enforced in code, not left to the model.

    Asked about a sub-part the past answers cannot support, the model correctly refuses to
    invent an answer -- but it does not always express that refusal the same way. Observed
    across runs: sometimes it writes an explicit `gaps` entry (right), and sometimes it
    emits the part with an EMPTY text and no citation (wrong).

    A blank "(c)" in a draft an officer may paste into a reply to Parliament is worse than
    an explicit "no past answer covers this". So an empty part is MOVED to gaps here. The
    model's variability cannot reach the officer.
    """
    kept, moved = [], []
    for p in obj.get("parts") or []:
        if (p.get("text") or "").strip():
            kept.append(p)
        else:
            moved.append({
                "part": p.get("label") or "",
                "reason": "no past reply covers this — officer input needed",
            })
    if moved:
        obj["parts"] = kept
        gaps = obj.get("gaps") or []
        have = {(g.get("part") or "").strip() for g in gaps}
        for g in moved:
            if g["part"].strip() not in have:      # do not duplicate a gap it already flagged
                gaps.append(g)
        obj["gaps"] = gaps
        log.info("draft: %d empty part(s) moved to gaps", len(moved))


def build_draft(cfg, llm, query: str, results: list, run_id: str | None = None) -> dict:
    """
    Build the draft. Returns a dict with `ok`; NEVER raises.

        {"ok": True,  "draft": {...}}
        {"ok": False, "error": "...", "reason": "<human-readable>"}

    A failure here must never cost the officer their retrieval results.
    """
    t0 = time.time()
    k = cfg.draft_context_k
    usable = [r for r in results[:k] if (r.get("answer_text") or "").strip()]

    if not usable:
        return {"ok": False, "reason": "none of the retrieved results has an answer to "
                                       "ground a draft on"}

    pattern, counts, pattern_note = _answer_pattern(usable)
    allowed = {_cite(r): r for r in usable}

    user = (
        f"OFFICER'S NEW QUESTION:\n{query}\n\n"
        f"ANSWER PATTERN — what NHPC has actually done on this subject:\n{pattern_note}\n\n"
        f"PAST ANSWERS — the ONLY facts you may use:\n\n{_context(usable, k)}\n\n"
        f"Draft the reply. Cite every point. Flag anything the past answers do not cover.")

    try:
        raw = llm.complete_text(_SYSTEM, user,
                                max_tokens=cfg.draft_max_tokens,
                                temperature=cfg.draft_temperature)
    except Exception as e:      # noqa: BLE001 -- OPTIONAL layer, degrade never break
        log.warning("draft: llm failed (%s: %s)", type(e).__name__, e)
        return {"ok": False, "error": f"{type(e).__name__}",
                "reason": "the drafting model is unavailable"}

    obj = _parse_json(raw)
    if not isinstance(obj, dict) or not obj.get("parts"):
        log.warning("draft: unusable model output (%d chars)", len(raw or ""))
        return {"ok": False, "error": "unparseable",
                "reason": "the drafting model returned nothing usable"}

    dropped = _verify_cites(obj, allowed)
    _empty_parts_become_gaps(obj)

    # The full record of what the draft was built from, so an officer -- or an auditor --
    # can open every source. This is not decoration: a government reply must be traceable.
    sources = [{
        "citation": _cite(r),
        "doc_key": r["doc_key"],
        "session": r["session"], "house": r["house"],
        "diary_number": r["diary_number"],
        "reply_date": r.get("reply_date"),
        "answer_type": r["answer_type"],
        "question_text": r["question_text"],
        "answer_text": r["answer_text"],
        "file_available": bool((r.get("reply_file") or {}).get("available")),
    } for r in usable]

    draft = {
        # Unmissable, and repeated in the UI and the DOCX. This is not an approved reply.
        "status": "DRAFT — FOR OFFICER REVIEW",
        "notice": ("Generated from NHPC's past replies only. Verify every figure and claim "
                   "against the cited sources before use. This is not an approved reply."),
        "language": obj.get("language") or "en",
        "pattern": obj.get("pattern") or pattern,
        "pattern_counts": dict(counts),
        "parts": obj.get("parts") or [],
        "key_points": obj.get("key_points") or [],
        "gaps": obj.get("gaps") or [],
        "sources": sources,
        "run_id": run_id,
        "model": getattr(llm, "model", None) or getattr(llm, "name", "llm"),
        "citations_dropped": dropped,     # >0 means the model invented some; they are gone
        "ms": int((time.time() - t0) * 1000),
    }
    log.info("draft: %d part(s), %d key point(s), %d gap(s), pattern=%s, %d invented "
             "citation(s) dropped, %dms",
             len(draft["parts"]), len(draft["key_points"]), len(draft["gaps"]),
             draft["pattern"], dropped, draft["ms"])
    return {"ok": True, "draft": draft}
