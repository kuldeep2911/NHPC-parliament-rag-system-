"""
Line-span extraction contract: the LLM locates, we slice.

The model is shown the reply with line numbers and returns, for every sub-question,
the line range of the question and the line range of the answer that addresses it.
It never emits question or answer PROSE. We slice those ranges out of the source
ourselves, so the model cannot invent, paraphrase, or drop a word — the same
anti-hallucination guarantee the old `answer_block_index` contract gave, but with
the BOUNDARIES decided by the model instead of by per-session regexes.

Why line ranges rather than character offsets: LLMs count characters unreliably but
read line numbers accurately. A span is trivially verifiable either way.

`verify_and_repair` enforces structural invariants that hold for ANY parliamentary
reply in any session -- it never encodes what a particular session's documents look
like:

  * every span is in range, ordered, and non-empty
  * a QUESTION span may not contain an answer marker (Comment:/Answer:/Reply:) --
    this catches the model bleeding a question one line into the answer, the single
    error class seen on the 3043 probe
  * an ANSWER span must start at, or after, its answer marker
  * questions must not overlap each other; part labels must be unique
  * two questions may share one answer span (that is how sharing is expressed)

Anything unrepairable is reported, and the caller falls back to the deterministic
extractor rather than emitting a wrong pairing.
"""

from __future__ import annotations

import re

# The one lexical cue used here. It is not a question-shape rule: it marks where an
# ANSWER begins, and is used only to validate/repair spans the model chose -- never
# to decide how many questions exist or where they start.
#
# Two document generations are covered:
#   current  "Comment:" / "Answer:" / "Reply:" / "उत्तर:"
#   legacy   "Material for reply:" / "Material for Reply of ..." (2014-17 sessions)
# The legacy alternative is listed first so it wins over the bare "reply" alternative,
# which would otherwise never match because it is not at the start of the line.
ANSWER_MARKER = re.compile(
    r"^\s*(material\s+for\s+(?:the\s+)?repl(?:y|ies)|"
    r"comments?|answers?|repl(?:y|ies)|उत्तर)\s*[:.\-]", re.I)

# Marks where the QUESTION block begins in legacy documents ("Questions:" / "Question:").
# Used only to strip the heading from a sliced question; never to count questions.
QUESTION_HEADING = re.compile(r"^\s*(questions?|प्रश्न)\s*[:.\-]", re.I)

# The opening TITLE line of a legacy reply:
#   "Material for Reply of Lok Sabha starred Question Dy. No. 1999 for reply on ... regarding X"
# It names the question and its subject — it is neither a sub-question nor an answer marker.
# The model sometimes emits it as sub-question (a); this pattern lets the validator drop it.
DOC_TITLE_LINE = re.compile(
    r"^\s*material\s+for\s+repl(?:y|ies)\s+of\b.*\bquestion\b", re.I)


def number_lines(text: str) -> str:
    """Render `text` with 0-based line numbers for the prompt."""
    return "\n".join(f"{i:3d}| {ln}" for i, ln in enumerate(text.split("\n")))


def _has_marker(lines, lo, hi) -> bool:
    return any(ANSWER_MARKER.match(lines[i]) for i in range(lo, hi + 1))


def _marker_line(lines, lo, hi):
    for i in range(lo, hi + 1):
        if ANSWER_MARKER.match(lines[i]):
            return i
    return None


def strip_marker(text: str) -> str:
    """Remove a leading Comment:/Answer:/Reply:/Material for reply: from a sliced answer."""
    return ANSWER_MARKER.sub("", text, count=1).strip()


def strip_question_heading(text: str) -> str:
    """Remove a leading 'Questions:' heading from a sliced legacy question."""
    return QUESTION_HEADING.sub("", text, count=1).strip()


def _answer_blocks(lines):
    """
    The document's answer blocks, as [(marker_line, last_line), ...] in order.

    A block starts at an answer marker and runs to the line before the next marker,
    or to end-of-text. Used only to check COVERAGE (every block is claimed by some
    question) and to resolve a null answer -- never to decide how many questions
    exist or where they start. A marker may sit alone on its line with the answer
    text following (5341: "Comment:" then "Not applicable."), so a block is not
    assumed to be one line.
    """
    marks = [i for i, l in enumerate(lines) if ANSWER_MARKER.match(l)]
    out = []
    for k, m in enumerate(marks):
        end = (marks[k + 1] - 1) if k + 1 < len(marks) else len(lines) - 1
        out.append((m, max(m, end)))
    return out


def slice_lines(lines, span) -> str:
    """
    Join a line range into one string.

    Runs of whitespace are collapsed: text extracted from a JUSTIFIED PDF arrives
    padded ("whether  it  is  a  fact"), and that padding would otherwise be stored
    in parsed.json and embedded verbatim in Phase 3.
    """
    lo, hi = span
    joined = " ".join(l.strip() for l in lines[lo:hi + 1] if l.strip())
    return re.sub(r"\s+", " ", joined).strip()


def _coerce_span(v):
    """Accept [a,b] / (a,b) / {'start':a,'end':b}; return (a,b) ints or None."""
    if isinstance(v, dict):
        v = [v.get("start", v.get("first")), v.get("end", v.get("last"))]
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        return None
    try:
        lo, hi = int(v[0]), int(v[1])
    except (TypeError, ValueError):
        return None
    return (lo, hi)


def verify_and_repair(obj, text):
    """
    Validate the model's line spans against `text`.

    Returns (sub_questions, errors, repairs) where sub_questions is a list of
      {part_label, question_lines, answer_lines, question_text, answer_text}
    with the text SLICED FROM `text` (never from the model). A non-empty `errors`
    means the result must not be used.
    """
    lines = text.split("\n")
    n = len(lines)
    errors, repairs = [], []

    raw = obj.get("sub_questions") if isinstance(obj, dict) else None
    if not isinstance(raw, list) or not raw:
        return [], ["no sub_questions returned"], []

    out = []
    for i, sq in enumerate(raw):
        if not isinstance(sq, dict):
            errors.append(f"sub_question[{i}] not an object")
            continue
        label = str(sq.get("part_label") or "").strip().lower().strip("()。.")
        qs = _coerce_span(sq.get("question_lines"))
        as_ = _coerce_span(sq.get("answer_lines"))

        if qs is None:
            errors.append(f"sub_question[{i}] bad question_lines")
            continue

        qlo, qhi = qs
        if not (0 <= qlo <= qhi < n):
            errors.append(f"sub_question[{i}] question_lines {qs} out of range 0..{n-1}")
            continue

        # REPAIR: a question span must not swallow the answer marker. The 3043 probe
        # returned b.question_lines=[6,7] where line 7 is "Comment: ...".
        mk = _marker_line(lines, qlo, qhi)
        if mk is not None:
            if mk == qlo:
                errors.append(
                    f"sub_question[{i}] question_lines {qs} starts at an answer marker")
                continue
            repairs.append(f"({label}) question_lines {qs} -> [{qlo},{mk-1}] (dropped answer marker)")
            qhi = mk - 1
            qs = (qlo, qhi)

        # answer span is optional: a question may genuinely have no answer block
        ans_text = ""
        if as_ is not None:
            alo, ahi = as_
            if not (0 <= alo <= ahi < n):
                errors.append(f"sub_question[{i}] answer_lines {as_} out of range")
                continue
            # REPAIR: align the answer span with its marker line.
            #   forward  - the span starts a line or two BEFORE the marker
            #   backward - the marker sits alone on its own line and the model started
            #              at the text below it (5341: "Comment:" on 10, text on 11).
            # Both keep the coverage check below honest: a block is claimed only when
            # some answer span actually contains its marker line.
            if not ANSWER_MARKER.match(lines[alo]):
                mk2 = _marker_line(lines, alo, min(ahi, alo + 2))
                if mk2 is not None and mk2 != alo:
                    repairs.append(f"({label}) answer_lines {as_} -> [{mk2},{ahi}] (snapped forward to marker)")
                    alo = mk2
                    as_ = (alo, ahi)
                elif alo > 0 and ANSWER_MARKER.match(lines[alo - 1]) \
                        and not strip_marker(lines[alo - 1]).strip():
                    # bare marker directly above, with no text of its own
                    repairs.append(f"({label}) answer_lines {as_} -> [{alo-1},{ahi}] (snapped back to bare marker)")
                    alo = alo - 1
                    as_ = (alo, ahi)
            if alo > ahi:
                errors.append(f"sub_question[{i}] answer_lines empty after repair")
                continue
            ans_text = strip_marker(slice_lines(lines, as_))

        # Legacy documents put a bare "Questions:" heading above the first sub-question;
        # it is a section label, not part of the question, so drop it.
        q_text = strip_question_heading(slice_lines(lines, qs))
        if not q_text:
            errors.append(f"sub_question[{i}] question span is blank")
            continue

        # REPAIR: the model sometimes emits the document's TITLE line as sub-question (a)
        # ("Material for Reply of Lok Sabha Question Dy. No. 1999 ... regarding POWER
        # TARIFF"). That is the heading naming the question, not a question — drop it and
        # keep the real sub-questions. Dropping (rather than erroring) is right because the
        # rest of the span set is usually correct.
        if DOC_TITLE_LINE.match(q_text):
            repairs.append(f"dropped sub_question[{i}]: document title line, not a question")
            continue

        out.append({
            "part_label": label or chr(ord("a") + len(out)),
            "question_lines": list(qs),
            "answer_lines": list(as_) if as_ else None,
            "question_text": q_text,
            "answer_text": ans_text,
            # carried through so callers never have to re-key the raw model output by
            # part_label -- unlabeled replies come back with part_label null and the
            # label above is SYNTHESISED, so a raw-output lookup would miss.
            "answer_table_ids": [str(t) for t in (sq.get("answer_table_ids") or [])],
            "answer_is_table": bool(sq.get("answer_is_table")),
            "confidence": sq.get("confidence") if sq.get("confidence") in ("high", "low") else "high",
        })

    if not out:
        errors.append("no usable sub_questions after verification")
        return [], errors, repairs

    out.sort(key=lambda d: d["question_lines"][0])

    # ---- REPAIR: a null answer_lines is the model declining to decide -----------
    # Every question in a reply is answered by the first answer BLOCK that begins
    # after it. If the following question already claims that block, the two SHARE
    # it (the "d) ... e) ... Comment: <one answer>" shape). The model returns null
    # here rather than expressing the share (2392: d=null while e took line 10).
    blocks = _answer_blocks(lines)          # [(marker_line, last_line)] in order
    if blocks:
        # Prefer the span a SIBLING already assigned to the same block, so a shared
        # answer is expressed identically (group_answers keys on span equality).
        claimed_by_marker = {}
        for sq in out:
            if sq["answer_lines"]:
                claimed_by_marker.setdefault(sq["answer_lines"][0], sq["answer_lines"])
        for sq in out:
            if sq["answer_lines"] is not None:
                continue
            qstart = sq["question_lines"][0]
            nxt = next((b for b in blocks if b[0] > qstart), None)
            if nxt is None:
                continue
            span = claimed_by_marker.get(nxt[0]) or [nxt[0], nxt[1]]
            sq["answer_lines"] = list(span)
            sq["answer_text"] = strip_marker(slice_lines(lines, sq["answer_lines"]))
            repairs.append(
                f"({sq['part_label']}) answer_lines null -> {sq['answer_lines']} "
                f"(first block after the question; shares it with a sibling if claimed)")

    # ---- INVARIANT: every answer block must be consumed -------------------------
    # A reply's answer blocks all belong to some question. If the model silently
    # drops one (8826 line 9 'not applicable.'; 5341 lines 10-11 the CERC/R&M
    # answer), the output is missing document content -- reject rather than emit a
    # parsed.json whose answers are incomplete. This is the guarantee that catches
    # the model quietly under-answering.
    if blocks:
        claimed = set()
        for sq in out:
            if sq["answer_lines"]:
                lo, hi = sq["answer_lines"]
                for b in blocks:
                    if lo <= b[0] <= hi:
                        claimed.add(b[0])
        missed = [b for b in blocks if b[0] not in claimed]
        if missed:
            preview = "; ".join(
                f"line {b[0]}: {strip_marker(lines[b[0]])[:44]!r}" for b in missed[:3])
            errors.append(
                f"{len(missed)} answer block(s) not assigned to any question "
                f"({preview}) -- every Comment:/Answer: block must be used")

    # NOTE: an answer may legitimately PRECEDE its question. 5341 answers "Whether the
    # efficiency has reduced?" with a detailed block, and the follow-up "if so, the
    # details thereof" is printed AFTER that block but answered by it. So there is no
    # "answer must follow its question" invariant -- asserting one rejected the
    # model's correct reading three attempts running.

    # questions must not overlap one another
    for a, b in zip(out, out[1:]):
        if b["question_lines"][0] <= a["question_lines"][1]:
            errors.append(
                f"question spans overlap: ({a['part_label']}){a['question_lines']} "
                f"and ({b['part_label']}){b['question_lines']}")

    labels = [d["part_label"] for d in out]
    if len(set(labels)) != len(labels):
        errors.append(f"duplicate part_labels: {labels}")

    return out, errors, repairs


def table_previews(table_outs, max_rows=3, max_cols=6, cell_chars=28):
    """
    Compact, prompt-sized previews of each table so the model can tell what a table
    IS ABOUT and hand it to the right sub-question.

    Only headers and the first couple of data rows are sent — enough to recognise
    "GENERATION | DESIGN ENERGY | GENERATION (MU)" as the generation table, without
    pasting hundreds of rows into the prompt.
    """
    out = []
    for t in table_outs:
        cols = [str(getattr(c, "name", c))[:cell_chars]
                for c in (t.columns or [])[:max_cols]]
        rows = []
        for r in (t.rows or [])[:max_rows]:
            cells = getattr(r, "cells", r)          # Row.cells is a {column: value} dict
            vals = list(cells.values()) if isinstance(cells, dict) else list(cells or [])
            vals = [str(v)[:cell_chars] for v in vals[:max_cols]]
            if vals:
                rows.append(vals)
        out.append({
            "table_id": t.table_id,
            "caption": (t.caption or "")[:120],
            "columns": cols,
            "sample_rows": rows,
        })
    return out


def group_answers(sub_questions):
    """
    Collapse identical answer spans into shared answer groups.

    Two sub-questions that returned the SAME answer_lines share one answer — that is
    the model's way of saying so, and it needs no rule to detect.
    """
    groups, by_span = [], {}
    for sq in sub_questions:
        key = tuple(sq["answer_lines"]) if sq["answer_lines"] else ("none", sq["part_label"])
        if key not in by_span:
            by_span[key] = {
                "answer_lines": sq["answer_lines"],
                "answer_text": sq["answer_text"],
                "parts": [],
            }
            groups.append(by_span[key])
        by_span[key]["parts"].append(sq["part_label"])
    return groups
