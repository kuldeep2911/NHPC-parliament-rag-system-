"""
Q&A extraction — two paths behind one swappable interface.

`extract_qa(doc, layout, meta, cfg, backend)` returns (pairs, tables_index, flags).

PATH 1 — PROSE (mostly_prose / prose_with_tables):
    Split the reply into question-number-anchored segments, pair questions to
    answers across layout cases A/B/C, split clubbed answers (Case C) and record
    related_question_numbers. Attach Type-1/2 tables to the relevant answer.

PATH 2 — QA-TABLE (Table-Type-3):
    Read one Q&A pair per row from the identified question/answer columns.

The LLM call is abstracted: when the backend is a real model, we send the IR +
strict-JSON schema and use the model's structuring. With the deterministic
backend (default) we compute the structure with rules and pass it THROUGH the
same interface, so switching to a model changes one config value, not this code.
"""

from __future__ import annotations

import json
import re

from .ir import Pair, TableOut, detect_language
from .tables import build_table_out, clean_table
from .llm import BackendError

QNUM_RE = re.compile(
    r"\b(?:Q\.?\s*No\.?|Question\s*No\.?|Dy\.?\s*No\.?|Diary\s*No\.?|"
    r"USQ|LSUSQ|RSUQ|Starred\s*Question|Unstarred\s*Question)\s*[:.]?\s*"
    r"([A-Z]?\s*\d{2,6})",
    re.I,
)
INLINE_QNUM_RE = re.compile(r"\b([USL]?\s?\d{3,6})\b")


# ---------------------------------------------------------------------------
# JSON schema (validated on every extraction, whichever backend produced it)
# ---------------------------------------------------------------------------

def validate_pairs(obj) -> list:
    """Validate the extraction result dict. Returns list of error strings (empty=ok)."""
    errs = []
    if not isinstance(obj, dict):
        return ["result is not an object"]
    if "pairs" not in obj or not isinstance(obj["pairs"], list):
        return ["missing 'pairs' list"]
    for i, p in enumerate(obj["pairs"]):
        if not isinstance(p, dict):
            errs.append(f"pair[{i}] not object"); continue
        if "question_number" not in p:
            errs.append(f"pair[{i}] missing question_number")
        if "question_text" not in p and not p.get("answer_is_table"):
            errs.append(f"pair[{i}] missing question_text")
        if "answer_text" not in p and not p.get("answer_is_table"):
            errs.append(f"pair[{i}] missing answer_text")
        conf = p.get("confidence", "high")
        if conf not in ("high", "low"):
            errs.append(f"pair[{i}] bad confidence '{conf}'")
    return errs


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def extract_qa(doc, layout, meta, cfg, backend, tracer=None):
    """Route to the right path, run the backend, validate, return outputs."""
    flags = list(layout.get("flags", []))
    qid = meta.get("question_id", "q")

    if layout["structure"] == "qa_table":
        pairs, tindex, f2 = _extract_qa_table(doc, layout, qid, meta)
        flags += f2
        if tracer:
            tracer.step("extraction", {
                "path": "qa_table", "structure": layout["structure"],
                "n_pairs": len(pairs), "flags": f2,
                "raw_model_output": None, "note": "deterministic table read"},
                model_name="deterministic", duration_ms=None)
    else:
        pairs, tindex, f2 = _extract_prose(doc, layout, qid, meta, cfg, backend, tracer)
        flags += f2

    # metadata cross-check on question numbers
    _crosscheck_metadata(pairs, meta, flags)

    # LLM cross-check: when enabled AND a real model is available, ask the LLM to
    # independently count the questions and distinct answers in the reply, and
    # compare with the deterministic split. Disagreement -> flag for review (the
    # deterministic result is kept; the LLM is a second opinion, not the source).
    if getattr(cfg, "llm_crosscheck", False) and layout["structure"] != "qa_table":
        _llm_crosscheck(doc, pairs, cfg, backend, flags, tracer)

    return pairs, tindex, _dedup(flags)


def _llm_crosscheck(doc, pairs, cfg, backend, flags, tracer=None):
    """
    Second-opinion check: have the LLM count questions + distinct answers and
    compare to the deterministic split. On disagreement add 'llm_crosscheck_disagree'
    (review). Never overwrites the deterministic result; never crashes the run.
    """
    if not _is_real_model(backend):
        return
    text = doc.full_text()[:12000]
    system = (
        "You audit a parliamentary reply. Count structure ONLY, do not extract text. "
        "Return STRICT JSON: {\"n_questions\":int,\"n_distinct_answers\":int}. "
        "A sub-part is one lettered/numbered question (a),(b),(c)... or one distinct "
        "question sentence. n_distinct_answers = how many DIFFERENT answer blocks "
        "there are (a 'Comment:'/'Answer:' introduces each). If several questions "
        "share one answer, count that answer once.")
    user = json.dumps({"reply_text": text}, ensure_ascii=False)
    import time
    det_q = len(pairs)
    det_a = len({(p.answer_text or "").strip()[:80] for p in pairs
                 if (p.answer_text or "").strip()})
    try:
        t0 = time.time()
        obj = backend.complete_json(system, user)
        dt = int((time.time() - t0) * 1000)
        llm_q = int(obj.get("n_questions", 0))
        llm_a = int(obj.get("n_distinct_answers", 0))
        # Small local models are UNRELIABLE at counting sub-questions (they over-
        # split clauses), so the question count is not a trustworthy signal. The
        # number of DISTINCT ANSWERS is what indicates a grouping error, and the
        # model tracks it well. Flag only when the distinct-answer count differs by
        # more than one (tolerate off-by-one from model noise).
        disagree = bool(llm_a) and abs(llm_a - det_a) >= 2
        if tracer:
            tracer.step("llm_crosscheck", {
                "deterministic": {"n_questions": det_q, "n_distinct_answers": det_a},
                "llm": {"n_questions": llm_q, "n_distinct_answers": llm_a},
                "disagree": bool(disagree), "raw_model_output": obj},
                model_name=backend.model_for("llm") if hasattr(backend, "model_for") else "?",
                duration_ms=dt)
        if disagree:
            flags.append("llm_crosscheck_disagree")
    except Exception:
        # cross-check is best-effort; a failure must not affect the parse
        if tracer:
            tracer.step("llm_crosscheck", {"error": "llm_crosscheck_failed"},
                        model_name="?", duration_ms=None)


def build_document_fields(doc, layout, meta, pairs, flags, qdir=None,
                          organized_root=None, answer_file_path=None):
    """
    Produce the diary-as-container document fields (Change 3/4/5) from the parsed
    doc + extracted pairs. Returns a dict of new schema fields merged into
    parsed.json. Backward-compatible: `pairs` is preserved by the caller.
    """
    from .ir import SubQuestion, AnswerGroup, AnswerBlock
    text = doc.full_text()
    out_flags = list(flags)

    # --- identity: folder id vs diary numbers (kept SEPARATE) ----------------
    dnums = diary_numbers(text)
    if not dnums:
        out_flags.append("no_diary_number_found")
        fallback = str(meta.get("question_id", "") or "unknown")
        dnums = [fallback.upper()] if fallback != "unknown" else []
    if len(dnums) > 1:
        out_flags.append("multi_diary_document")

    # tables the extractor produced (dedup by table_id)
    all_tables, seen_tid = [], set()
    for p in pairs:
        for t in p.tables:
            if t.table_id not in seen_tid:
                seen_tid.add(t.table_id)
                all_tables.append(t)

    # part_records: ordered {part_label, question_text, answer_text}.
    # SOURCE PRIORITY:
    #  1. If the LLM produced the grouping ('model_extracted'), TRUST its per-pair
    #     answers — the LLM handles ambiguous mappings the splitter can't (e.g.
    #     12183: a,b,c share boilerplate but d gets its own substantive answer).
    #  2. Else deterministic sub-part split.
    #  3. Else fall back to LLM pairs / single block.
    llm_made_pairs = "model_extracted" in flags and len(pairs) >= 1
    subparts = None if llm_made_pairs else _split_subparts(text, meta)
    reply_format = classify_reply_format(text, subparts)
    if reply_format == "covering_letter":
        out_flags.append("covering_letter_format")

    part_records = []
    if llm_made_pairs:
        for i, p in enumerate(pairs):
            lm = re.match(r"\s*\(?([a-h])\)", p.question_text or "", re.I)
            label = lm.group(1).lower() if lm else chr(ord("a") + i)
            part_records.append({
                "part_label": label, "question_text": p.question_text,
                "answer_text": p.answer_text})
    elif subparts:
        for sp in subparts:
            part_records.append({
                "part_label": sp["label"].strip("()"),
                "question_text": sp["question_text"],
                "answer_text": sp["answer_text"]})
    elif len(pairs) > 1:
        for i, p in enumerate(pairs):
            lm = re.match(r"\s*\(?([a-h])\)", p.question_text or "", re.I)
            label = lm.group(1).lower() if lm else chr(ord("a") + i)
            part_records.append({
                "part_label": label, "question_text": p.question_text,
                "answer_text": p.answer_text})
        if reply_format == "unknown":
            reply_format = "questions_then_answers"
    else:
        part_records.append({
            "part_label": "(single)",
            "question_text": _question_body(text) or "(question text not separated)",
            "answer_text": _answer_body(text)})

    # --- group parts by SHARED answer (answer stored once) -------------------
    groups_raw, part_to_group = _group_answers(part_records)

    # safeguard: questions-then-answers with parts that never got an answer AND no
    # shared group covers them -> counts don't line up; flag rather than force-fit.
    n_empty = sum(1 for r in part_records if not r["answer_text"].strip())
    has_shared = any(len(g["answers_parts"]) > 1 for g in groups_raw)
    if len(part_records) > 1 and n_empty and not has_shared:
        out_flags.append("qa_count_mismatch")

    # refine reply_format from the actual grouping
    reply_format = _refine_reply_format(reply_format, part_records, groups_raw)

    # --- GLOBALLY-UNIQUE IDs (Phase-3 prep) ----------------------------------
    # Prefix every id with the folder question_id so ids are unique across the whole
    # corpus and DETERMINISTIC (same input -> same id on re-run). part_label is NOT
    # unique on its own; the local group ids ("g1") are only locally unique.
    qid = str(meta.get("question_id", "") or "q").strip()
    # local group id ("g1") -> global ("<qid>_g1")
    gid_map = {g["answer_group_id"]: f"{qid}_{g['answer_group_id']}" for g in groups_raw}

    # --- build sub_questions (lean, pointer to group) ------------------------
    sub_questions, annexures_all = [], []
    qtext_by_part = {r["part_label"]: r["question_text"] for r in part_records}
    for rec in part_records:
        part = rec["part_label"]
        refs = annexure_refs(rec["question_text"]) + annexure_refs(rec["answer_text"])
        refs = _dedup(refs)
        annexures_all += [r for r in refs if r not in annexures_all]
        sub_questions.append(SubQuestion(
            sub_question_id=f"{qid}_{part}",              # globally unique, e.g. 4570_a
            part_label=part,
            question_text=rec["question_text"],
            answer_group_id=gid_map[part_to_group[part]],  # globally-unique pointer
            question_language=detect_language(rec["question_text"]),
            annexure_refs=refs,
        ))

    # --- build answer_groups (answer once; annexure_refs = UNION over parts) -
    answer_groups = []
    part_answer_text = {r["part_label"]: r["answer_text"] for r in part_records}
    for g in groups_raw:
        atext = g["answer_text"]
        atype = classify_answer_type(atext)
        # union of annexure refs across every part this group covers (+ its answer)
        grp_refs = annexure_refs(atext)
        for part in g["answers_parts"]:
            grp_refs += annexure_refs(qtext_by_part.get(part, ""))
        grp_refs = _dedup(grp_refs)
        annexures_all += [r for r in grp_refs if r not in annexures_all]
        answer_groups.append(AnswerGroup(
            answer_group_id=gid_map[g["answer_group_id"]],   # globally unique
            answers_parts=g["answers_parts"],
            answer_text=atext,
            answer_type=atype,
            answer_language=detect_language(atext),
            answer_blocks=[AnswerBlock(**b) for b in split_answer_blocks(atext)],
            tables=[],
            annexure_refs=grp_refs,
            confidence="high" if atext.strip() else "low",
        ))

    # --- attach tables INSIDE their answer group -----------------------------
    diary_level_tables = _attach_tables_to_groups(
        all_tables, answer_groups, part_records, part_to_group, out_flags)

    # Change 3: re-key table_id/row_id so they are group-prefixed and globally
    # unique: "<qid>_<group>_t<n>" and "..._r<m>". A table lives in exactly one
    # group, so numbering per group is stable and deterministic.
    for g in answer_groups:
        for ti, t in enumerate(g.tables, start=1):
            _rekey_table(t, f"{g.answer_group_id}_t{ti}")
    for ti, t in enumerate(diary_level_tables, start=1):
        _rekey_table(t, f"{qid}_dl_t{ti}")

    starred = _detect_starred(text)
    subject = _detect_subject(text)

    # --- annexure FILE resolution (Change 2): map each referenced label to a file
    # in answer_all_versions/, path-only (no parsing). Build the per-part citation
    # map from the sub_questions so the UI knows which parts cite which annexure.
    refs_by_part = {}
    for sq in sub_questions:
        for lab in sq.annexure_refs:
            refs_by_part.setdefault(lab, [])
            if sq.part_label not in refs_by_part[lab]:
                refs_by_part[lab].append(sq.part_label)
    ordered_labels = [l for l in annexures_all]  # preserve first-seen order

    annexures = []
    if ordered_labels and qdir and organized_root:
        annexures, ann_flags = resolve_annexures(
            ordered_labels, refs_by_part, qdir, organized_root)
        out_flags += ann_flags
    elif ordered_labels:
        # resolver context missing -> record refs without resolution
        for lab in ordered_labels:
            annexures.append({
                "ref_label": lab, "referenced_in_parts": refs_by_part.get(lab, []),
                "file_path": None, "file_present": False, "match_confidence": "none"})

    # --- Change 5: validate the retrieval -> display link chain resolves from
    # each sub_question_id (pointer to a real group; the display payload — answer
    # file + annexures — is doc-level and always reachable). Flags group_link_broken.
    _validate_group_links(sub_questions, answer_groups, out_flags)
    _validate_display_chain(sub_questions, annexures, out_flags)

    fields = {
        # Change 4: the RETRIEVAL/embedding unit for Phase 3 is the sub-question's
        # question text. Answers/tables/annexures are DISPLAY PAYLOAD, not search
        # targets. Phase 3 embeds sub_questions[].question_text keyed by
        # sub_question_id and, on a hit, fetches this document's answer + file paths.
        "embedding_unit": "sub_question.question_text",
        "answer_file_path": answer_file_path,
        "diary_numbers": dnums,
        "starred": starred,
        "subject": subject,
        "reply_format": reply_format,
        "is_nhpc_relevant": _is_nhpc_relevant(text, answer_groups),
        "sub_questions": [sq.to_dict() for sq in sub_questions],
        "answer_groups": [g.to_dict() for g in answer_groups],
        "diary_level_tables": [t.to_dict() for t in diary_level_tables],
        "annexures_referenced": annexures_all,
        "annexures": annexures,
        "annexure_content_present": any(a["file_present"] for a in annexures) if annexures else False,
    }
    return fields, _dedup(out_flags)


def _validate_display_chain(sub_questions, annexures, out_flags):
    """
    Change 5: confirm each sub_question can reach its display payload. The answer
    link (sub_question.answer_group_id -> answer_group) is checked by
    _validate_group_links. Here we check that any annexure a sub-part cites resolves
    to an entry in the document annexures[] (so the annexure button has a target).
    """
    annex_labels = {a["ref_label"] for a in annexures}
    for sq in sub_questions:
        for ref in sq.annexure_refs:
            if ref not in annex_labels:
                if "annexure_ref_unresolved" not in out_flags:
                    out_flags.append("annexure_ref_unresolved")
                return


def _refine_reply_format(reply_format, part_records, groups_raw):
    """Refine reply_format from the actual grouping produced."""
    n_parts = len(part_records)
    n_groups = len(groups_raw)
    if reply_format == "covering_letter":
        return reply_format
    if n_parts <= 1:
        return reply_format if reply_format != "unknown" else "single"
    if n_groups == 1:
        return "questions_then_shared_answer"      # all parts, one answer
    if n_groups < n_parts:
        return "mixed_grouping"                     # some parts share, some don't
    # one group per part
    return reply_format if reply_format in ("interleaved", "questions_then_answers") \
        else "interleaved"


def _attach_tables_to_groups(all_tables, answer_groups, part_records,
                             part_to_group, out_flags):
    """
    Put each table INSIDE the answer group it belongs to. Tables are assigned
    INDIVIDUALLY (a doc with 2 tables for parts c and d must put one in each group,
    not both in one). Strategy, in order:
      - single answer group  -> all tables go to it.
      - groups whose answer 'points to a table' ("given below/as under"): if the
        count of such groups equals the number of tables, map them 1:1 in document
        order (tables sorted by page). This handles c->table1, d->table2.
      - one pointer group -> it takes all tables.
      - counts don't line up / no pointer -> best-guess, mark low-confidence +
        table_group_uncertain; NEVER silently orphan.
    Returns diary_level_tables (rare).
    """
    if not all_tables:
        return []
    tables = sorted(all_tables, key=lambda t: (getattr(t, "page", 0),
                                               getattr(t, "table_id", "")))
    if len(answer_groups) == 1:
        answer_groups[0].tables = tables
        _mark_answer_is_table(answer_groups[0])
        return []

    # groups whose answer points to an inline table, IN DOCUMENT ORDER (by the first
    # part each group covers, a<b<c<d).
    def group_order(g):
        return min((p for p in g.answers_parts), default="z")
    pointer_groups = sorted(
        [g for g in answer_groups if _points_to_table(g.answer_text)],
        key=group_order)

    if len(pointer_groups) == len(tables) and pointer_groups:
        # 1:1 positional map — each pointer group gets its own table in order
        for g, t in zip(pointer_groups, tables):
            g.tables = [t]
            _mark_answer_is_table(g)
        return []

    if len(pointer_groups) == 1:
        pointer_groups[0].tables = tables
        _mark_answer_is_table(pointer_groups[0])
        return []

    # ambiguous: distribute best-effort but flag + lower confidence.
    if pointer_groups:
        # spread tables across pointer groups round-robin so none is silently dropped
        for i, t in enumerate(tables):
            g = pointer_groups[i % len(pointer_groups)]
            g.tables.append(t)
            t.extraction_confidence = "low"
        for g in pointer_groups:
            _mark_answer_is_table(g)
    else:
        answer_groups[-1].tables = tables
        for t in tables:
            t.extraction_confidence = "low"
    if "table_group_uncertain" not in out_flags:
        out_flags.append("table_group_uncertain")
    return []


def _rekey_table(table, new_table_id):
    """
    Set a globally-unique table_id and re-derive each row_id from it. Deterministic:
    same table -> same id. Only touches the ID strings, not the table data.
    """
    table.table_id = new_table_id
    for ri, row in enumerate(table.rows, start=1):
        row.row_id = f"{new_table_id}_r{ri}"


def _mark_answer_is_table(group):
    """Set answer_is_table + role when the table IS the whole answer (little prose)."""
    prose = re.sub(r"\s+", "", group.answer_text or "")
    if group.tables and len(prose) < 60:
        group.answer_is_table = True
        for t in group.tables:
            t.table_role = "answer_data"
    else:
        for t in group.tables:
            if t.table_role not in ("answer_data",):
                t.table_role = "supporting"


def _validate_group_links(sub_questions, answer_groups, out_flags):
    """Every sub_question must point to a real group; groups must ref real parts."""
    gids = {g.answer_group_id for g in answer_groups}
    parts = {sq.part_label for sq in sub_questions}
    ok = True
    for sq in sub_questions:
        if sq.answer_group_id not in gids:
            ok = False
    for g in answer_groups:
        for p in g.answers_parts:
            if p not in parts:
                ok = False
    if not ok and "group_link_broken" not in out_flags:
        out_flags.append("group_link_broken")


def _group_answers(part_records):
    """
    Group sub-parts by SHARED answer into answer_groups. Input: ordered list of
    dicts {part_label, question_text, answer_text}. Parts whose answer_text is the
    same (or one is empty and shares the neighbouring answer) collapse into one
    group. Returns (groups, part_to_group) where:
        groups = [{answer_group_id, answers_parts, answer_text}]
        part_to_group = {part_label: answer_group_id}
    This is the deterministic grouping used for all layout cases; the LLM path
    produces the same structure for the ambiguous ones.
    """
    def norm(s):
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    groups = []
    part_to_group = {}
    for rec in part_records:
        atext = rec["answer_text"]
        na = norm(atext)
        # find an existing group with the SAME answer text (shared answer)
        gid = None
        if na:
            for g in groups:
                if norm(g["answer_text"]) == na:
                    gid = g["answer_group_id"]
                    g["answers_parts"].append(rec["part_label"])
                    break
        if gid is None:
            gid = f"g{len(groups) + 1}"
            groups.append({
                "answer_group_id": gid,
                "answers_parts": [rec["part_label"]],
                "answer_text": atext,
            })
        part_to_group[rec["part_label"]] = gid
    return groups, part_to_group


def _annexure_content_present(refs, doc):
    """True if the referenced annexure content actually appears in the parsed doc."""
    if not refs:
        return True
    # heuristic: an annexure is 'present' if the doc has a table, or the annexure
    # label appears again as a heading followed by substantial content.
    if doc.tables:
        return True
    text = doc.full_text()
    for r in refs:
        stem = r.replace("Annexure-", "").strip()
        # a heading line 'Annexure-I' followed by >200 chars of content
        m = re.search(r"annexure?\s*[-\s]?" + re.escape(stem), text, re.I)
        if m and len(text[m.end():]) > 200:
            return True
    return False


def _detect_starred(text):
    if re.search(r"\bunstarred\b", text, re.I):
        return False
    if re.search(r"\bstarred\b", text, re.I):
        return True
    return None


def _detect_subject(text):
    m = re.search(r"(?:regarding|subject\s*[:\-]|on\s+the\s+subject)\s*"
                  r"['\"‘“]?([^'\"’”\n]{4,120})", text, re.I)
    return m.group(1).strip() if m else None


def _is_nhpc_relevant(text, answer_groups):
    if answer_groups and all(g.answer_type == "not_applicable" for g in answer_groups):
        return False
    return bool(re.search(r"\bNHPC\b", text))


# ---------------------------------------------------------------------------
# PATH 2 — QA-TABLE (Table-Type-3)
# ---------------------------------------------------------------------------

def _extract_qa_table(doc, layout, qid, meta):
    flags = []
    raw = layout["qa_table_ref"]
    cleaned = clean_table(raw, f"{qid}_t1")
    if cleaned is None:
        flags.append("qa_table_empty")
        return [], [], flags

    columns = cleaned["columns"]
    header = cleaned["header"]
    body = cleaned["body"]

    # resolve column roles -> indices
    def col_idx(role):
        for i, c in enumerate(columns):
            if c.role == role:
                return i
        return None

    qno_i = col_idx("qno")
    q_i = col_idx("question")
    a_i = col_idx("answer")

    if q_i is None or a_i is None:
        # content-based inference: shortest col = qno, next = question, rest = answer
        flags.append("qa_table_columns_inferred")
        lens = []
        for i in range(len(header)):
            vals = [len(r[i]) for r in body if i < len(r)]
            lens.append(sum(vals) / len(vals) if vals else 0)
        order = sorted(range(len(header)), key=lambda i: lens[i])
        if qno_i is None and lens and lens[order[0]] < 8:
            qno_i = order[0]
        remaining = [i for i in range(len(header)) if i != qno_i]
        if remaining:
            q_i = remaining[0]
            a_i = remaining[1] if len(remaining) > 1 else remaining[0]

    # All answer-role columns (a Q&A table's answer may span several columns,
    # e.g. one per year 2022/2023/2024 under a spanning "reply" header).
    answer_cols = [i for i, c in enumerate(columns) if c.role == "answer"]
    if not answer_cols and a_i is not None:
        answer_cols = [a_i]

    # emit the underlying table object for traceability
    tout, tflags = build_table_out(raw, f"{qid}_t1", "qa_pairs", answer_is_table=False)
    flags += tflags
    tables_index = [tout.table_id] if tout else []

    base_conf = "low" if ("qa_table_columns_inferred" in flags or tout is None
                          or (tout and tout.extraction_confidence == "low")) else "high"

    def cell(row, i):
        return row[i].strip() if (i is not None and i < len(row) and row[i]) else ""

    def answer_of(row):
        parts = []
        for i in answer_cols:
            v = cell(row, i)
            if v:
                # label multi-column answers with their header (year, etc.)
                hdr = header[i] if i < len(header) else ""
                parts.append(f"{hdr}: {v}" if hdr and not hdr.lower().startswith("format") else v)
        return " | ".join(parts)

    pairs = []
    prev = None
    prev_qnum = None
    for ri, row in enumerate(body, start=1):
        raw_qnum = cell(row, qno_i)
        qnum = re.sub(r"\s+", "", raw_qnum)
        qtext = cell(row, q_i)
        atext = answer_of(row)

        # Continuation of the previous question when the qno is blank OR repeats
        # the previous row's qno (nested answer sub-rows: year lists, breakups).
        is_continuation = prev is not None and (
            not raw_qnum or (qnum and qnum == prev_qnum))
        if is_continuation:
            if qtext and qtext.lower() not in prev.question_text.lower():
                prev.question_text = (prev.question_text + " " + qtext).strip()
            if atext:
                prev.answer_text = (prev.answer_text + "\n" + atext).strip()
            prev.confidence = "low"  # stitched answers merit a look
            if "qa_table_row_stitched" not in flags:
                flags.append("qa_table_row_stitched")
            continue

        p = Pair(
            question_number=qnum or str(ri),
            question_text=qtext,
            answer_text=atext,
            question_language=detect_language(qtext),
            answer_language=detect_language(atext),
            answer_is_table=False,
            related_question_numbers=[qnum or str(ri)],
            tables=[tout] if tout else [],
            confidence=base_conf,
        )
        pairs.append(p)
        prev = p
        prev_qnum = qnum

    return pairs, tables_index, flags


# ---------------------------------------------------------------------------
# PATH 1 — PROSE
# ---------------------------------------------------------------------------

def _extract_prose(doc, layout, qid, meta, cfg, backend, tracer=None):
    flags = []
    text = doc.full_text()

    # Build the table objects first (Type-1 supporting / Type-2 answer-is-table)
    table_objs = []
    tables_index = []
    for ti, raw in enumerate(doc.tables, start=1):
        table_id = f"{qid}_t{ti}"
        # heuristic role: a table that is large & entity-per-row and the prose is
        # short => Type-2 (answer_is_table); otherwise Type-1 supporting.
        is_answer_table = _is_answer_table(raw, text)
        role = "answer_data" if is_answer_table else "supporting"
        tout, tflags = build_table_out(raw, table_id, role, is_answer_table)
        if tout:
            table_objs.append((tout, is_answer_table))
            tables_index.append(tout.table_id)
            flags += tflags

    # LLM-PRIMARY grouping (cfg.llm_grouping with a capable model, e.g. groq 70B):
    # TRUST THE LLM. The prompt now gives it the question_block + n_answer_markers so
    # it counts questions/answers correctly and never mines a question from '(A)/(i)'
    # headings inside an answer. We do NOT police it with the deterministic count
    # (that caused a false conflict when deterministic was itself wrong). The only
    # guard is grounding: _model_prose rejects output whose answers are not present
    # in the source (invented text) and falls back to deterministic in that case.
    subparts = _split_subparts(text, meta)
    if getattr(cfg, "llm_grouping", False) and _is_real_model(backend):
        result = _model_prose(doc, layout, meta, cfg, backend, table_objs, tracer)
        if result is not None:
            r_pairs, r_index, r_flags = result
            return r_pairs, r_index, flags + r_flags + ["llm_grouping"]
        # only if the model failed / produced ungrounded output -> deterministic

    # DETERMINISTIC (default path / fallback when LLM unavailable or ungrounded).
    if subparts:
        pairs = _deterministic_prose(text, layout, meta, table_objs, flags)
        if tracer:
            tracer.step("extraction", {
                "path": "subpart_deterministic", "n_pairs": len(pairs),
                "raw_text_in": text[:4000], "raw_model_output": None,
                "note": "deterministic (a)/(b)/(c) sub-part split with answers"},
                model_name="deterministic", duration_ms=None)
        return pairs, tables_index, flags

    # No sub-part structure and no LLM grouping: let a real model pair if configured.
    if _is_real_model(backend):
        result = _model_prose(doc, layout, meta, cfg, backend, table_objs, tracer)
        if result is not None:
            r_pairs, r_index, r_flags = result
            return r_pairs, r_index, flags + r_flags

    # Fallback: generic deterministic pairing.
    pairs = _deterministic_prose(text, layout, meta, table_objs, flags)
    if tracer:
        tracer.step("extraction", {
            "path": "prose", "structure": layout.get("structure"),
            "layout_case": layout.get("case"), "n_pairs": len(pairs),
            "raw_text_in": text[:4000], "raw_model_output": None,
            "note": "deterministic prose pairing (no model call)"},
            model_name="deterministic", duration_ms=None)
    return pairs, tables_index, flags



def _is_answer_table(raw, prose_text):
    cleaned = clean_table(raw, "probe")
    if not cleaned:
        return False
    body = cleaned["body"]
    roles = [c.role for c in cleaned["columns"]]
    entity_like = any(r in roles for r in ("project_name", "location", "status"))
    # answer-is-table when there are several entity rows and little surrounding prose
    prose_wo_table = len(re.sub(r"\s+", "", prose_text))
    table_chars = sum(len(x) for row in body for x in row)
    return entity_like and len(body) >= 2 and table_chars >= 0.3 * max(1, prose_wo_table)


# Sub-part markers: "(a)", "a)", "a." at a boundary. Parliamentary Q&A replies list
# sub-questions this way, each followed by its answer under Comment:/Answer:/Reply:.
_SUBPART = re.compile(r"(?:(?<=\n)|(?<=\s)|^)\(?([a-h])\)\s+", re.I)
_ANSWER_MARKER = re.compile(r"\b(comments?|answer|reply|उत्तर)\b\s*[:.\-]", re.I)


def _diary_number(text, meta):
    """The parliamentary diary/question number for this reply (content, else meta)."""
    nums = diary_numbers(text)
    if nums:
        return nums[0]
    return str(meta.get("question_id", "") or "unknown")


def diary_numbers(text):
    """
    Extract ALL parliamentary diary numbers from the reply header (an ARRAY — one
    file may cover several, e.g. 'S2542 & S2544'). Returns [] if none found.
    """
    m = re.search(
        r"(?:dy\.?\s*no\.?s?|diary\s*no\.?s?|question\s*(?:dy\.?\s*)?no\.?s?)\s*[:.]?\s*"
        r"([A-Z]?[-\s]?\d{2,6}[A-Z]?(?:\s*(?:,|&|and|/)\s*[A-Z]?[-\s]?\d{2,6}[A-Z]?)*)",
        text, re.I)
    if not m:
        return []
    span = m.group(1)
    nums = re.findall(r"[A-Z]?[-\s]?\d{2,6}[A-Z]?", span)
    out, seen = [], set()
    for n in nums:
        n = re.sub(r"\s+", "", n).upper()
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


# --- answer-type classification (per sub-part) ------------------------------

_DEFER_RE = re.compile(
    r"may\s+be\s+replied\s+by\s+(mop|cea|mowr|ministry|the\s+ministry)|"
    r"pertains?\s+to\s+(mop|cea|ministry)|to\s+be\s+replied\s+by\s+m", re.I)
_NA_RE = re.compile(
    r"does\s+not\s+pertain\s+to\s+nhpc|not\s+(?:applicable|related)\s+to\s+nhpc|"
    r"no\s+such\s+.*nhpc|as\s+far\s+as\s+nhpc.*not\s+applicable", re.I)
_NIL_RE = re.compile(
    r"\b(treated\s+as\s+nil|may\s+be\s+treated\s+as\s+nil|information.*nil|"
    r"\bnil\b\s*$|no\s+such\s+instances?)\b", re.I)


def classify_answer_type(answer_text):
    """substantive | deferred_to_ministry | nil | not_applicable for one answer."""
    t = answer_text or ""
    if not t.strip():
        return "substantive"  # empty handled elsewhere (clubbed)
    if _NA_RE.search(t):
        return "not_applicable"
    if _NIL_RE.search(t):
        return "nil"
    # deferred only if that's essentially the WHOLE answer (short) — a long answer
    # that merely mentions MoP is still substantive.
    if _DEFER_RE.search(t) and len(re.sub(r"\s+", "", t)) < 120:
        return "deferred_to_ministry"
    return "substantive"


# --- annexure detection (rule, not just the LLM) ----------------------------

_ANNEX_REF_RE = re.compile(
    r"(annexure?\s*[-\s]?[IVXLC0-9A-Z]+|annex\.?\s*[-\s]?[IVXLC0-9A-Z]+)", re.I)
_ANNEX_TRIGGER_RE = re.compile(
    r"\b(annex|annexure|enclosed|given\s+below\s+at|as\s+at\s+annex|placed\s+at|"
    r"details?\s+(?:are|is)\s+(?:given|placed|at)|reflected\s+in)\b", re.I)


_ANNEX_LABEL_RE = re.compile(r"annex(?:ure)?\.?\s*[-\s]?\s*([IVXLC]+|\d+|[A-H])\b", re.I)


def annexure_refs(text):
    """Return the list of distinct Annexure labels referenced (e.g. Annexure-I)."""
    refs, seen = [], set()
    for m in _ANNEX_LABEL_RE.finditer(text or ""):
        label = "Annexure-" + m.group(1).upper()
        if label not in seen:
            seen.add(label)
            refs.append(label)
    return refs


def references_annexure(text):
    """True if the answer defers substantive content to an annexure/enclosure."""
    return bool(_ANNEX_TRIGGER_RE.search(text or "")) and bool(annexure_refs(text))


_POINTS_TO_TABLE = re.compile(
    r"\b(as\s+(?:given|shown|under)\s+below|given\s+below|as\s+under|"
    r"following\s+(?:table|details)|details\s+(?:is|are)\s+as\s+(?:given|under|below)|"
    r"as\s+follows|tabulated\s+below)\b", re.I)


def _points_to_table(answer_text):
    """True when an answer explicitly points to an inline table (not an annexure)."""
    t = answer_text or ""
    if references_annexure(t):   # 'given at Annexure-I' points to a FILE, not inline
        return False
    return bool(_POINTS_TO_TABLE.search(t))


# --- annexure file resolution (path capture only; no parsing) ----------------

_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7,
          "viii": 8, "ix": 9, "x": 10}
_ROMAN_REV = {v: k for k, v in _ROMAN.items()}
# files that are never an annexure (reply/note/cover/question/person-doc)
_NONANNEX_FILE = re.compile(
    r"^(reply|note|covering|cover|question|~\$)|kundan|bajpai|palit|shaw|gautam|anil",
    re.I)


def _label_number(label):
    """Return a canonical number for an annexure label ('Annexure-II' -> 2), or None."""
    m = re.search(r"([IVXLC]+|\d+|[A-H])$", (label or "").strip(), re.I)
    if not m:
        return None
    tok = m.group(1).lower()
    if tok.isdigit():
        return int(tok)
    if tok in _ROMAN:
        return _ROMAN[tok]
    if len(tok) == 1 and tok.isalpha():
        return ord(tok) - ord("a") + 1  # A->1, B->2 ...
    return None


def _file_annex_tokens(fname):
    """
    All annexure numbers a filename encodes. Handles 'Annexure-I', 'Annex-II',
    'annexure 3', ranges 'Annexure-II & III'. Returns a set of ints ({} if none).
    Unnumbered 'Annexure.pdf' -> {0} (a numberless annexure marker).
    """
    low = fname.lower()
    if not re.search(r"annex", low):
        return set()
    nums = set()
    for m in re.finditer(r"annex(?:ure)?\.?\s*[-\s]?\s*([ivxlc]+|\d+|[a-h])\b", low):
        n = _label_number(m.group(1))
        if n:
            nums.add(n)
    # bare 'Annexure.pdf' with no number
    if not nums and re.search(r"annex(?:ure)?\b", low):
        nums.add(0)
    return nums


def resolve_annexures(annex_labels, refs_by_part, qdir, organized_root):
    """
    Resolve each referenced annexure LABEL to a file in <qdir>/answer_all_versions/.
    Returns (annexures:list[dict], flags:list[str]). Paths are RELATIVE to
    organized_root. Never parses files, never fabricates a path (flag-over-guess).

    refs_by_part: {annexure_label: [part_labels that cite it]}
    """
    import os
    flags = []
    av_dir = os.path.join(qdir, "answer_all_versions")
    # candidate annexure files (exclude replies/notes/cover/person docs)
    candidates = []
    if os.path.isdir(av_dir):
        for root, _d, files in os.walk(av_dir):
            for fn in files:
                if _NONANNEX_FILE.search(fn):
                    continue
                if os.path.splitext(fn)[1].lower() not in (
                        ".pdf", ".docx", ".doc", ".xlsx", ".xls"):
                    continue
                candidates.append(os.path.join(root, fn))

    def rel(p):
        return os.path.relpath(p, organized_root).replace(os.sep, "/")

    def prefer_pdf(paths):
        pdfs = [p for p in paths if p.lower().endswith(".pdf")]
        return (pdfs or paths)[0]

    # annexure-named candidate files, indexed by the number(s) they encode
    annex_files = [(p, _file_annex_tokens(os.path.basename(p))) for p in candidates]
    annex_files = [(p, toks) for p, toks in annex_files if toks]
    non_reply_files = candidates  # for elimination rule

    out = []
    n_refs = len(annex_labels)
    for label in annex_labels:
        num = _label_number(label)
        parts = refs_by_part.get(label, [])
        # rule 1: a file whose name encodes this annexure number
        by_num = [p for p, toks in annex_files if num in toks] if num else []
        # unnumbered file (token 0) can satisfy a single unnumbered/any reference
        by_zero = [p for p, toks in annex_files if toks == {0}]

        if by_num:
            path = prefer_pdf(by_num)
            conf, present = "high", True
            if len(set(os.path.splitext(os.path.basename(p))[0].lower() for p in by_num)) > 1:
                pass  # multiple distinct names encode same num -> still high (pick pdf)
        elif n_refs == 1 and len(annex_files) == 1:
            # rule 2: single annexure referenced, single annexure file -> elimination
            path = prefer_pdf([annex_files[0][0]])
            conf, present = "high", True
        elif by_zero and n_refs == 1:
            path = prefer_pdf(by_zero)
            conf, present = "high", True
        elif annex_files:
            # rule 3: something annexure-ish exists but can't be pinned -> ambiguous
            path = prefer_pdf([p for p, _ in annex_files])
            conf, present = "low", True
            if "annexure_match_ambiguous" not in flags:
                flags.append("annexure_match_ambiguous")
        else:
            # rule 4: referenced but no candidate file found
            path, conf, present = None, "none", False
            if "answer_in_annexure_not_present" not in flags:
                flags.append("answer_in_annexure_not_present")

        out.append({
            "ref_label": label,
            "referenced_in_parts": parts,
            "file_path": rel(path) if path else None,
            "file_present": present,
            "match_confidence": conf,
        })
    return out, flags


# --- reply-format classification --------------------------------------------

def classify_reply_format(text, subparts):
    """interleaved | questions_then_answers | covering_letter | unknown."""
    t = text or ""
    # covering letter: formal letter markers, little/no a/b/c structure
    letter = re.search(
        r"\b(महोदय|sir\s*,|dear\s+sir|yours\s+faithfully|covering\s+letter|"
        r"kind\s+attention|with\s+reference\s+to\s+(?:your|the)\s+letter)\b", t, re.I)
    if letter and (not subparts or len(subparts) < 2):
        return "covering_letter"
    if not subparts or len(subparts) < 2:
        return "unknown"
    # interleaved: each sub-question is immediately followed by its Comment/answer.
    # questions_then_answers: all questions listed, then all answers.
    with_ans = sum(1 for s in subparts if s.get("answer_text", "").strip())
    if with_ans >= max(2, len(subparts) - 1):
        return "interleaved"
    return "questions_then_answers"


def split_answer_blocks(answer_text):
    """
    Split a long headed answer into blocks. A 'heading' is a short line ending with
    ':' or a bold-ish label; otherwise the whole answer is a single unnamed block.
    Returns list[dict{heading,text}] or [] for short/simple answers.
    """
    if not answer_text or len(answer_text) < 300:
        return []
    blocks = []
    # split on lines that look like headings (<=60 chars, ends with ':')
    parts = re.split(r"\n(?=[A-Z(][^\n]{0,60}:\s*(?:\n|$))", answer_text)
    if len(parts) < 2:
        return []
    for part in parts:
        m = re.match(r"([^\n:]{0,60}):\s*(.*)", part, re.S)
        if m and m.group(2).strip():
            blocks.append({"heading": m.group(1).strip(), "text": m.group(2).strip()})
        elif part.strip():
            blocks.append({"heading": "", "text": part.strip()})
    return blocks if len(blocks) >= 2 else []


# A Comment:/Answer:/Reply: marker that INTRODUCES an answer block. Anchored so we
# find each one's position in the reply.
_COMMENT_MARKER = re.compile(r"\b(comments?|answer|reply|उत्तर)\b\s*[:.\-]", re.I)
# An explicit sub-part label at a boundary: (a) / a) / a.
_PART_LABEL = re.compile(r"(?:(?<=\n)|(?<=\s)|^)\(?([a-h])\)[\s.]", re.I)


def _split_subparts(text, meta):
    """
    Split a reply into sub-parts using the ANSWER-MARKER rule (the reliable signal):

      "Comment:" (or Answer:/Reply:) marks where an answer STARTS. All the question
      sentences accumulated SINCE the previous Comment: are answered by this one —
      so if a Comment: follows N questions, those N questions SHARE this answer.

    This one algorithm handles every layout uniformly (marked (a)/(b)/(c), unmarked
    bare sentences, interleaved, questions-then-shared-answer, mixed grouping),
    because it keys on Comment: positions, not on fragile letter markers.

    Returns a list of {label, question_text, answer_text} or None if there is no
    Comment: structure at all (single-answer / covering-letter docs handled upstream).
    """
    # strip the standard preamble ("...reply of Lok Sabha ... is as below:")
    m_head = re.search(r"is\s+as\s+below\s*:?\s*", text, re.I)
    body = text[m_head.end():] if m_head else text

    comments = [(m.start(), m.end()) for m in _COMMENT_MARKER.finditer(body)]
    if not comments:
        return None  # no answer markers -> not a sub-part reply

    # Determine the QUESTIONS and their positions. Two shapes:
    #  (A) explicit (a)/(b)/(c) labels -> each label starts a question.
    #  (B) no labels -> questions are the sentences BEFORE the first Comment:.
    # Only accept labels that are genuine QUESTION markers: a lowercase ascending
    # run starting at 'a' or 'b', located in the question region (before the last
    # comment). This rejects '(A)/(B)' that appear INSIDE answer prose (e.g. 3213).
    # A label "(x)" is a QUESTION label (vs content like "(A) Natural Hurdles" or
    # "(i)/(ii)" bunched inside an answer) if it forms an ascending a,b,c... sequence
    # AND its region is a "question position": either before the first Comment:, or a
    # Comment: appears between it and the next label (i.e. it's answered). Case is
    # IGNORED for the letter (4491 has (a)(b)(C)(d)(e) — the 'C' is a real question).
    raw_labels = [(m.start(), m.group(1).lower()) for m in _PART_LABEL.finditer(body)]
    first_com = comments[0][0]

    def _answered(pos, next_pos):
        # is there a Comment: between this label and the next label?
        return any(pos < c[0] < next_pos for c in comments)

    seq_labels, expected = [], None
    for i, (pos, lab) in enumerate(raw_labels):
        next_pos = raw_labels[i + 1][0] if i + 1 < len(raw_labels) else len(body)
        in_question_position = pos < first_com or _answered(pos, next_pos)
        if not in_question_position:
            continue  # a content label inside an answer block -> skip
        if expected is None:
            if lab in ("a", "b"):     # sequence may start at a, or b (a unlabeled)
                expected = lab
            else:
                continue
        if lab == expected:
            seq_labels.append((pos, lab))
            expected = chr(ord(expected) + 1)
    labels = seq_labels

    q_events = []   # (position, question_text)
    if len(labels) >= 2:
        # capture any question text BEFORE the first label as part 'a' (mixed case,
        # e.g. 6330: unlabeled Q1 then b) c)).
        pre = re.sub(r"\s+", " ", body[:labels[0][0]]).strip(" .;")
        pre = re.sub(r"^.*?is as below\s*:?\s*", "", pre, flags=re.I)
        if len(pre) >= 15:
            q_events.append((0, pre))
        for i, (pos, lab) in enumerate(labels):
            nxt_lab = labels[i + 1][0] if i + 1 < len(labels) else len(body)
            nxt_com = next((c[0] for c in comments if c[0] > pos), len(body))
            qend = min(nxt_lab, nxt_com)
            qtext = re.sub(r"^\(?[a-h]\)[\s.]*", "",
                           re.sub(r"\s+", " ", body[pos:qend]).strip())
            q_events.append((pos, qtext.strip(" .;")))
    else:
        # unlabeled: the questions are the sentences BEFORE the first Comment:.
        # Split on newline/semicolon; keep sentences in order with synthetic
        # ascending positions (they all sit before the first comment).
        head = body[:comments[0][0]]
        base = 0
        for s in re.split(r"\n|;", head):
            s = re.sub(r"\s+", " ", s).strip(" .;")
            base += 1
            if len(s) >= 15:   # a real question clause (drops 'and', blanks)
                q_events.append((base, s))
        # if nothing split out but there is a single long question, keep it whole
        if not q_events:
            whole = re.sub(r"\s+", " ", head).strip(" .;")
            if len(whole) >= 15:
                q_events.append((0, whole))

    if not q_events:
        return None

    q_events = [(qp, q) for qp, q in q_events if q]   # drop blanks
    questions = [q for _, q in q_events]
    n_q = len(questions)

    # Compute each Comment:'s answer text (this comment -> next comment, minus any
    # question label that sits between them for the interleaved case).
    answers = []
    for ci, (cstart, cend) in enumerate(comments):
        next_com = comments[ci + 1][0] if ci + 1 < len(comments) else len(body)
        next_q = next((qp for qp, _ in q_events if qp > cstart), len(body))
        answers.append(re.sub(r"\s+", " ", body[cend:min(next_com, next_q)]).strip())
    # merge comments that have no question before them into the previous answer
    # (a stray "Comment:" continuation), keeping answers aligned to question breaks.
    n_a = len(answers)

    # Decide the mapping between questions and answers:
    first_com = comments[0][0]
    all_before = all(qp <= first_com for qp, _ in q_events)  # questions-then-answers
    labels = [chr(ord("a") + i) for i in range(n_q)]
    parts = []

    if all_before and n_a >= 2:
        # QUESTIONS-THEN-ANSWERS: map positionally q[i] -> answer[i].
        if n_a == n_q:
            for i in range(n_q):
                parts.append({"label": f"({labels[i]})", "question_text": questions[i],
                              "answer_text": answers[i]})
        elif n_a < n_q:
            # fewer answers than questions -> the LAST answer is shared by the
            # remaining questions (leading ones map 1:1, tail shares last).
            for i in range(n_q):
                a = answers[i] if i < n_a else answers[-1]
                parts.append({"label": f"({labels[i]})", "question_text": questions[i],
                              "answer_text": a})
        else:  # more answers than questions -> extra answers append to the last q
            for i in range(n_q):
                parts.append({"label": f"({labels[i]})", "question_text": questions[i],
                              "answer_text": answers[i]})
            extra = " ".join(answers[n_q:]).strip()
            if extra and parts:
                parts[-1]["answer_text"] = (parts[-1]["answer_text"] + " " + extra).strip()
    else:
        # INTERLEAVED (or labeled): each question is followed by its own comment;
        # assign by walking questions and comments in document order.
        # A question takes the answer of the FIRST comment that follows its position.
        for i, (qp, q) in enumerate(q_events):
            q_end = q_events[i + 1][0] if i + 1 < len(q_events) else len(body)
            # find comments between this question and the next question
            own = [answers[ci] for ci, (cs, _ce) in enumerate(comments)
                   if qp <= cs < q_end]
            atext = " ".join(a for a in own if a).strip()
            parts.append({"label": f"({labels[i]})", "question_text": q,
                          "answer_text": atext})
        # back-fill: a question with no comment before the next question shares the
        # NEXT non-empty answer (combined-answer case, e.g. a/b share one Comment).
        for i in range(len(parts)):
            if not parts[i]["answer_text"].strip():
                nxt = next((parts[j]["answer_text"] for j in range(i + 1, len(parts))
                            if parts[j]["answer_text"].strip()), "")
                parts[i]["answer_text"] = nxt

    return parts or None


def _deterministic_prose(text, layout, meta, table_objs, flags):
    """Split by question-number anchors; attach answers and tables."""
    case = layout.get("case", "unknown")
    qid = meta.get("question_id", "")
    club = _clubbed_numbers(text)

    # Preferred path: explicit (a)/(b)/(c) sub-parts, each with its own answer.
    subparts = _split_subparts(text, meta)
    if subparts:
        diary = _diary_number(text, meta)
        answer_tables = [t for (t, isans) in table_objs if isans]
        all_tabs = [t for (t, _i) in table_objs]
        pairs = []
        for sp in subparts:
            atext = sp["answer_text"]
            pairs.append(Pair(
                question_number=diary,
                question_text=f"{sp['label']} {sp['question_text']}".strip(),
                answer_text=atext,
                question_language=detect_language(sp["question_text"]),
                answer_language=detect_language(atext),
                answer_is_table=False,
                related_question_numbers=[diary],
                tables=all_tabs,
                confidence="high" if atext else "low",
            ))
        flags.append("subpart_split")
        if any(not p.answer_text.strip() for p in pairs):
            flags.append("subpart_missing_answer")
        return pairs

    # collect all question numbers found, in order
    anchors = [(m.start(), re.sub(r"\s+", "", m.group(1))) for m in QNUM_RE.finditer(text)]
    seen = []
    for _, q in anchors:
        if q not in seen:
            seen.append(q)

    answer_tables = [t for (t, isans) in table_objs if isans]
    support_tables = [t for (t, isans) in table_objs if not isans]

    # Case C: one combined answer for clubbed numbers
    if case == "C" or (len(club) >= 2):
        nums = club or (seen[:1] or [qid])
        primary = nums[0]
        atext = _answer_body(text)
        p = Pair(
            question_number=primary,
            question_text=_question_body(text) or "(question text not separated)",
            answer_text="" if answer_tables else atext,
            question_language=detect_language(text),
            answer_language=detect_language(atext),
            answer_is_table=bool(answer_tables),
            related_question_numbers=nums,
            tables=[t for (t, _i) in table_objs],
            confidence="low",
        )
        flags.append("clubbed_answer_split")
        return [p]

    # Case A/B/unknown: one pair per distinct question number
    if not seen:
        seen = [qid] if qid else ["unknown"]
        flags.append("no_question_number_detected")

    pairs = []
    single_answer = _answer_body(text)
    for i, qn in enumerate(seen):
        # crude per-question answer slice: text between this anchor and next
        qtext, atext = _slice_qa(text, qn, seen, i)
        is_ans_tab = bool(answer_tables) and len(seen) == 1
        p = Pair(
            question_number=qn,
            question_text=qtext or "(question text not separated)",
            answer_text="" if is_ans_tab else (atext or single_answer),
            question_language=detect_language(qtext or text),
            answer_language=detect_language(atext or single_answer),
            answer_is_table=is_ans_tab,
            related_question_numbers=[qn],
            tables=([t for (t, _i) in table_objs] if len(seen) == 1 else support_tables),
            confidence="high" if (qtext and (atext or single_answer)) else "low",
        )
        pairs.append(p)
    return pairs


def _model_prose(doc, layout, meta, cfg, backend, table_objs, tracer=None):
    """Send IR to a real model for pairing; validate + one stricter retry.

    Every attempt records a trace step with the EXACT prompt, the RAW model output
    (not just the parsed result), the parsed result, the model name + backend, and
    timing — so a later 👎 can be traced to the failing step and local-vs-nvidia
    outputs are directly comparable.
    """
    import time
    meta_qid = str(meta.get("question_id", ""))
    system = (
        "You extract parliamentary Question-Answer pairs from Indian parliament "
        "replies. Return STRICT JSON only, no prose. Preserve original text "
        "exactly, including Hindi/Devanagari. Do not translate. Do not invent "
        "content.\n"
        "IMPORTANT about question_number: it is the PARLIAMENT DIARY / QUESTION "
        "NUMBER (e.g. '1102', 'S-77', 'U1059'), NOT a sub-part label. Sub-parts "
        "(a)/(b)/(c) of ONE parliamentary question all share the SAME "
        "question_number; put the '(a)'/'(b)' label at the start of question_text, "
        "never in question_number. If the reply covers several diary numbers "
        "(clubbed), emit them in related_question_numbers. If you cannot find a "
        "diary number in the text, use \"" + meta_qid + "\" as the question_number. "
        "Never put question TEXT inside question_number.\n"
        "HOW TO FIND THE QUESTIONS (critical):\n"
        "- The questions come FIRST, after 'is as below:'. They may or may not have "
        "  (a)/(b)/(c) labels. If there are NO letter labels, each QUESTION is one "
        "  sentence/clause in that opening block (they usually start with 'Whether', "
        "  'The details', 'The number', 'The steps', 'If so').\n"
        "- The answers come AFTER, each introduced by 'Comment:'/'Comments:'/'Answer:'"
        "/'Reply:'. The NUMBER OF ANSWERS equals the number of Comment:/Answer: "
        "markers.\n"
        "- CRITICAL: markers like '(A)', '(B)', '(i)', '(ii)', '(a) Natural Hurdles' "
        "  that appear INSIDE an answer block (after a Comment:) are CONTENT HEADINGS "
        "  of that answer, NOT new questions. NEVER create a question from text that "
        "  sits inside an answer. The question count comes ONLY from the opening "
        "  question block, never from headings inside answers.\n"
        "answer_text: copy each answer VERBATIM from its Comment:/Answer: block; do "
        "not summarize; never leave it empty when a Comment follows.\n"
        "GROUPING (use EVERY answer block; match by MEANING — the key task):\n"
        "- There are n_answer_markers answer blocks (each begins at Comment:/Answer:)."
        " You MUST use the content of EVERY block — never discard a substantive "
        "answer. Read all blocks in order; concatenate consecutive blocks that form "
        "one answer.\n"
        "- Assign each answer to the question whose SUBJECT it addresses. Example: an "
        "answer 'NHPC is implementing 2000MW Subansiri...' belongs to the question "
        "asking the status of Subansiri, not to unrelated questions.\n"
        "- Two questions SHARE an answer ONLY when the same answer text genuinely "
        "addresses both (then put identical answer_text on both). Do NOT collapse a "
        "question's OWN substantive answer into a generic 'May be replied by MoP' "
        "bucket — if a block has NHPC-specific content for a question, that question "
        "gets that content, not the generic line.\n"
        "- A bare 'May be replied by MoP/CEA.' with no NHPC specifics applies only to "
        "questions that have no dedicated substantive block. NEVER invent questions.\n"
        "Schema: {\"pairs\":[{\"question_number\":str,\"question_text\":"
        "str,\"answer_text\":str,\"question_language\":\"en|hi\",\"answer_language\""
        ":\"en|hi\",\"answer_is_table\":bool,\"related_question_numbers\":[str],"
        "\"confidence\":\"high|low\"}]}\n"
        # --- SHAPE-ONLY example using <PLACEHOLDER> tokens (NOT real text); shows
        # the JSON structure only. The model must copy answer_text verbatim from
        # reply_text and never reuse these placeholder words. ---
        "EXAMPLE (structure only — do not copy this text):\n"
        "INPUT: reply of dy. no. <NUM> ... (a) <question A text>; Comment: <answer A "
        "text copied from the reply>. (b) <question B text>; Comment: <answer B "
        "text copied from the reply>.\n"
        "OUTPUT: {\"pairs\":[{\"question_number\":\"<NUM>\",\"question_text\":\"(a) "
        "<question A text>\",\"answer_text\":\"<answer A text copied from reply>\","
        "\"question_language\":\"en\",\"answer_language\":\"en\",\"answer_is_table\""
        ":false,\"related_question_numbers\":[\"<NUM>\"],\"confidence\":\"high\"},"
        "{\"question_number\":\"<NUM>\",\"question_text\":\"(b) <question B text>\","
        "\"answer_text\":\"<answer B text copied from reply>\",\"question_language\""
        ":\"en\",\"answer_language\":\"en\",\"answer_is_table\":false,"
        "\"related_question_numbers\":[\"<NUM>\"],\"confidence\":\"high\"}]}"
    )
    reply_text = doc.full_text()
    # Concrete structural hints so the model doesn't have to guess the counts:
    #  - n_answer_markers: how many Comment:/Answer:/Reply: blocks exist = #answers.
    #  - question_block: the text BEFORE the first answer marker = where the
    #    questions live (so the model never mines questions from answer content).
    m_head = re.search(r"is\s+as\s+below\s*:?\s*", reply_text, re.I)
    after_head = reply_text[m_head.end():] if m_head else reply_text
    first_ans = _COMMENT_MARKER.search(after_head)
    question_block = (after_head[:first_ans.start()] if first_ans else after_head).strip()
    n_answer_markers = len(_COMMENT_MARKER.findall(reply_text))
    ir_payload = {
        "reply_text": reply_text[:12000],
        "question_block": question_block[:4000],
        "n_answer_markers": n_answer_markers,
        "instructions": (
            "The questions are in question_block. There are exactly n_answer_markers "
            "answer blocks in reply_text (each starts at Comment:/Answer:/Reply:). "
            "Do NOT create more questions than are in question_block, and do NOT treat "
            "(A)/(B)/(i)/(ii) headings inside an answer as questions."),
        "metadata_question_id": meta.get("question_id"),
        "n_tables": len(doc.tables),
        "answer_table_ids": [t.table_id for (t, isans) in table_objs if isans],
        "layout_case": layout.get("case"),
    }
    user = json.dumps(ir_payload, ensure_ascii=False)
    model_name = backend.model_for("llm") if hasattr(backend, "model_for") else getattr(backend, "name", "?")
    for attempt in range(cfg.llm_max_retries + 1):
        raw_out = None
        t0 = time.time()
        try:
            obj = backend.complete_json(system, user, schema_hint="pairs")
            raw_out = obj  # provider returns parsed dict; keep it as the raw record
            errs = validate_pairs(obj)
            dt = int((time.time() - t0) * 1000)
            if tracer:
                tracer.step("extraction", {
                    "path": "prose_model", "attempt": attempt,
                    "system_prompt": system, "user_prompt": user[:8000],
                    "raw_model_output": raw_out, "parsed_valid": not errs,
                    "validation_errors": errs, "layout_case": layout.get("case")},
                    model_name=model_name, duration_ms=dt)
            if not errs and _answers_grounded(obj, reply_text):
                return _pairs_from_model(obj, table_objs, meta, reply_text)
            if not errs:
                # JSON was valid but answers are NOT grounded in the source text
                # (the small model hallucinated / copied the few-shot example).
                errs = ["answers not grounded in source text"]
            user = (json.dumps(ir_payload, ensure_ascii=False)
                    + "\n\nYour previous output was invalid: " + "; ".join(errs)
                    + ". Return corrected STRICT JSON. Copy answers VERBATIM from "
                    + "reply_text only; never invent text.")
        except (BackendError, Exception) as e:
            dt = int((time.time() - t0) * 1000)
            if tracer:
                tracer.step("extraction", {
                    "path": "prose_model", "attempt": attempt,
                    "system_prompt": system, "user_prompt": user[:8000],
                    "raw_model_output": None, "error": f"{type(e).__name__}: {e}"},
                    model_name=model_name, duration_ms=dt)
            break
    return None  # fall back to deterministic


# Phrases from the few-shot examples in the prompt — if these appear in the model
# output it copied the example instead of extracting, so the output is rejected.
_FEWSHOT_LEAK = re.compile(
    r"achieved its targets|generation increased|DO letter is sent monthly|"
    r"attention is drawn to DISCOM dues|abnormal increase", re.I)


def _answers_grounded(obj, reply_text, min_overlap=0.5):
    """
    True if the model's answers are actually present in the source reply_text (not
    hallucinated or copied from the prompt examples). For each non-empty answer, the
    fraction of its words that occur in the source must be >= min_overlap, and it
    must not contain a known few-shot leak phrase.
    """
    if _FEWSHOT_LEAK.search(json.dumps(obj, ensure_ascii=False)):
        return False
    src_words = set(re.findall(r"\w+", (reply_text or "").lower()))
    if not src_words:
        return True  # nothing to check against
    for p in obj.get("pairs", []):
        ans = (p.get("answer_text") or "").strip()
        if len(ans) < 15:
            continue  # short/deferred answers (e.g. "May be replied by MoP.")
        words = re.findall(r"\w+", ans.lower())
        if not words:
            continue
        overlap = sum(1 for w in words if w in src_words) / len(words)
        if overlap < min_overlap:
            return False
    return True


def _pairs_from_model(obj, table_objs, meta=None, reply_text=""):
    all_tabs = [t for (t, _i) in table_objs]
    ans_tabs = [t for (t, isans) in table_objs if isans]
    pairs = []
    flags = ["model_extracted"]
    tables_index = [t.table_id for t in all_tabs]
    meta_qid = str((meta or {}).get("question_id", "")).strip()
    for p in obj["pairs"]:
        raw_qnum = str(p.get("question_number", "")).strip()
        qtext = p.get("question_text", "")
        qnum, moved_label = _sanitize_qnum(raw_qnum, meta_qid, qtext)
        if moved_label and moved_label.lower() not in qtext.lower():
            qtext = f"{moved_label} {qtext}".strip()
        if qnum != raw_qnum and "model_qnum_normalized" not in flags:
            flags.append("model_qnum_normalized")
        rel = [str(x).strip() for x in (p.get("related_question_numbers") or []) if str(x).strip()]
        rel = [r for r in rel if _looks_like_qnum(r)] or [qnum]
        pairs.append(Pair(
            question_number=qnum,
            question_text=qtext,
            answer_text=p.get("answer_text", ""),
            question_language=p.get("question_language") or detect_language(qtext),
            answer_language=p.get("answer_language") or detect_language(p.get("answer_text", "")),
            answer_is_table=bool(p.get("answer_is_table")),
            related_question_numbers=rel,
            # Attach tables. Use answer-role tables only when the LLM says the answer
            # IS a table AND such tables exist; otherwise attach ALL tables so a
            # detected table is NEVER dropped (fix 24: LLM said answer_is_table but
            # the classifier found only a 'supporting' table -> ans_tabs was empty).
            tables=ans_tabs if (p.get("answer_is_table") and ans_tabs) else all_tabs,
            confidence=p.get("confidence", "high"),
        ))

    # Answer-recovery: small models often return empty answer_text even though the
    # source clearly has a Comment:/Answer: for each question. Backfill those from
    # the reply text by aligning on question text and slicing to the next question.
    empties = [p for p in pairs if not p.answer_text.strip() and not p.answer_is_table]
    if empties and reply_text:
        recovered = _recover_answers(pairs, reply_text)
        if recovered:
            flags.append("answers_recovered_from_text")
        # any still-empty answer is a real gap -> mark low confidence for review
        for p in pairs:
            if not p.answer_text.strip() and not p.answer_is_table:
                p.confidence = "low"
        if any(not p.answer_text.strip() and not p.answer_is_table for p in pairs):
            flags.append("empty_answer_after_recovery")

    return pairs, tables_index, flags


def _recover_answers(pairs, text):
    """
    Fill empty pair.answer_text by locating each question in the source text and
    taking the segment after its Comment:/Answer:/Reply: marker up to the next
    question. Returns True if any answer was recovered.
    """
    ans_marker = re.compile(r"\b(comment|comments|answer|reply|उत्तर)\b\s*[:.\-]",
                            re.I)
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()

    # locate each question's start offset in the text by its first ~40 chars
    positions = []
    for p in pairs:
        needle = norm(re.sub(r"^\([a-z0-9]+\)\s*", "", p.question_text))[:40]
        idx = -1
        if needle:
            hay = norm(text)
            hpos = hay.find(needle)
            if hpos != -1:
                # map normalized position back approximately to raw text
                idx = _approx_raw_offset(text, hpos)
        positions.append(idx)

    recovered_any = False
    for i, p in enumerate(pairs):
        if p.answer_text.strip() or p.answer_is_table:
            continue
        start = positions[i]
        if start < 0:
            continue
        # end = next located question, or end of text
        later = [positions[j] for j in range(i + 1, len(pairs)) if positions[j] > start]
        end = min(later) if later else len(text)
        segment = text[start:end]
        m = ans_marker.search(segment)
        if m:
            ans = segment[m.end():].strip()
            # trim a trailing next-question label if it slipped in
            ans = re.sub(r"\s*\([a-z]\)\s.*$", "", ans, flags=re.S).strip() or ans
            if ans:
                p.answer_text = ans
                p.answer_language = detect_language(ans)
                recovered_any = True
    return recovered_any


def _approx_raw_offset(text, norm_pos):
    """Map a position in whitespace-normalized text back to raw text (approx)."""
    count = 0
    prev_space = False
    for i, ch in enumerate(text):
        is_space = ch.isspace()
        if is_space and prev_space:
            continue  # collapsed in normalized form
        if count >= norm_pos:
            return i
        count += 1
        prev_space = is_space
    return 0


# a plausible parliamentary diary/question number: optional letter prefix + digits,
# short. Accepts 1102, S-77, U1059, S3845, 12561; rejects "(a)" and long sentences.
_QNUM_OK = re.compile(r"^[A-Za-z]{0,4}[-\s]?\d{1,6}[A-Za-z]?$")
_SUBLABEL = re.compile(r"^\(?([a-z]|[ivx]{1,4})\)?$", re.I)


def _looks_like_qnum(s: str) -> bool:
    return bool(_QNUM_OK.match(s.strip())) if s else False


def _sanitize_qnum(raw: str, meta_qid: str, qtext: str):
    """
    Return (question_number, moved_sublabel_or_None). If the model put a sub-part
    label like '(a)' or a whole sentence into question_number, replace it with the
    metadata diary number and surface any sub-label so the caller can prefix it to
    question_text.
    """
    raw = (raw or "").strip()
    if _looks_like_qnum(raw):
        return raw, None
    m = _SUBLABEL.match(raw)
    if m:  # it was just a sub-part label
        return (meta_qid or raw), raw
    # long text or empty or junk -> fall back to metadata id; keep no label
    return (meta_qid or raw or "unknown"), None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _clubbed_numbers(text):
    m = re.search(
        r"(?:diary|question|dy\.?)\s*no\.?s?\.?\s*[:.]?\s*"
        r"([A-Z]?\s*\d{2,6}(?:\s*(?:,|and|&|/)\s*[A-Z]?\s*\d{2,6})+)",
        text, re.I,
    )
    if not m:
        return []
    nums = re.findall(r"[A-Z]?\s*\d{2,6}", m.group(1))
    return [re.sub(r"\s+", "", n) for n in nums]


def _question_body(text):
    # text before the first "Comment:/Answer:/Reply:" marker
    m = re.search(r"\b(comment|answer|reply|उत्तर)\b\s*[:.-]", text, re.I)
    return text[:m.start()].strip() if m else ""


def _answer_body(text):
    m = re.search(r"\b(comment|answer|reply|उत्तर)\b\s*[:.-]", text, re.I)
    return text[m.end():].strip() if m else text.strip()


def _slice_qa(text, qn, seen, i):
    """Best-effort slice of one question's text + answer from the reply."""
    body = _answer_body(text)
    qbody = _question_body(text)
    return qbody, body


def _crosscheck_metadata(pairs, meta, flags):
    """
    Reframed (Change 5): the folder question_id and the document diary number are
    SEPARATE identifiers and may legitimately differ (e.g. the folder is named for a
    provisional/advance number). That is NOT an error, so we no longer flag a
    mismatch. Identity separation is recorded in build_document_fields(); the only
    real problem is when NO diary number can be found at all, flagged there.
    """
    return


def _is_real_model(backend) -> bool:
    """True when the provider will make actual model calls (not deterministic)."""
    # LocalProvider sets llm_is_deterministic; NvidiaProvider is always a model.
    if getattr(backend, "kind", None) == "nvidia":
        return True
    return not getattr(backend, "llm_is_deterministic", True)


def _dedup(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out
