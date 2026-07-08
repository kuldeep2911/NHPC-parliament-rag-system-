"""
Layout + table-structure analysis. Runs BEFORE Q&A extraction and decides which
extraction path is used.

Outputs:
    layout_structure : mostly_prose | prose_with_tables | qa_table
    layout_case      : A | B | C | unknown   (prose paths only)
    qa_table_ref     : the RawTable that IS the Q&A table (when qa_table), else None

Classification is intentionally conservative and explainable (regex/geometry, no
model), so the routing decision is auditable.
"""

from __future__ import annotations

import re
from .tables import ROLE_PATTERNS, clean_table

# question-number markers seen in these documents
QNUM_RE = re.compile(
    r"\b(?:Q\.?\s*No\.?|Question\s*No\.?|Dy\.?\s*No\.?|Diary\s*No\.?|"
    r"U\s*No\.?|S\s*No\.?|USQ|LSUSQ|RSUQ)\s*[:.]?\s*([A-Z]?\s*\d{2,6})",
    re.I,
)
PART_RE = re.compile(r"\((?:[a-e]|[ivx]{1,4})\)", re.I)  # (a) (b) (i) (ii) parts


def _table_qa_role(raw):
    """Return the cleaned table dict if this table looks like a Q&A-in-columns
    table (has both a question-role and an answer-role column), else None."""
    cleaned = clean_table(raw, "probe")
    if not cleaned:
        return None
    roles = [c.role for c in cleaned["columns"]]
    has_q = "question" in roles
    has_a = "answer" in roles
    if has_q and has_a:
        return {"cleaned": cleaned, "header_derived": True}

    # Entity/data tables (Type-2) must NOT be inferred as Q&A tables: the presence
    # of a project/location/status/capacity column is a strong Type-2 signal.
    entity_roles = {"project_name", "location", "status", "capacity", "cost",
                    "date_timeline", "percentage_complete"}
    if entity_roles & set(roles):
        return None

    # Content-based inference for HEADERLESS Q&A tables only. Requires: a short
    # id-ish first column, a genuinely interrogative question column (varied text,
    # not a repeated boilerplate), and a distinct answer column.
    header = cleaned["header"]
    body = cleaned["body"]
    if not (2 <= len(header) <= 4 and len(body) >= 3):
        return None
    import statistics
    col_len = [statistics.mean([len(r[c]) for r in body if c < len(r)] or [0])
               for c in range(len(header))]
    long_cols = [i for i, L in enumerate(col_len) if L > 40]
    if len(long_cols) < 2:
        return None  # a real Q&A table has BOTH a long question and a long answer

    # the candidate question column must contain DISTINCT text across rows
    qcol = long_cols[0]
    qvals = [r[qcol] for r in body if qcol < len(r) and r[qcol]]
    distinct_ratio = len(set(qvals)) / max(1, len(qvals))
    if distinct_ratio < 0.6:
        return None  # repeated/boilerplate column -> not per-row questions

    # and it should look interrogative / question-like at least sometimes
    interrog = sum(1 for v in qvals if re.search(
        r"\b(whether|what|how|why|number of|details|give|status of)\b", v, re.I)
        or v.strip().endswith("?"))
    if interrog == 0:
        return None
    return {"cleaned": cleaned, "header_derived": False}


def analyze_layout(doc):
    """
    Classify document layout. Returns a dict:
        {structure, case, qa_table_ref, qa_table_header_derived, flags}
    """
    flags = []
    text = doc.full_text()
    n_tables = len(doc.tables)
    text_len = len(text.strip())

    # 0) email-wrapper detection: many "answers" are Zimbra/forwarded emails whose
    # real reply is an attachment, not inline text. Flag so it goes to review.
    if _looks_like_email_wrapper(text):
        flags.append("email_wrapper_reply_may_be_attachment")

    # 1) qa_table? — a single dominant table whose columns ARE the Q&A structure
    qa_ref = None
    qa_header_derived = None
    for raw in doc.tables:
        probe = _table_qa_role(raw)
        if probe:
            # qualifies as QA table if the table is the bulk of the document
            table_chars = sum(len(x or "") for row in raw.grid for x in row)
            if table_chars >= 0.4 * max(1, text_len) or text_len < 400:
                qa_ref = raw
                qa_header_derived = probe["header_derived"]
                break

    if qa_ref is not None:
        if not qa_header_derived:
            flags.append("qa_table_columns_inferred")
        return {
            "structure": "qa_table", "case": "unknown",
            "qa_table_ref": qa_ref, "qa_table_header_derived": qa_header_derived,
            "flags": flags,
        }

    # 2) prose vs prose_with_tables
    structure = "prose_with_tables" if n_tables > 0 else "mostly_prose"

    # 3) layout case detection (prose)
    case = _detect_case(text)
    return {
        "structure": structure, "case": case,
        "qa_table_ref": None, "qa_table_header_derived": None, "flags": flags,
    }


def _detect_case(text: str) -> str:
    """
    A: all questions first, then all answers.
    B: Q, A, Q, A interleaved.
    C: one combined answer for 2-3 clubbed question numbers.
    """
    qnums = QNUM_RE.findall(text)
    uniq = []
    for q in qnums:
        qn = re.sub(r"\s+", "", q)
        if qn not in uniq:
            uniq.append(qn)

    # clubbed marker: "diary no. U1059, U3429, U3475" or "Question No 1894, 1908"
    clubbed = re.search(
        r"(?:diary|question|dy\.?)\s*no\.?s?\.?\s*[:.]?\s*"
        r"[A-Z]?\s*\d{2,6}\s*(?:,|and|&|/)\s*[A-Z]?\s*\d{2,6}",
        text, re.I,
    )
    if clubbed or len(uniq) >= 2 and _single_answer_block(text):
        return "C"

    # interleaving: answers appear between question markers
    ans_markers = [m.start() for m in re.finditer(r"\b(answer|reply|उत्तर)\b", text, re.I)]
    q_markers = [m.start() for m in re.finditer(QNUM_RE, text)]
    if len(q_markers) >= 2 and len(ans_markers) >= 2:
        # if answer markers are interspersed among question markers -> B
        interleaved = 0
        for i in range(len(q_markers) - 1):
            if any(q_markers[i] < a < q_markers[i + 1] for a in ans_markers):
                interleaved += 1
        if interleaved >= 1:
            return "B"
        return "A"
    if len(uniq) <= 1:
        return "B"  # single Q with its answer following
    return "unknown"


def _single_answer_block(text: str) -> bool:
    """Heuristic: the reply reads as one continuous answer (few 'answer' breaks)."""
    return len(re.findall(r"\b(answer|reply)\b", text, re.I)) <= 1


def _looks_like_email_wrapper(text: str) -> bool:
    """Detect Zimbra / forwarded-email wrappers whose real reply is an attachment."""
    head = text[:1200]
    signals = 0
    if re.search(r"\bZimbra\b", head, re.I):
        signals += 1
    # count distinct email headers present (From/To/Cc/Subject/Sent)
    hdrs = len(set(re.findall(r"\b(From|To|Cc|Subject|Sent)\s*:", head, re.I)))
    if hdrs >= 2:
        signals += 1
    if re.search(r"\b\d+\s+attachments?\b", text, re.I) or \
       re.search(r"\b\w+\.(docx?|pdf|xlsx?)\b\s+\d+\s*(KB|MB)", text, re.I):
        signals += 1
    if re.search(r"\b(Fwd:|Re:)\s", head):
        signals += 1
    # An email wrapper is signalled by multiple email cues AND either strong
    # corroboration (3+ cues) or a short inline body (the reply is elsewhere).
    if signals >= 2:
        return signals >= 3 or len(text.strip()) < 1800
    return False
