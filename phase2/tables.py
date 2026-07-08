"""
Table geometry cleaning + role tagging.

Raw table extraction (esp. pdfplumber) is noisy: empty padding columns, cells
split across several physical header rows, merged cells appearing as blank
neighbours. This module normalizes a RawTable grid into a clean logical table,
then tags column roles so downstream contradiction detection is possible.

It deliberately does NOT flatten tables into prose. If geometry is inconsistent
it lowers confidence and flags rather than emitting a confident-but-wrong table.
"""

from __future__ import annotations

import re
from .ir import Column, Row, RawTable, TableOut, detect_language

# --- column role vocabulary -------------------------------------------------

ROLE_PATTERNS = [
    ("qno", re.compile(r"\b(q\.?\s*no|sl\.?\s*no|s\.?\s*no|sr\.?\s*no|serial|क्र|क्रम)\b", re.I)),
    ("question", re.compile(r"\b(question|query|प्रश्न|text of question)\b", re.I)),
    ("answer", re.compile(r"\b(answer|reply|comment|response|उत्तर|format for reply)\b", re.I)),
    ("project_name", re.compile(r"\b(project|power ?station|plant|scheme|परियोजना)\b", re.I)),
    ("location", re.compile(r"\b(state|location|district|site|स्थान|राज्य)\b", re.I)),
    ("status", re.compile(r"\b(status|stage|commissioned|under construction|स्थिति)\b", re.I)),
    ("date_timeline", re.compile(r"\b(date|timeline|year|schedule|cod|from|on|तिथि|वर्ष)\b", re.I)),
    ("capacity", re.compile(r"\b(capacity|mw|मेगावाट|installed)\b", re.I)),
    ("cost", re.compile(r"\b(cost|crore|₹|rs\.?|expenditure|लागत|amount)\b", re.I)),
    ("percentage_complete", re.compile(r"(%|percent|progress|completion|प्रतिशत)", re.I)),
]


def _clean_cell(v):
    if v is None:
        return ""
    return re.sub(r"\s+", " ", str(v)).strip()


def _drop_empty_columns(grid):
    """Remove columns that are empty in every row (pdfplumber padding artifacts)."""
    if not grid:
        return grid, False
    ncol = max(len(r) for r in grid)
    norm = [list(r) + [""] * (ncol - len(r)) for r in grid]
    keep = []
    for c in range(ncol):
        if any(_clean_cell(norm[r][c]) for r in range(len(norm))):
            keep.append(c)
    dropped = len(keep) < ncol
    trimmed = [[row[c] for c in keep] for row in norm]
    return trimmed, dropped


def _merge_split_columns(grid):
    """
    Repair the pdfplumber pattern where ONE logical column is split into two:
    a column that only carries header text (data rows blank) immediately followed
    by a column that only carries data (header blank). Merge such adjacent pairs
    so the header label and its data live in the same column.
    """
    if not grid or len(grid) < 2:
        return grid, False
    ncol = max(len(r) for r in grid)
    norm = [list(r) + [""] * (ncol - len(r)) for r in grid]
    hdr = 0

    def header_only(c):
        return bool(_clean_cell(norm[hdr][c])) and not any(
            _clean_cell(norm[r][c]) for r in range(1, len(norm)))

    def data_only(c):
        return (not _clean_cell(norm[hdr][c])) and any(
            _clean_cell(norm[r][c]) for r in range(1, len(norm)))

    merged_any = False
    c = 0
    out_cols = []          # list of column-index groups to merge
    while c < ncol:
        if c + 1 < ncol and header_only(c) and data_only(c + 1):
            out_cols.append([c, c + 1])
            merged_any = True
            c += 2
        else:
            out_cols.append([c])
            c += 1

    if not merged_any:
        return grid, False

    new_grid = []
    for row in norm:
        new_row = []
        for group in out_cols:
            vals = [_clean_cell(row[i]) for i in group if _clean_cell(row[i])]
            new_row.append(" ".join(vals))
        new_grid.append(new_row)
    return new_grid, True


def _merge_header_rows(grid):
    """
    Detect multi-row headers and merge them into a single logical header.
    Heuristic: leading rows where the FIRST data-ish column is empty or which
    have many empty cells are treated as header continuation and joined
    column-wise with the first row.
    Returns (header:list[str], body:list[list[str]], header_rows_used:int).
    """
    if not grid:
        return [], [], 0
    ncol = len(grid[0])

    # Skip leading PREAMBLE/title rows (common in xlsx exports): a row that is
    # mostly empty except a single long text cell, sitting above a denser row that
    # is the real header. Only skip while a later, denser row exists to be the
    # header — never strip everything.
    def _density(row):
        return sum(1 for c in row if _clean_cell(c))

    # Find the first "dense" row (>=2 filled cells or >=40% of columns) — that is
    # the real header. Everything strictly above it that is sparse (<=1 filled cell)
    # is preamble/title/unit noise and is skipped (captured as caption).
    if ncol >= 2:
        dense_at = None
        for i, row in enumerate(grid):
            filled = _density(row)
            if filled >= max(2, int(0.4 * ncol)):
                dense_at = i
                break
        if dense_at and all(_density(grid[j]) <= 1 for j in range(dense_at)):
            preamble_rows = grid[:dense_at]
            grid = grid[dense_at:]
            ncol = len(grid[0])
        else:
            preamble_rows = []
    else:
        preamble_rows = []

    def _starts_serial(row):
        first = _clean_cell(row[0]) if row else ""
        return bool(re.match(r"^\d+$", first))

    def _is_labelish(v):
        # short label under a spanning header: a year, a unit, a code, a word or two
        if not v or len(v) > 24:
            return False
        return bool(re.match(r"^(19|20)\d\d(-\d\d)?$", v)) or len(v.split()) <= 3

    def looks_like_header_cont(row, idx):
        cells = [_clean_cell(c) for c in row]
        nonempty = sum(1 for c in cells if c)
        if _starts_serial(cells):
            return False  # a data row, not a header continuation
        # (a) sparse continuation row (few cells filled), OR
        if 0 < nonempty <= max(1, ncol // 2):
            return True
        # (b) a DENSE row of short label-like cells (spanning sub-header, e.g. years
        #     2018-19 | 2017-18 | ...) followed by a clear data row (serial start).
        labelish = sum(1 for c in cells if _is_labelish(c))
        next_is_data = idx + 1 < len(grid) and _starts_serial(grid[idx + 1])
        if next_is_data and labelish >= max(2, nonempty - 1):
            return True
        return False

    header_rows = 1
    for i in range(1, min(4, len(grid))):
        if looks_like_header_cont(grid[i], i):
            header_rows += 1
        else:
            break

    header_cells = []
    for c in range(ncol):
        parts = []
        for r in range(header_rows):
            v = _clean_cell(grid[r][c]) if c < len(grid[r]) else ""
            if v and v not in parts:
                parts.append(v)
        header_cells.append(" ".join(parts).strip())

    # fill blank header names with a positional label (keeps columns addressable)
    for c in range(ncol):
        if not header_cells[c]:
            header_cells[c] = f"col{c+1}"

    body = [[_clean_cell(x) for x in row] for row in grid[header_rows:]]
    preamble_text = " ".join(
        _clean_cell(c) for row in preamble_rows for c in row if _clean_cell(c)
    ).strip() or None
    return header_cells, body, header_rows, preamble_text


def _tag_role(name: str) -> str:
    for role, pat in ROLE_PATTERNS:
        if pat.search(name or ""):
            return role
    return "other"


def _stitch_wrapped_rows(header, body):
    """
    Stitch rows that are clearly continuations of the previous row (a wrapped
    cell): a row whose leading qno/serial column is empty but which has content
    elsewhere gets appended to the row above. Returns (rows, stitched_count).
    """
    if not body:
        return body, 0
    stitched = 0
    out = []
    for row in body:
        first = row[0] if row else ""
        has_id = bool(re.match(r"^\s*\d", first))
        if out and not has_id and any(c for c in row):
            # continuation: append cellwise
            prev = out[-1]
            for i in range(min(len(prev), len(row))):
                if row[i]:
                    prev[i] = (prev[i] + " " + row[i]).strip()
            stitched += 1
        else:
            out.append(list(row))
    return out, stitched


def clean_table(raw: RawTable, table_id: str):
    """
    Normalize a RawTable into (header, body, columns, flags, confidence).
    Pure geometry — role/answer semantics decided by caller/classifier.
    """
    flags = []
    grid = [[_clean_cell(c) for c in row] for row in raw.grid]
    grid = [row for row in grid if any(row)]  # drop fully-empty rows
    if not grid:
        return None

    grid, dropped = _drop_empty_columns(grid)
    grid, split_merged = _merge_split_columns(grid)
    if split_merged:
        # re-drop any columns left empty after merging
        grid, _ = _drop_empty_columns(grid)
    header, body, hrows, preamble_text = _merge_header_rows(grid)
    if hrows > 1:
        flags.append("merged_cells_present")  # split/merged header collapsed
    if preamble_text:
        flags.append("table_preamble_skipped")  # title rows above header captured

    body, stitched = _stitch_wrapped_rows(header, body)

    # consistency check: rows wildly varying width => low confidence
    widths = {len(r) for r in body} | {len(header)}
    confidence = "high"
    if len(widths) > 1 and (max(widths) - min(widths)) > 1:
        confidence = "low"
        flags.append("table_alignment_uncertain")
    if raw.extraction_confidence == "low":
        confidence = "low"

    columns = [Column(name=h, role=_tag_role(h), language=detect_language(h))
               for h in header]
    if all(c.role == "other" for c in columns):
        flags.append("table_role_uninferred")

    return {
        "header": header, "body": body, "columns": columns,
        "flags": flags, "confidence": confidence, "stitched": stitched,
        "preamble_text": preamble_text,
    }


def build_table_out(raw: RawTable, table_id: str, table_role: str,
                    answer_is_table: bool):
    """
    Produce a TableOut (+ flags) from a RawTable. Emits one Row per body row with
    an NL rendering so each row is independently embeddable later.
    """
    cleaned = clean_table(raw, table_id)
    if cleaned is None:
        return None, ["empty_table"]

    header = cleaned["header"]
    columns = cleaned["columns"]
    rows = []
    for i, body_row in enumerate(cleaned["body"], start=1):
        cells = {}
        for c, colname in enumerate(header):
            cells[colname] = body_row[c] if c < len(body_row) else ""
        nl = "; ".join(f"{k}: {v}" for k, v in cells.items() if v)
        entities = _guess_entities(cells, columns)
        rows.append(Row(
            row_id=f"{table_id}_r{i}", cells=cells,
            row_language=detect_language(nl), nl_rendering=nl, entities=entities,
        ))

    tout = TableOut(
        table_id=table_id, table_role=table_role,
        answer_is_table=answer_is_table, columns=columns, rows=rows,
        caption=raw.caption, stitched_across_pages=raw.stitched_across_pages,
        extraction_confidence=cleaned["confidence"],
    )
    flags = list(cleaned["flags"])
    if raw.stitched_across_pages:
        flags.append("table_stitched_across_pages")
    if raw.source == "ocr":
        flags.append("table_ocr_extracted")
    if cleaned["stitched"]:
        flags.append("table_wrapped_rows_stitched")
    return tout, flags


def _guess_entities(cells: dict, columns):
    ents = []
    for col in columns:
        if col.role in ("project_name", "location") and cells.get(col.name):
            ents.append(cells[col.name])
    return ents
