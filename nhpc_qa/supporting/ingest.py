"""
Ingest ONE supporting document into the supporting_* tables.

REUSES the Q&A parse layer, does NOT touch it. We call read_document() -- the same
Nemotron/Docling reader, per-page routing and OCR the Q&A pipeline uses -- and take its
Document (full text + RawTable objects). We deliberately do NOT call pipeline.process_one()
or extract_qa(): those find parliamentary QUESTIONS, and a financial report has none.

WHOLE-DOCUMENT, NO CHUNKING. These files are 1-5 pages and mostly dense tables. Chunking
would split a 5-fiscal-year table down the middle and destroy it. The whole document text
and every table go into the DB intact, and the whole thing is passed to the LLM at draft
time -- small enough to fit.

FLAG, DON'T GUESS. A low-confidence table extraction is recorded (needs_review + a
parse_flag), never silently trusted. A wrong financial figure in an officer's draft is the
exact failure this whole system exists to prevent.

IDEMPOTENT. doc_key = '<category>/<sha256[:16]>'. Re-uploading the same bytes upserts the
same row -- no duplicate. A soft-deleted document with the same bytes REACTIVATES.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os

from nhpc_qa.core.logging import get_logger
from nhpc_qa.pipeline.parse.reader import read_document
from nhpc_qa.pipeline.parse.dates import validate as validate_date

log = get_logger("nhpc.supporting.ingest")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _table_orientation(grid) -> str:
    """
    'transposed' when the FIRST COLUMN looks like a list of attribute names and the header
    row looks like entity names -- the UC-projects layout, where PROJECTS ARE COLUMNS.

    Heuristic, and it only sets a HINT the draft prompt passes to the LLM; it never rewrites
    the data. Getting it wrong costs a prompt hint, not a figure.
    """
    if len(grid) < 3 or (grid and len(grid[0]) < 3):
        return "rows"
    # transposed tables tend to have a tall, narrow shape with text-heavy first column
    first_col = [(_r[0] or "") for _r in grid if _r]
    texty = sum(1 for c in first_col if c and not _looks_numeric(c))
    # attribute-labels down the side: most of the first column is non-numeric text
    return "transposed" if texty >= max(3, 0.7 * len(first_col)) else "rows"


def _looks_numeric(s: str) -> bool:
    t = str(s or "").replace(",", "").replace("%", "").replace("(", "").replace(")", "").strip()
    if not t:
        return False
    try:
        float(t)
        return True
    except ValueError:
        return False


def _nl_render(grid) -> str:
    """Flatten a table to a natural-language block the LLM can read. Whole cells; no loss."""
    if not grid:
        return ""
    header = grid[0]
    lines = [" | ".join(str(c or "") for c in header)]
    for row in grid[1:]:
        lines.append(" | ".join(str(c or "") for c in row))
    return "\n".join(lines)


def parse_supporting_file(cfg, abs_path: str, provider=None) -> dict:
    """
    Read a file through the existing parse layer. Returns a dict ready to store -- text,
    tables (with orientation + confidence), page_count, flags. Never raises.
    """
    doc = read_document(abs_path, cfg, provider=provider)

    flags = list(doc.flags)
    needs_review = False
    tables = []
    for i, t in enumerate(doc.tables):
        conf = getattr(t, "extraction_confidence", "high")
        if conf == "low":
            needs_review = True
            flags.append(f"table_{i}_low_confidence")
        if getattr(t, "merged_cells_present", False):
            flags.append(f"table_{i}_merged_cells")
        grid = t.grid or []
        orientation = _table_orientation(grid)
        tables.append({
            "table_index": i,
            "page": getattr(t, "page", None),
            "orientation": orientation,
            "extraction_confidence": conf,
            "n_rows": len(grid),
            "columns": [{"name": str(c or ""), "role": "other"}
                        for c in (grid[0] if grid else [])],
            "nl_rendering": _nl_render(grid),
            "rows": [{"row_index": ri, "cells": {str(ci): (cell or "")
                                                 for ci, cell in enumerate(row)},
                      "nl_rendering": " | ".join(str(c or "") for c in row)}
                     for ri, row in enumerate(grid[1:], start=1)],
        })

    if not doc.tables:
        flags.append("no_tables_extracted")   # a table-heavy doc with none is worth noting
    if doc.parser_used == "error":
        needs_review = True

    return {
        "document_text": doc.full_text(),
        "page_count": doc.page_count or None,
        "parser_used": doc.parser_used,
        "tables": tables,
        "parse_flags": _dedup(flags),
        "needs_review": needs_review,
    }


def propose_as_of(cfg, llm, document_text: str) -> dict:
    """
    Ask the LLM for the document's as-of date and period label. The ADMIN confirms it before
    it is stored (see the upload route) -- this only PRE-FILLS the form.

    Returns {"as_of_date": "YYYY-MM-DD"|None, "period_label": str|None}. Never raises; on any
    failure returns nulls and the admin fills them in by hand (the field stays mandatory).
    """
    if llm is None or not (document_text or "").strip():
        return {"as_of_date": None, "period_label": None}

    system = (
        "You read NHPC internal reference documents (financial digests, project progress, "
        "CSR) and report their reporting PERIOD. Return STRICT JSON only:\n"
        '{"as_of_date": "YYYY-MM-DD" or null, "period_label": "<short label>" or null}\n'
        "- as_of_date: a single 'as on'/'as at' date if the document states one "
        "(e.g. 'Status as on 30.06.2026' -> 2026-06-30). null if there is none.\n"
        "- period_label: the human-readable reporting period. For a multi-year financial "
        "table spanning several fiscal years, give the RANGE (e.g. 'FY 2020-21 to "
        "2024-25'). For a status snapshot, echo it ('as on 30.06.2026').\n"
        "- Report the DOCUMENT'S period, never today's date. If unclear, use null.")
    try:
        raw = llm.complete_text(system, f"DOCUMENT:\n{document_text[:6000]}",
                                max_tokens=200, temperature=0.0)
    except Exception as e:      # noqa: BLE001 -- the admin can always type it
        log.warning("propose_as_of: llm failed (%s) — admin will enter it", type(e).__name__)
        return {"as_of_date": None, "period_label": None}

    obj = _loads(raw)
    if not isinstance(obj, dict):
        return {"as_of_date": None, "period_label": None}
    # Validate the date the SAME way reply dates are validated -- calendar-valid and inside
    # the plausible window. The model is trusted to FIND, not to hand us a valid date.
    d = validate_date(obj.get("as_of_date")) if obj.get("as_of_date") else None
    return {"as_of_date": d.isoformat() if d else None,
            "period_label": (obj.get("period_label") or None)}


def make_doc_key(category: str, sha256: str) -> str:
    return f"{category}/{sha256[:16]}"


def store(conn, cfg, *, category, display_name, file_path, original_filename, sha256,
          parsed: dict, as_of_date, period_label, uploaded_by):
    """
    Upsert the document and its tables. Idempotent on doc_key; reactivates a soft-deleted
    row with the same bytes. Returns the supporting_documents.id.
    """
    doc_key = make_doc_key(category, sha256)
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO supporting_documents
                (category, doc_key, display_name, file_path, original_filename, sha256,
                 page_count, as_of_date, period_label, document_text, parse_flags,
                 needs_review, raw_parse, uploaded_by, is_active, deleted_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,true,NULL)
            ON CONFLICT (doc_key) DO UPDATE SET
                display_name = EXCLUDED.display_name,
                file_path    = EXCLUDED.file_path,
                as_of_date   = EXCLUDED.as_of_date,
                period_label = EXCLUDED.period_label,
                document_text= EXCLUDED.document_text,
                parse_flags  = EXCLUDED.parse_flags,
                needs_review = EXCLUDED.needs_review,
                raw_parse    = EXCLUDED.raw_parse,
                is_active    = true,           -- re-upload REACTIVATES a soft-deleted doc
                deleted_at   = NULL
            RETURNING id
        """, (category, doc_key, display_name, file_path, original_filename, sha256,
              parsed.get("page_count"), as_of_date, period_label,
              parsed.get("document_text"), parsed.get("parse_flags") or [],
              parsed.get("needs_review", False),
              json.dumps({"parser_used": parsed.get("parser_used")}), uploaded_by))
        doc_id = cur.fetchone()[0]

        # rewrite the tables (cascade-delete then insert -> clean on re-upload)
        cur.execute("DELETE FROM supporting_document_tables WHERE supporting_doc_id = %s",
                    (doc_id,))
        for t in parsed.get("tables") or []:
            cur.execute("""
                INSERT INTO supporting_document_tables
                    (supporting_doc_id, table_index, page, columns, n_rows, orientation,
                     extraction_confidence, nl_rendering)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id
            """, (doc_id, t["table_index"], t.get("page"),
                  json.dumps(t.get("columns") or []), t.get("n_rows"),
                  t.get("orientation", "rows"), t.get("extraction_confidence", "high"),
                  t.get("nl_rendering")))
            table_id = cur.fetchone()[0]
            for r in t.get("rows") or []:
                cur.execute("""
                    INSERT INTO supporting_document_rows
                        (table_id, row_index, cells, nl_rendering)
                    VALUES (%s,%s,%s,%s)
                """, (table_id, r["row_index"], json.dumps(r.get("cells") or {}),
                      r.get("nl_rendering")))
    conn.commit()
    log.info("supporting: stored %s (%d table(s), needs_review=%s)",
             doc_key, len(parsed.get("tables") or []), parsed.get("needs_review"))
    return doc_id


def _dedup(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def _loads(raw: str):
    import re
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None
