"""
Intermediate Representation (IR) shared across the pipeline.

The reader/router produces a `Document` (ordered Blocks + Tables with page numbers
and language tags). Layout analysis annotates it. Extraction consumes it and
produces the `parsed.json` output objects (Pair / TableOut).

Everything is plain dataclasses with `to_dict()` so JSON output is explicit and
schema-stable. No hidden serialization magic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# --- language detection primitives -----------------------------------------

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """Return 'en', 'hi', or 'mixed' for a piece of text. Never translates."""
    if not text:
        return "en"
    has_hi = bool(_DEVANAGARI.search(text))
    has_en = bool(_LATIN.search(text))
    if has_hi and has_en:
        # decide mixed vs dominant by rough character share
        hi = len(_DEVANAGARI.findall(text))
        en = len(_LATIN.findall(text))
        if hi >= 3 and en >= 3:
            return "mixed"
        return "hi" if hi > en else "en"
    if has_hi:
        return "hi"
    return "en"


# --- IR blocks & tables -----------------------------------------------------

@dataclass
class Block:
    """A run of text (paragraph / line group) in reading order."""
    text: str
    page: int
    kind: str = "text"           # text | heading | list_item
    language: str = "en"

    def to_dict(self):
        return {"text": self.text, "page": self.page, "kind": self.kind,
                "language": self.language}


@dataclass
class RawTable:
    """
    A table as extracted from the source (geometry cleaned but roles not yet
    assigned). `grid` is a list of rows; each row is a list of cell strings.
    """
    grid: list                    # list[list[str|None]]
    page: int
    caption: Optional[str] = None
    stitched_across_pages: bool = False
    merged_cells_present: bool = False
    extraction_confidence: str = "high"   # high | low
    source: str = "docling"       # docling | pdfplumber | ocr | docx | xlsx

    def n_rows(self):
        return len(self.grid)

    def n_cols(self):
        return max((len(r) for r in self.grid), default=0)

    def to_dict(self):
        return {
            "grid": self.grid, "page": self.page, "caption": self.caption,
            "stitched_across_pages": self.stitched_across_pages,
            "merged_cells_present": self.merged_cells_present,
            "extraction_confidence": self.extraction_confidence,
            "source": self.source,
        }


@dataclass
class Document:
    """Ordered blocks + tables plus document-level metadata."""
    blocks: list = field(default_factory=list)     # list[Block]
    tables: list = field(default_factory=list)     # list[RawTable]
    parser_used: str = ""
    ocr_used: bool = False
    visual_used: bool = False
    format_converted: bool = False
    page_count: int = 0
    flags: list = field(default_factory=list)      # extraction flags accumulate here
    page_routing: list = field(default_factory=list)   # list[dict] per-page decisions
    models_used: dict = field(default_factory=dict)    # {op: model_name} actually used

    def full_text(self) -> str:
        return "\n".join(b.text for b in self.blocks if b.text)

    def language(self) -> str:
        return detect_language(self.full_text()[:20000])

    def add_flag(self, flag: str):
        if flag not in self.flags:
            self.flags.append(flag)


# --- output objects (parsed.json) ------------------------------------------

@dataclass
class Column:
    name: str
    role: str = "other"          # qno|question|answer|project_name|location|status|...
    language: str = "en"

    def to_dict(self):
        return {"name": self.name, "role": self.role, "language": self.language}


@dataclass
class Row:
    row_id: str
    cells: dict
    row_language: str = "en"
    nl_rendering: str = ""
    entities: list = field(default_factory=list)

    def to_dict(self):
        return {"row_id": self.row_id, "cells": self.cells,
                "row_language": self.row_language,
                "nl_rendering": self.nl_rendering, "entities": self.entities}


@dataclass
class TableOut:
    table_id: str
    table_role: str              # qa_pairs | answer_data | supporting
    answer_is_table: bool
    columns: list = field(default_factory=list)   # list[Column]
    rows: list = field(default_factory=list)      # list[Row]
    caption: Optional[str] = None
    stitched_across_pages: bool = False
    extraction_confidence: str = "high"

    def to_dict(self):
        return {
            "table_id": self.table_id, "caption": self.caption,
            "table_role": self.table_role, "answer_is_table": self.answer_is_table,
            "columns": [c.to_dict() for c in self.columns],
            "rows": [r.to_dict() for r in self.rows],
            "stitched_across_pages": self.stitched_across_pages,
            "extraction_confidence": self.extraction_confidence,
        }


@dataclass
class Pair:
    question_number: str
    question_text: str
    answer_text: str = ""
    question_language: str = "en"
    answer_language: str = "en"
    answer_is_table: bool = False
    related_question_numbers: list = field(default_factory=list)
    tables: list = field(default_factory=list)     # list[TableOut]
    confidence: str = "high"

    def to_dict(self):
        return {
            "question_number": self.question_number,
            "question_text": self.question_text,
            "question_language": self.question_language,
            "answer_text": self.answer_text,
            "answer_language": self.answer_language,
            "answer_is_table": self.answer_is_table,
            "related_question_numbers": self.related_question_numbers,
            "tables": [t.to_dict() for t in self.tables],
            "confidence": self.confidence,
        }


@dataclass
class AnswerBlock:
    """A headed section within a long answer (preserves internal structure)."""
    heading: str
    text: str

    def to_dict(self):
        return {"heading": self.heading, "text": self.text}


@dataclass
class SubQuestion:
    """
    One sub-part (a/b/c...) of a parliamentary diary: its question text and a POINTER
    to the answer group that answers it. The answer itself lives ONCE on the
    AnswerGroup (never duplicated across parts). `annexure_refs` lists the annexure
    labels this part cites (for per-part UI buttons).
    """
    part_label: str                       # "a", "b", ... or "(single)"
    question_text: str
    sub_question_id: str = ""             # GLOBALLY unique: "<question_id>_<part>"
    answer_group_id: str = ""             # -> AnswerGroup.answer_group_id (globally uniq)
    question_language: str = "en"
    annexure_refs: list = field(default_factory=list)

    def to_dict(self):
        return {
            # sub_question_id is the RETRIEVAL UNIT id (Phase 3 embeds question_text
            # and stores it keyed by this id). Globally unique across the corpus.
            "sub_question_id": self.sub_question_id,
            "part_label": self.part_label,
            "question_text": self.question_text,
            "question_language": self.question_language,
            "answer_group_id": self.answer_group_id,
            "annexure_refs": self.annexure_refs,
        }


@dataclass
class AnswerGroup:
    """
    One answer, stored ONCE, declaring which sub-parts it covers. Tables that belong
    to this answer live INSIDE it (structural answer<->table link). A shared answer
    (covering several parts) is a single group listing all those parts.
    """
    answer_group_id: str                  # "g1", "g2", ...
    answers_parts: list                   # ["a","b"] — parts this answer covers
    answer_text: str = ""
    answer_type: str = "substantive"      # substantive|deferred_to_ministry|nil|not_applicable
    answer_language: str = "en"
    answer_is_table: bool = False
    answer_blocks: list = field(default_factory=list)   # list[AnswerBlock]
    tables: list = field(default_factory=list)          # list[TableOut] — THIS answer's tables
    annexure_refs: list = field(default_factory=list)
    confidence: str = "high"

    def to_dict(self):
        return {
            "answer_group_id": self.answer_group_id,
            "answers_parts": self.answers_parts,
            "answer_text": self.answer_text,
            "answer_type": self.answer_type,
            "answer_language": self.answer_language,
            "answer_is_table": self.answer_is_table,
            "answer_blocks": [b.to_dict() for b in self.answer_blocks],
            "tables": [t.to_dict() for t in self.tables],
            "annexure_refs": self.annexure_refs,
            "confidence": self.confidence,
        }
