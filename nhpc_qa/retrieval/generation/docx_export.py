"""
The draft as a Word document.

python-docx is ALREADY a dependency (the parser reads .docx); it writes them too, so this
adds nothing to the install.

DEVANAGARI. A .docx is UTF-8 XML, so the text survives regardless. What does NOT survive is
the RENDERING: Word picks the font for a script from the run's *complex-script* setting
(w:cs), and python-docx does not expose it. Set only the Latin font and Hindi text falls
back to whatever Word guesses -- often a box glyph. _font() sets w:cs explicitly, so
Devanagari renders in a font that actually has the glyphs.

The DRAFT notice appears THREE times -- a red banner at the top, a line under every drafted
part, and the footer of every page. That is deliberate. This document will be printed,
forwarded, and pasted from. A single notice at the top disappears the moment someone copies
the middle of page 2.
"""

from __future__ import annotations

import datetime as dt
import io
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

# A font with BOTH Latin and Devanagari coverage, present on any Windows/Office install.
# Nirmala UI is Microsoft's own Indic UI font.
_LATIN = "Calibri"
_DEVA = "Nirmala UI"

_RED = RGBColor(0xB0, 0x2A, 0x22)
_GREY = RGBColor(0x6B, 0x71, 0x78)
_NAVY = RGBColor(0x1B, 0x3A, 0x63)


def _font(run, *, size=11, bold=False, italic=False, color=None, hindi=False):
    """
    Style a run, and set the COMPLEX-SCRIPT font too.

    Without the w:cs element, Word renders Devanagari with whatever it falls back to --
    frequently tofu boxes. python-docx has no API for it, so we reach into the XML.
    """
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color
    name = _DEVA if hindi else _LATIN
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    # ascii/hAnsi = Latin; cs = complex script (Devanagari). Setting cs is the whole point.
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    rfonts.set(qn("w:cs"), _DEVA)
    return run


_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def _is_hindi(text: str) -> bool:
    return bool(_DEVANAGARI.search(text or ""))


def _para(doc, text="", *, size=11, bold=False, italic=False, color=None,
          space_after=6, align=None):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    if text:
        _font(p.add_run(text), size=size, bold=bold, italic=italic, color=color,
              hindi=_is_hindi(text))
    return p


def _heading(doc, text):
    p = _para(doc, text, size=12, bold=True, color=_NAVY, space_after=4)
    return p


NOTICE = ("DRAFT — FOR OFFICER REVIEW. Generated from NHPC's past parliamentary replies. "
          "Verify every figure and claim against the cited sources before use. "
          "This is NOT an approved reply.")


def build_docx(query: str, draft: dict, *, user_email: str | None = None,
               run_id: str | None = None) -> bytes:
    """The draft as .docx bytes. Pure function -- no filesystem, no network."""
    doc = Document()

    # Base style, so anything we do not touch still has Devanagari coverage.
    normal = doc.styles["Normal"]
    normal.font.name = _LATIN
    normal.font.size = Pt(11)
    normal.element.rPr.rFonts.set(qn("w:cs"), _DEVA)

    # ---- title + the notice, unmissable --------------------------------
    _para(doc, "NHPC — Parliamentary Reply", size=17, bold=True, color=_NAVY, space_after=2)
    _para(doc, "Draft assistance", size=11, color=_GREY, space_after=10)

    banner = doc.add_paragraph()
    banner.paragraph_format.space_after = Pt(12)
    _font(banner.add_run("⚠ " + NOTICE), size=10, bold=True, color=_RED)

    _rule(doc)

    # ---- the question ---------------------------------------------------
    _heading(doc, "QUESTION")
    _para(doc, query or "(none)", size=11, space_after=12)

    # ---- the draft ------------------------------------------------------
    _heading(doc, "DRAFT ANSWER")
    parts = draft.get("parts") or []
    if not parts:
        _para(doc, "(no draft was produced)", italic=True, color=_GREY)
    for p in parts:
        label = (p.get("label") or "").strip()
        text = (p.get("text") or "").strip()
        para = doc.add_paragraph()
        para.paragraph_format.space_after = Pt(3)
        if label:
            _font(para.add_run(f"{label} "), size=11, bold=True)
        _font(para.add_run(text), size=11, hindi=_is_hindi(text))

        cites = p.get("cites") or []
        cp = doc.add_paragraph()
        cp.paragraph_format.space_after = Pt(10)
        cp.paragraph_format.left_indent = Pt(18)
        if cites:
            _font(cp.add_run("Source: " + "; ".join(cites)), size=8.5, italic=True,
                  color=_GREY)
        else:
            # An uncited point is NOT quietly presented as if it were sourced.
            _font(cp.add_run("⚠ UNCITED — no past reply supports this. Verify before use."),
                  size=8.5, bold=True, color=_RED)

    # ---- key points -----------------------------------------------------
    kps = draft.get("key_points") or []
    if kps:
        _heading(doc, "KEY POINTS THE REPLY SHOULD COVER")
        for kp in kps:
            text = (kp.get("point") or "").strip()
            para = doc.add_paragraph(style="List Bullet")
            para.paragraph_format.space_after = Pt(2)
            _font(para.add_run(text), size=10.5, hindi=_is_hindi(text))
            cites = kp.get("cites") or []
            if cites:
                _font(para.add_run("  [" + "; ".join(cites) + "]"), size=8.5, italic=True,
                      color=_GREY)
            else:
                _font(para.add_run("  [UNCITED]"), size=8.5, bold=True, color=_RED)
        _para(doc, "", space_after=8)

    # ---- gaps -----------------------------------------------------------
    gaps = draft.get("gaps") or []
    if gaps:
        _heading(doc, "GAPS — OFFICER INPUT NEEDED")
        _para(doc, "The past replies do not cover the following. Nothing has been invented "
                   "for them.", size=9.5, italic=True, color=_GREY, space_after=4)
        for g in gaps:
            part = (g.get("part") or "").strip()
            reason = (g.get("reason") or "").strip()
            para = doc.add_paragraph(style="List Bullet")
            para.paragraph_format.space_after = Pt(2)
            if part:
                _font(para.add_run(f"{part}: "), size=10.5, bold=True, color=_RED)
            _font(para.add_run(reason), size=10.5, hindi=_is_hindi(reason))
        _para(doc, "", space_after=8)

    # ---- sources --------------------------------------------------------
    sources = draft.get("sources") or []
    if sources:
        _heading(doc, "SOURCES — THE PAST REPLIES THIS DRAFT WAS BUILT FROM")
        t = doc.add_table(rows=1, cols=5)
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.LEFT
        for i, h in enumerate(("Citation", "Session", "House", "Answered", "Type")):
            cell = t.rows[0].cells[i]
            cell.text = ""
            _font(cell.paragraphs[0].add_run(h), size=9, bold=True)
        for s in sources:
            row = t.add_row().cells
            vals = (s.get("citation") or "", s.get("session") or "", s.get("house") or "",
                    s.get("reply_date") or "—", s.get("answer_type") or "")
            for i, v in enumerate(vals):
                row[i].text = ""
                _font(row[i].paragraphs[0].add_run(str(v)), size=9)
        _para(doc, "", space_after=10)

    # ---- footer, on every page -----------------------------------------
    _rule(doc)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    meta = f"Generated {stamp}"
    if user_email:
        meta += f" for {user_email}"
    if run_id:
        meta += f" · retrieval run {run_id}"
    _para(doc, meta, size=8.5, color=_GREY, space_after=2)

    sec = doc.sections[0]
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _font(fp.add_run("DRAFT — for officer review. Not an approved reply."),
          size=8, bold=True, color=_RED)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _rule(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    _font(p.add_run("─" * 58), size=8, color=RGBColor(0xD0, 0xD0, 0xD0))


_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(query: str, run_id: str | None = None) -> str:
    """
    A filesystem- and header-safe filename.

    The query is officer-supplied, so it is hostile until sanitised: a newline in a
    Content-Disposition header is a response-splitting primitive, and a '/' or '..' would
    matter the moment anyone saved this by hand. Strip to a known-safe alphabet rather than
    blacklisting.

    A fully non-Latin query (Hindi) sanitises to NOTHING, which produced the useless
    "NHPC-draft-draft-<id>.docx". Rather than transliterate -- guessing at romanisation is
    its own bug -- we simply drop the stem and let the run id name the file. The document's
    CONTENT is Devanagari either way; only the filename is ASCII, because a Content-
    Disposition header with raw UTF-8 is not portable across browsers.
    """
    stem = _SAFE.sub("-", (query or "").strip())[:48]
    # Collapse dots too. '../../etc/passwd' otherwise survives as '..-..-etc-passwd': it is
    # only a download filename and we never open it, but a name carrying '..' is the kind
    # of thing that becomes a real bug the moment someone later uses it as a path.
    stem = re.sub(r"[.\-]{2,}", "-", stem).strip("-.")
    tail = _SAFE.sub("-", (run_id or dt.datetime.now().strftime("%Y%m%d-%H%M%S"))[:24])
    return f"NHPC-draft-{stem}-{tail}.docx" if stem else f"NHPC-draft-{tail}.docx"
