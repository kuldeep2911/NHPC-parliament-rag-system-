"""
The reply date: VALIDATION, and a rule-based FALLBACK.

WHO EXTRACTS THE DATE
    1. THE LLM, during the existing span-extraction call (see _SPAN_SYSTEM in extract.py).
       It already reads the whole document to find questions, answers and tables; the
       header date is one more field in the same JSON. It costs NO extra API call, and the
       model handles the cases a regex cannot: erratic PDF spacing, Hindi headers, unusual
       phrasings, and -- crucially -- telling the reply date apart from the dozen other
       dates in the document.
    2. THE RULES BELOW, only when the model returns nothing (or is not in use at all: the
       parser can run without an LLM backend, and a date is not worth failing a parse over).

⚠️ THE MODEL IS NEVER TRUSTED RAW. ⚠️

validate() is the single gate. Whatever the model says goes through exactly the same
parsing and the same plausibility checks as a regex match. A model that returns
'31.02.2024', '15.06.1823', or a confidently-wrong 2019 date for a 2025 session is
rejected here.

This is the same discipline as the span extractor: the model is trusted to FIND things, not
to hand us well-formed output. It was right about the spans and my post-processing was
wrong; the answer was not to trust it more, it was to verify with universal invariants and
reject rather than emit something wrong.

WHY A WRONG DATE IS WORSE THAN NO DATE. reply_date orders what the officer reads. A NULL is
shown as "date unknown" and sorted last -- visibly uncertain. A wrong date silently moves a
2019 reply to the top of a "most recent first" list, and nobody ever notices.
"""

from __future__ import annotations

import datetime as dt
import re

# The plausible window for a parliamentary reply date. Matches the DB CHECK in migration 011
# and the crawler's session-year range.
YEAR_MIN, YEAR_MAX = 2000, 2050

# How far from its own session a reply date may fall. A session runs a few months and a
# reply may be filed a little either side, so a year is generous. Anything further out is
# not a reply date -- it is a reference letter, an MOU, or a commissioning date.
#
# This is the CORROBORATION check and it is the last line of defence, for the model's output
# as much as the regex's. Measured on the real corpus: without it, 'dated 08.11.2019' was
# recorded as the reply date of a 2025 document.
MAX_YEARS_FROM_SESSION = 1

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

# \s* around every separator is not paranoia: the PDF text layer inserts spaces mid-token
# ("0 6 . 0 2 . 2 0 2 4"), so a strict \d{2}\.\d{2} misses real matches. Measured.
# \d{2,5}: a 5-digit year is a source-document TYPO ("03.02.02022"), repaired in
# _parse(). The regex has to match it before it can be repaired.
_NUMERIC = r"\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{2,5}"
_NAMED = r"\d{1,2}\s*(?:st|nd|rd|th)?\s*[-\s]\s*[A-Za-z]{3,9}\.?\s*,?\s*[-\s]?\s*\d{2,4}"
_ISO = r"\d{4}-\d{2}-\d{2}"
_DATE = rf"({_ISO}|{_NUMERIC}|{_NAMED})"

# ---------------------------------------------------------------------------
# THE FALLBACK ANCHORS
# ---------------------------------------------------------------------------
# A date only counts when it FOLLOWS a phrase meaning "this is when the question is
# answered". Never take the first bare date in the document: these replies are full of dates
# that are not the reply date ("NHPC signed a MOU with GEDCOL on 20.07.2020" -- in a 2022
# document). An unanchored date is ignored and the document is reported as a miss.
_ANCHORS = [
    ("answered_on",
     re.compile(r"(?:to\s+be\s+)?answer(?:ed)?\s+on\s*:?\s*" + _DATE, re.I | re.S)),
    ("for_answer_on",
     re.compile(r"for\s+answer\s+on\s*:?\s*" + _DATE, re.I | re.S)),
    # the trailing 'on the subject' is what makes this an anchor rather than a bare 'for N'
    ("for_date",
     re.compile(r"\bfor\s+" + _DATE + r"\s+on\s+the\s+subject", re.I | re.S)),
    ("hindi",
     re.compile(r"(?:दिनांक|दिनाक|दिनांकित|तारीख|तिथि)\s*:?\s*" + _DATE, re.I | re.S)),
    # weakest, and last: a covering letter carries its own date, which is NOT the reply date
    ("dated",
     re.compile(r"\bdated?\s*:?\s*" + _DATE, re.I | re.S)),
]


def _parse(raw: str, session_year: int | None = None):
    """A date string in any observed form -> date, or None if it is not a plausible one."""
    s = re.sub(r"\s+", "", str(raw or ""))
    if not s:
        return None

    # TYPOS IN THE SOURCE DOCUMENT. A real header reads "for answer on 03.02.02022" -- the
    # year is typed with a stray leading zero. This is not ambiguous and it is not a guess:
    # 02022 can only be 2022. Rejecting it loses a date that is plainly there.
    #
    # Only a LEADING zero on a 5-digit year is repaired. Anything else stays rejected --
    # we are correcting an obvious slip, not inventing a plausible number.
    s = re.sub(r"(?<![\d])0(\d{4})(?![\d])", r"\1", s)

    # ISO first -- that is what we ask the model for.
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        two_digit = False
    else:
        m = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})", s)
        if m:
            d, mo, y = (int(x) for x in m.groups())
            two_digit = len(m.group(3)) <= 2
        else:
            m = re.fullmatch(
                r"(\d{1,2})(?:st|nd|rd|th)?[-,]?([A-Za-z]{3,9})\.?[-,]?(\d{2,4})", s, re.I)
            if not m:
                return None
            d = int(m.group(1))
            mo = _MONTHS.get(m.group(2).lower().rstrip("."))
            y = int(m.group(3))
            two_digit = len(m.group(3)) <= 2
            if not mo:
                return None

    y = _resolve_year(y, two_digit, session_year)
    if y is None:
        return None

    # DD/MM vs MM/DD. Indian government documents are DD.MM.YYYY, always. Swap only when the
    # first field CANNOT be a day.
    if d > 31 or mo > 12:
        if mo <= 31 and d <= 12:
            d, mo = mo, d
        else:
            return None

    if not (YEAR_MIN <= y <= YEAR_MAX):
        return None
    try:
        return dt.date(y, mo, d)
    except ValueError:
        return None                       # 31.02.2024 and friends


def _resolve_year(y: int, two_digit: bool, session_year: int | None):
    """
    Turn a possibly-truncated year into a real one.

    THE TRAP: the PDF text layer drops characters. 'to be answered on 28.11.2024' arrives as
    '28.11.20', and reading 20 -> 2020 puts a 2024 reply FOUR YEARS in the past. The session
    year disambiguates it: a reply is answered during its session.
    """
    if not two_digit:
        return y
    if session_year is None:
        return y + 2000 if y < 100 else y
    # A 2-digit token is part of the real year. Prefer the session (or the next year, for a
    # nov-dec session answering into January) over a naive 2000+y.
    for cand in (session_year, session_year + 1):
        if y == cand % 100 or y == cand // 100:
            return cand
    return 2000 + y


def validate(value, session_year: int | None = None):
    """
    THE SINGLE GATE. Everything -- the LLM's answer and the regex's -- passes through here.

    Returns a date, or None. `value` may be a date, a datetime, or a string in any of the
    observed forms.

    A model that hallucinates a well-formed but WRONG date (a 2019 date on a 2025 document)
    is rejected by the session corroboration, not merely by the calendar.
    """
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        value = value.date()
    if isinstance(value, dt.date):
        d = value
        if not (YEAR_MIN <= d.year <= YEAR_MAX):
            return None
    else:
        d = _parse(value, session_year=session_year)
        if d is None:
            return None
    if session_year and abs(d.year - session_year) > MAX_YEARS_FROM_SESSION:
        return None                       # implausible for this session -> reject, not guess
    return d


def extract_from_text(*texts, session_year: int | None = None):
    """
    THE FALLBACK. Anchored regex over the given texts, searched in order (header first).

    Used only when the LLM returned no date, or when no LLM backend is configured. Returns a
    date or None; None is a reported miss, never a silent default.
    """
    for text in texts:
        if not text:
            continue
        for _name, rx in _ANCHORS:
            for m in rx.finditer(text):
                d = validate(m.group(1), session_year=session_year)
                if d is not None:
                    return d
    return None


def pdf_header_text(path: str, pages: int = 2) -> str:
    """
    The text layer of the first `pages` pages. No Docling, no OCR -- the cheap path, used by
    the backfill only. Returns "" for a scanned PDF with no text layer.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        rdr = PdfReader(path)
        return "\n".join((rdr.pages[i].extract_text() or "")
                         for i in range(min(pages, len(rdr.pages))))
    except Exception:      # noqa: BLE001 -- an unreadable PDF is a miss, never a crash
        return ""
