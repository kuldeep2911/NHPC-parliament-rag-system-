
"""
NHPC Parliament Data — Phase 1: read-only crawler & reorganizer.

Walks a messy source tree of parliamentary Q/A records and COPIES a clean,
canonical structure into an `organized/` directory. It NEVER modifies, moves,
deletes, or renames anything in the source. It does NOT parse file *content*;
this phase is purely about file/folder organization.

Canonical output layout:
    organized/<session>/<house>[/<state>]/<question_id>/
        question_original.<ext>        (question file, if identified)
        answer_latest.<ext>            (selected latest reply)
        answer_all_versions/<untouched original filenames>
        metadata.json

Reports:
    organized/_reports/report.json
    organized/_reports/report.csv
    organized/_reports/orphans.csv

Usage:
    python organize_parliament.py [--source SRC] [--out organized] [--dry-run]

Rules confirmed with stakeholder (see module docstring / README):
  * Session token = normalized month-range slug (e.g. 2025-nov-dec), not season.
    Exception: "MONSOON 25" -> 2025-monsoon.
  * Cross-year winter sessions -> best-effort START year + flag.
  * Duplicate question IDs (category-nested vs flat; base vs "(Revised)") are
    MERGED into one question folder; all source paths recorded + flagged.
  * Assembly/Assemble question -> vidhan_sabha. Vidhan Sabha may nest a state.
  * Word locks (~$), Thumbs.db, desktop.ini, *.tmp, *.lnk, *.download excluded.
  * Note*/Annexure* are copied but never SELECTED as the answer.
  * Zips are copied and flagged, never silently skipped.
  * Dates in filenames are treated as question DUE dates, NOT version dates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration / vocabulary discovered during exploration
# ---------------------------------------------------------------------------

DEFAULT_SOURCE = "."          # run from the data root
DEFAULT_OUT = "organized"


class LockedOutputError(Exception):
    """An output folder could not be cleared because a file was locked (WinError 32)."""

# Top-level directories that are never source session folders.
#
# supporting_documents is here for defence in depth: it lives under organized/, and the
# crawler only ever writes session/house/question paths, so it is already out of reach --
# but naming it explicitly means that if anyone ever points the crawler at organized/ (or
# nests the trees), a financial report can never be mistaken for a parliamentary session.
SKIP_TOP_LEVEL = {
    "organized", "_reports", "_supporting", "supporting_documents", "__pycache__",
    ".git", ".venv", "venv", ".idea", ".vscode",
}

# Files that are never real content — pure noise / OS / lock artifacts.
NOISE_EXACT = {"thumbs.db", "desktop.ini", ".ds_store"}
NOISE_EXTS = {".tmp", ".lnk", ".download", ".db", ".ini"}
LOCKFILE_PREFIX = "~$"        # Word/Excel lock files
LOCKFILE_TMP = re.compile(r"^~wrl\d+\.tmp$", re.IGNORECASE)

ARCHIVE_EXTS = {".zip", ".rar", ".7z"}

# Filenames that are attachments/notes, not the answer itself.
NON_ANSWER_PATTERNS = re.compile(
    r"^(note|annexure|annex|binder|input planning|om\b|e-?mail|letter|~\$)",
    re.IGNORECASE,
)

# Filename NAME-RANK signals used to break the no-version/mtime tie so the actual
# reply is preferred over cover letters, routing notes, and person-named docs.
# Positive = looks like the real reply; negative = looks like a cover/routing doc.
REPLY_NAME_POSITIVE = re.compile(
    r"\b(reply|reply\s*with|final\s*reply|approved\s*reply|answer|response|"
    r"draft\s*reply|reply\s*to)\b",
    re.IGNORECASE,
)
REPLY_NAME_NEGATIVE = re.compile(
    r"\b(covering\s*letter|cover\s*letter|forwarding|routing|note|noting|"
    r"input\s*planning|briefing|minutes|agenda|list of|index|"
    r"do\s*letter|d\.?o\.?\s*letter|email|e-?mail|corrigendum)\b",
    re.IGNORECASE,
)
# A filename that is essentially just a person's name (e.g. "surjit shaw.doc",
# "Kundan Kumar.doc", "24 GAUTAM PALIT.doc") — 1-4 alphabetic words, no reply/
# question/annexure signal. Such docs are cover/routing artifacts, not the reply.
_PERSON_NAME_TOKENS = re.compile(r"^[A-Za-z][A-Za-z.'-]*$")

# House-folder vocabulary (lowercased, stripped). Anything not here at the
# house level is treated as an orphan/supporting folder.
HOUSE_MAP = {
    "lok sabha": "lok_sabha",
    "loksabha": "lok_sabha",
    "rajya sabha": "rajya_sabha",
    "rajyasabha": "rajya_sabha",
    "vidhan sabha": "vidhan_sabha",
    "vidhansabha": "vidhan_sabha",
    "assembly question": "vidhan_sabha",
    "assemble question": "vidhan_sabha",
    "assembly": "vidhan_sabha",
}

# ---------------------------------------------------------------------------
# LEGACY LAYOUT (roughly 2014-2017 sessions).
#
# Those sessions have NO house folder. The house is encoded in the question
# folder's own name instead, e.g.
#     LSQ 1999, 26.02.15      -> lok_sabha,   diary 1999
#     RSQ S 2524 19.03.15     -> rajya_sabha, diary S2524 (starred)
#     ASQ 12-8-1197, 17.4.15  -> vidhan_sabha (assembly)
#     AQ 125                  -> vidhan_sabha
#     Assurance LSQ 1867 …    -> lok_sabha (an assurance follow-up)
# The folders may sit directly under the session or under a wrapper such as
# "Due up to FEB- MAR 15" / "Assembly Question" / "Assurance".
#
# QUESTION_PREFIX_HOUSE maps the leading token to a house so these folders can be
# classified without a house directory. Order matters: longest/most specific first.
# ---------------------------------------------------------------------------
QUESTION_PREFIX_HOUSE = [
    (re.compile(r"^\s*assurance\s*[-–:]?\s*ls\s*q", re.I), "lok_sabha"),
    (re.compile(r"^\s*assurance\s*[-–:]?\s*rs\s*q", re.I), "rajya_sabha"),
    (re.compile(r"^\s*ls\s*q", re.I), "lok_sabha"),      # LSQ, LS Q
    (re.compile(r"^\s*rs\s*q", re.I), "rajya_sabha"),    # RSQ, RS Q
    (re.compile(r"^\s*as\s*q", re.I), "vidhan_sabha"),   # ASQ  (assembly)
    (re.compile(r"^\s*a\s*q\b", re.I), "vidhan_sabha"),  # AQ
    (re.compile(r"^\s*usq", re.I), "lok_sabha"),         # unstarred LS question
]


def house_from_question_name(name: str):
    """
    House implied by a legacy question-folder name ("LSQ 1999, 26.02.15" -> lok_sabha),
    or None when the name carries no house prefix. Used only where no house folder exists.
    """
    n = norm_ws(name)
    for pat, house in QUESTION_PREFIX_HOUSE:
        if pat.match(n):
            return house
    return None

# Category / grouping folders that wrap question folders — pass through them.
CATEGORY_FOLDERS = {
    "csr", "generation", "land", "safety", "tariff", "hydroprojects",
    "hydro projects", "re", "environment", "finance", "hr", "onm", "o&m",
    "planning", "dmp",
}

# Reply-folder names (any case).
REPLY_FOLDER_NAMES = {"reply", "replies"}

# Version subfolders inside a reply, ranked (higher rank == more authoritative).
VERSION_SUBFOLDER_RANK = [
    (re.compile(r"approved", re.I), 40),
    (re.compile(r"final", re.I), 30),
    (re.compile(r"revised", re.I), 20),
    (re.compile(r"prev|previous|old|draft", re.I), -10),
]

# Version tokens inside a filename, ranked.
VERSION_TOKEN_RANK = [
    (re.compile(r"\bapproved\b", re.I), 40),
    (re.compile(r"\bfinal\b", re.I), 30),
    (re.compile(r"revised\s*(\d+)?", re.I), 20),
    (re.compile(r"\breply\s*(\d+)\b", re.I), None),   # reply1, reply2 -> numeric
    (re.compile(r"\bv(\d+)\b", re.I), None),          # v1, v2 -> numeric
    (re.compile(r"\br(\d+)\b", re.I), None),          # r0, r1 -> numeric
    (re.compile(r"\bdraft\b", re.I), -10),
]

MONTHS = {
    "jan": "jan", "january": "jan",
    "feb": "feb", "february": "feb",
    "mar": "mar", "march": "mar",
    "apr": "apr", "april": "apr",
    "may": "may",
    "jun": "jun", "june": "jun",
    "jul": "jul", "july": "jul",
    "aug": "aug", "august": "aug",
    "sep": "sep", "sept": "sep", "september": "sep",
    "oct": "oct", "october": "oct",
    "nov": "nov", "november": "nov",
    "dec": "dec", "december": "dec",
}

# Known Vidhan-Sabha state folder names (lowercased). Used to decide whether an
# extra nesting level is a state vs a question folder.
STATE_HINTS = {
    "himachal pradesh", "hp", "j&k", "jammu and kashmir", "jammu & kashmir",
    "assam", "uttarakhand", "west bengal", "sikkim", "manipur", "odisha",
    "arunachal pradesh", "madhya pradesh",
}

# Chamber-code prefixes on question-folder names to strip for the canonical id.
# Prefixes stripped from a question-folder name to leave the bare diary number.
# LSQ/RSQ/ASQ/AQ come from the legacy 2014-17 layout and must be matched BEFORE the
# shorter ls/rs/s/u forms, or "LSQ 1999" strips only "LS" and leaves "Q 1999".
QID_PREFIX = re.compile(
    r"^(assurance[\s\.\-]*)?(lsq|rsq|asq|usq|aq|sq|dy\.?|ls|rs|s|u|us)[\s\.\-]*",
    re.IGNORECASE)
# A trailing date in the folder name ("LSQ 1999, 26.02.15", "RSQ 1689 13.03.15").
# It identifies the sitting, not the question, so it is dropped from the id.
QID_TRAILING_DATE = re.compile(
    r"[\s,]*\d{1,2}[\.\-/]\d{1,2}[\.\-/]\d{2,4}\s*$")
REVISED_IN_NAME = re.compile(r"\(?\s*revised[^)]*\)?", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Report:
    started: str = ""
    source_root: str = ""
    out_root: str = ""
    dry_run: bool = False
    session_folders_seen: int = 0
    question_folders_seen: int = 0
    organized_ok: int = 0
    needs_review: int = 0
    review_reasons: Counter = field(default_factory=Counter)
    orphans: list = field(default_factory=list)      # supporting/unmatched
    questions: list = field(default_factory=list)    # per-question metadata


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _skip_top_level(d: Path, out_root: Path) -> bool:
    """True if a top-level directory is not a source session folder."""
    return (
        d.name.lower() in SKIP_TOP_LEVEL
        or d.name.startswith(".")
        or d.name.startswith("__")
        or d.resolve() == out_root.resolve()
    )


# The plausible range for a parliamentary session year. THE SINGLE SOURCE OF TRUTH --
# every error message that quotes these bounds reads them from here, so the rule and what
# we tell the user can never drift apart. (They already had: the range lived here and was
# restated as a literal in three other files.)
#
# The bound is not decoration. A folder name is full of digits that are NOT years -- diary
# numbers, capacities, dates -- and without a range 'PARLIAMENT DEC-JAN 1915' would parse
# as the year 1915 and produce the session slug '1915-dec-jan'. A range keeps a number
# that cannot be a session year from silently becoming one.
#
# 2000..2050 is deliberately generous: it costs nothing to accept an old digitised session,
# and the point of the check is only to reject the absurd.
SESSION_YEAR_MIN = 2000
SESSION_YEAR_MAX = 2050


def normalize_session(name: str):
    """
    Return (session_slug, review_reason_or_None).

    Strategy: extract year(s) and month tokens from the folder label and build
    a `YYYY-mon[-mon]` slug. Special-case the season word 'monsoon'.
    Cross-year ranges resolve to the START year and are flagged.
    """
    low = name.lower()
    low = re.sub(r"^parliament\s*", "", low).strip()

    # Collect year tokens (2- or 4-digit). Interpret 2-digit as 20YY.
    years = []
    for yy in re.findall(r"(?<!\d)((?:19|20)?\d{2})(?!\d)", name):
        yy_i = int(yy)
        years.append(2000 + yy_i if yy_i < 100 else yy_i)
    years = [y for y in years if SESSION_YEAR_MIN <= y <= SESSION_YEAR_MAX]

    months = []
    for tok in re.findall(r"[a-zA-Z]+", low):
        if tok in MONTHS and MONTHS[tok] not in months:
            months.append(MONTHS[tok])

    has_monsoon = "monsoon" in low

    reason = None
    # Determine the canonical year.
    if not years:
        return None, "unrecognized_session_no_year"
    start_year = min(years)
    cross_year = len(set(years)) > 1
    if cross_year:
        reason = "ambiguous_session_year"

    if has_monsoon and not months:
        slug = f"{start_year}-monsoon"
    elif months:
        # keep at most the first two month tokens, in the order they appeared
        slug = f"{start_year}-" + "-".join(months[:2])
    else:
        # a year but no month/season signal
        slug = f"{start_year}-unknown"
        reason = reason or "session_month_unrecognized"

    return slug, reason


def normalize_house(name: str):
    """Return (house_slug_or_None, is_category, is_orphan)."""
    key = norm_ws(name).lower()
    if key in HOUSE_MAP:
        return HOUSE_MAP[key], False, False
    if key in CATEGORY_FOLDERS:
        return None, True, False
    return None, False, True


def looks_like_state(name: str) -> bool:
    key = norm_ws(name).lower()
    if key in STATE_HINTS:
        return True
    # a state folder is non-numeric and not a reply/category folder
    if re.search(r"\d", name):
        return False
    if key in REPLY_FOLDER_NAMES or key in CATEGORY_FOLDERS:
        return False
    # heuristic: alphabetic word(s), title-ish
    return bool(re.match(r"^[A-Za-z&\.\s]+$", name)) and len(key) > 1


def canonical_qid(folder_name: str):
    """
    Derive a canonical question id from a question-folder name.
    Returns (qid, base_qid, flags:set). base_qid strips revision markers so
    '5880' and '5880 (Revised l)' merge to the same base.
    """
    flags = set()
    name = norm_ws(folder_name)

    if REVISED_IN_NAME.search(name):
        flags.add("possible_revised_duplicate")

    # Legacy folders carry the sitting date after the diary number ("LSQ 1999, 26.02.15").
    # Strip it first: it is not part of the question id, and it would otherwise look like a
    # multi-question range or a vidhan-style date id. Only when something precedes it, so a
    # genuinely date-named folder is left alone.
    _no_date = QID_TRAILING_DATE.sub("", name).strip(" ,.-")
    if _no_date and _no_date != name and re.search(r"\d", _no_date):
        name = _no_date

    # Ranges / multi-question, e.g. "1894 - 1908", "S2542,S2544". A vidhan-style date id
    # ("12-8-1197", with or without a legacy ASQ/AQ prefix) is NOT a range, so test the
    # prefix-stripped form too before flagging.
    _bare = QID_PREFIX.sub("", name).strip(" ()")
    if (re.search(r"\d\s*[-,]\s*\d", name)
            and not re.match(r"^\d{1,2}[\s\.\-]\d{1,2}[\s\.\-]\d+$", name)
            and not re.match(r"^\d{1,2}[\s\.\-]+\d{1,2}[\s\.\-]+\d+$", _bare)):
        flags.add("multi_question")

    stripped = REVISED_IN_NAME.sub("", name).strip(" ()")

    # Vidhan-Sabha date-style IDs like "14 08 1039", "14-8-128", "13.16.1115".
    # These are NOT simple numeric IDs: the leading numbers are a date/session
    # code, so a leading-number base would wrongly merge distinct questions
    # (14-8-1039 and 14-8-128 both -> "14"). Keep the whole token as the base.
    # Test the prefix-stripped form too, so a legacy "ASQ 12-8-1197" is recognised as the
    # same kind of assembly date-id as a bare "12-8-1197".
    _date_src = stripped
    _pre = QID_PREFIX.sub("", stripped).strip(" ()")
    if re.match(r"^\s*\d{1,2}[\s\.\-]+\d{1,2}[\s\.\-]+\d+\s*$", _pre):
        _date_src = _pre
    date_id = re.match(r"^\s*\d{1,2}[\s\.\-]+\d{1,2}[\s\.\-]+\d+\s*$", _date_src)
    if date_id:
        norm = re.sub(r"[\s\.\-]+", "-", _date_src.strip())
        qid = norm
        base = norm.lower()
        return qid, base, flags

    # Detect a chamber-TYPE marker (Starred vs Unstarred) before stripping the
    # generic prefix. S-4961 and U-4961 are DIFFERENT questions that share a
    # diary number, so we keep the s/u marker in the base to avoid merging them.
    # Bare "S1234"/"U1234" and spaced "S 1234" both count; "Dy"/"LS"/"RS" don't.
    chamber = ""
    cm = re.match(r"^\s*(s|u|us|sq|usq)[\s\.\-]*\d", stripped, re.IGNORECASE)
    if cm:
        chamber = cm.group(1)[0].lower()  # 's' or 'u'

    stripped = QID_PREFIX.sub("", stripped).strip(" ()")
    stripped = norm_ws(stripped)

    if not stripped:
        stripped = name

    # base id = leading numeric run (plus any trailing suffix letter, e.g. the
    # "A" in "5240A", which marks a distinct/clubbed question) if present.
    m = re.match(r"^(\d{2,}[A-Za-z]?)", stripped)
    if m:
        base = m.group(1).lower()
        if chamber:
            base = f"{chamber}-{base}"
    else:
        base = stripped.lower()

    qid = re.sub(r"[^\w\-]+", "_", stripped).strip("_") or "unknown"
    return qid, base, flags


def path_hash(rel: str) -> str:
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]


# ---------------------------------------------------------------------------
# File classification
# ---------------------------------------------------------------------------

def is_noise(fname: str) -> bool:
    low = fname.lower()
    if fname.startswith(LOCKFILE_PREFIX):
        return True
    if LOCKFILE_TMP.match(fname):
        return True
    if low in NOISE_EXACT:
        return True
    ext = os.path.splitext(low)[1]
    if ext in NOISE_EXTS:
        return True
    return False


def is_archive(fname: str) -> bool:
    return os.path.splitext(fname.lower())[1] in ARCHIVE_EXTS


def is_selectable_answer(fname: str) -> bool:
    """A candidate that could be THE answer (not a note/annexure/archive)."""
    if NON_ANSWER_PATTERNS.match(fname.strip()):
        return False
    if is_archive(fname):
        return False
    ext = os.path.splitext(fname.lower())[1]
    return ext in {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".rtf", ".txt",
                   ".oxps", ".xps"}


# ---------------------------------------------------------------------------
# Reply selection
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    path: Path
    name: str
    mtime: float
    subfolder_rank: int
    version_rank: float
    version_num: int
    has_version_token: bool
    name_rank: int = 0


def name_score(fname: str, qid: str = "") -> int:
    """
    Heuristic rank for how much a filename looks like THE reply, used to break the
    no-version tie before falling back to mtime. Higher == more likely the reply.

      +3  contains an explicit reply/answer word
      +2  contains the question id/number
      -3  looks like a cover letter / routing / note / DO letter / email
      -2  filename is essentially just a person's name (cover/routing artifact)
    """
    stem = os.path.splitext(fname)[0].strip()
    low = stem.lower()
    score = 0
    if REPLY_NAME_POSITIVE.search(low):
        score += 3
    if REPLY_NAME_NEGATIVE.search(low):
        score -= 3
    # question-id / number present in the name (e.g. "Reply 5880", "S2542 ...")
    digits = re.sub(r"[^\d]", "", qid)
    if digits and len(digits) >= 3 and digits in re.sub(r"[^\d]", "", stem):
        score += 2
    # person-name-only heuristic: all tokens alphabetic, 1-4 of them, and no
    # reply/answer/question/annexure signal and no digits.
    tokens = [t for t in re.split(r"\s+", stem) if t]
    alpha_tokens = [t for t in tokens if _PERSON_NAME_TOKENS.match(t)]
    if (tokens and len(alpha_tokens) == len(tokens) and 1 <= len(tokens) <= 4
            and not REPLY_NAME_POSITIVE.search(low)
            and not re.search(r"\b(question|annexure|annex|note|report)\b", low)):
        score -= 2
    return score


def version_score(fname: str):
    """Return (rank, numeric, matched_token_bool)."""
    best_rank = None
    best_num = -1
    matched = False
    for pat, rank in VERSION_TOKEN_RANK:
        m = pat.search(fname)
        if not m:
            continue
        matched = True
        if rank is None:  # numeric token like reply1 / v2 / r0
            num = int(m.group(1)) if m.group(1) else 0
            best_num = max(best_num, num)
            best_rank = max(best_rank if best_rank is not None else 0, 10)
        else:
            best_rank = rank if best_rank is None else max(best_rank, rank)
    return (best_rank if best_rank is not None else 0), best_num, matched


def subfolder_score(rel_within_reply: str):
    score = 0
    for pat, rank in VERSION_SUBFOLDER_RANK:
        if pat.search(rel_within_reply):
            score += rank
    return score


def collect_reply_candidates(reply_dir: Path, qid: str = ""):
    """
    Walk a reply folder recursively, returning (candidates, all_files, archives,
    excluded_noise). Candidates are selectable answer files only.
    """
    candidates = []
    all_files = []
    archives = []
    noise = []
    for root, _dirs, files in os.walk(reply_dir):
        for f in files:
            p = Path(root) / f
            rel_within = os.path.relpath(p, reply_dir)
            if is_noise(f):
                noise.append(rel_within)
                continue
            all_files.append(rel_within)
            if is_archive(f):
                archives.append(rel_within)
                continue
            if not is_selectable_answer(f):
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                mtime = 0.0
            vrank, vnum, vmatched = version_score(f)
            candidates.append(Candidate(
                path=p, name=f, mtime=mtime,
                subfolder_rank=subfolder_score(os.path.dirname(rel_within)),
                version_rank=vrank, version_num=vnum,
                has_version_token=vmatched,
                name_rank=name_score(f, qid),
            ))
    return candidates, all_files, archives, noise


def collect_loose_reply_candidates(qdir: Path, qfile, qid: str = ""):
    """
    When a question folder has NO reply/ subfolder, treat its top-level document
    files as reply candidates. Only the immediate directory is scanned (nested
    supporting folders like 'info received' are left out of candidate selection,
    but everything is still recorded in all_files so nothing is lost).
    The identified question file is excluded from candidates.
    """
    candidates = []
    all_files = []
    archives = []
    noise = []
    qfile_low = qfile.lower() if qfile else None
    for entry in sorted(qdir.iterdir()):
        if not entry.is_file():
            continue
        f = entry.name
        if is_noise(f):
            noise.append(f)
            continue
        all_files.append(f)
        if is_archive(f):
            archives.append(f)
            continue
        if qfile_low and f.lower() == qfile_low:
            continue  # that's the question, not the answer
        if not is_selectable_answer(f):
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            mtime = 0.0
        vrank, vnum, vmatched = version_score(f)
        candidates.append(Candidate(
            path=entry, name=f, mtime=mtime,
            subfolder_rank=0, version_rank=vrank, version_num=vnum,
            has_version_token=vmatched,
            name_rank=name_score(f, qid),
        ))
    return candidates, all_files, archives, noise


def select_latest(candidates):
    """
    Return (selected:list[Candidate], reason, ambiguous:bool).
    Priority: version-subfolder rank -> version token -> mtime.
    Ties at the top -> return ALL tied candidates + ambiguous flag.
    """
    if not candidates:
        return [], "no_reply_candidates", False
    if len(candidates) == 1:
        return candidates, "single_candidate", False

    any_subfolder = any(c.subfolder_rank != 0 for c in candidates)
    any_version = any(c.has_version_token for c in candidates)
    any_name_signal = any(c.name_rank != 0 for c in candidates)

    if any_subfolder:
        reason = "version_parsed"
        key = lambda c: (c.subfolder_rank, c.version_rank, c.version_num,
                         c.name_rank, c.mtime)
    elif any_version:
        reason = "version_parsed"
        key = lambda c: (c.version_rank, c.version_num, c.name_rank, c.mtime)
    elif any_name_signal:
        # No explicit version signal, but filenames tell us which is the reply vs
        # a cover letter / routing note / person-named doc. Prefer by name, then
        # mtime. This is what fixes 'Covering letter.doc'/'Kundan Kumar.doc' wins.
        reason = "name_preference"
        key = lambda c: (c.name_rank, c.mtime)
    else:
        reason = "mtime_fallback"
        key = lambda c: (c.mtime,)

    best = max(candidates, key=key)
    best_key = key(best)
    # Ties: same top key. For mtime we require exact equality to be a tie;
    # otherwise prefer pdf over its docx sibling of the same stem.
    tied = [c for c in candidates if key(c) == best_key]

    if len(tied) > 1:
        # If tied files are just format siblings (same stem), prefer .pdf and
        # do NOT treat as ambiguous.
        stems = {os.path.splitext(c.name)[0].strip().lower() for c in tied}
        if len(stems) == 1:
            return [_prefer_pdf(tied)], reason, False
        return tied, "ambiguous", True

    # A same-stem PDF sibling of the winner (e.g. "Reply.pdf" next to "Reply.docx")
    # is the better artifact for downstream text extraction even if its mtime is
    # a few seconds older, so promote it.
    best_stem = os.path.splitext(best.name)[0].strip().lower()
    siblings = [c for c in candidates
                if os.path.splitext(c.name)[0].strip().lower() == best_stem]
    return [_prefer_pdf(siblings) if len(siblings) > 1 else best], reason, False


def _prefer_pdf(cands):
    pdfs = [c for c in cands if c.name.lower().endswith(".pdf")]
    return pdfs[0] if pdfs else cands[0]


# ---------------------------------------------------------------------------
# Question-folder discovery
# ---------------------------------------------------------------------------

def find_question_file(qdir: Path, qid: str):
    """
    Pick the most likely 'question' file directly in the question folder
    (not inside reply/). Heuristic: a pdf/docx whose name references the id or
    contains question-ish tokens; else the first top-level pdf.
    """
    top_files = []
    for f in os.listdir(qdir):
        p = qdir / f
        if p.is_file() and not is_noise(f) and not is_archive(f):
            top_files.append(f)
    if not top_files:
        return None
    id_num = re.search(r"\d+", qid)
    id_num = id_num.group(0) if id_num else None
    scored = []
    for f in top_files:
        low = f.lower()
        s = 0
        if id_num and id_num in f:
            s += 5
        if re.search(r"question|usq|starred|unstarred|dy\.?\s*no|diary", low):
            s += 3
        if low.endswith(".pdf"):
            s += 1
        scored.append((s, f))
    scored.sort(reverse=True)
    return scored[0][1] if scored else None


def is_question_folder(d: Path) -> bool:
    """
    A question folder is a leaf-ish folder that either has a reply subfolder or
    contains document files directly (and is not itself a reply/category node).
    """
    name = norm_ws(d.name).lower()
    if name in REPLY_FOLDER_NAMES:
        return False
    for child in d.iterdir():
        if child.is_dir() and norm_ws(child.name).lower() in REPLY_FOLDER_NAMES:
            return True
    # else: has document files directly?
    for child in d.iterdir():
        if child.is_file() and is_selectable_answer(child.name):
            return True
    return False


def find_reply_dir(qdir: Path):
    for child in qdir.iterdir():
        if child.is_dir() and norm_ws(child.name).lower() in REPLY_FOLDER_NAMES:
            return child
    return None


# ---------------------------------------------------------------------------
# Copy helpers (idempotent)
# ---------------------------------------------------------------------------

def safe_copy(src: Path, dst: Path, dry_run: bool):
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def reset_question_output(qout: Path, dry_run: bool):
    """
    Idempotency: clear just this question's output folder before rewriting.

    On Windows a file in the tree may be transiently locked (editor preview,
    indexer, antivirus). Retry with backoff; if it stays locked, raise
    LockedOutputError so the caller can skip THIS folder and continue the crawl
    rather than aborting the whole run and leaving the tree half-regenerated.
    """
    if dry_run:
        return
    if qout.exists():
        last_err = None
        for attempt in range(4):
            try:
                shutil.rmtree(qout)
                break
            except PermissionError as e:  # WinError 32 — file in use
                last_err = e
                time.sleep(0.5 * (attempt + 1))
        else:
            raise LockedOutputError(f"{qout} is locked: {last_err}")
    qout.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main crawl
# ---------------------------------------------------------------------------

def process_question(qdir: Path, ctx: dict, merged_group: dict,
                     report: Report, out_root: Path, source_root: Path,
                     dry_run: bool):
    """
    Build metadata + copy files for one physical question folder into the
    (possibly shared) canonical output folder. Supports merging: called once
    per source folder; accumulates into merged_group keyed by output path.
    """
    session = ctx["session"]
    house = ctx["house"]
    state = ctx.get("state")
    orig_name = qdir.name
    rel_src = os.path.relpath(qdir, source_root)

    qid, base, flags = canonical_qid(orig_name)
    if qid == "unknown" or not re.search(r"\w", qid):
        qid = "gen_" + path_hash(rel_src)
        flags.add("generated_question_id")

    parts = [session, house]
    if state:
        parts.append(re.sub(r"[^\w\-]+", "_", state).strip("_").lower())
    # merge key uses the BASE id so revised/nested duplicates collapse
    parts.append(base)
    out_key = "/".join(parts)

    qfile = find_question_file(qdir, qid)

    reply_dir = find_reply_dir(qdir)
    if reply_dir:
        candidates, all_reply_files, archives, noise = collect_reply_candidates(
            reply_dir, qid)
        reply_base = reply_dir
    else:
        # No reply/ subfolder: treat selectable docs sitting directly in the
        # question folder as reply candidates (common pattern, e.g. "Reply 520.pdf"
        # next to the question). Exclude the identified question file itself.
        candidates, all_reply_files, archives, noise = collect_loose_reply_candidates(
            qdir, qfile, qid)
        reply_base = qdir

    grp = merged_group.setdefault(out_key, {
        "session": session, "house": house, "state": state,
        "question_id": base,
        "original_folder_names": [],
        "source_paths": [],
        "flags": set(),
        "candidates": [],          # list[Candidate]
        "all_reply_files": [],     # (src_rel, filename)
        "archives": [],
        "question_files": [],      # (src Path, filename)
    })
    grp["original_folder_names"].append(orig_name)
    grp["source_paths"].append(rel_src)
    grp["flags"] |= flags
    grp["candidates"].extend(candidates)
    grp["archives"].extend(os.path.join(rel_src, a) for a in archives)
    for rel_within in all_reply_files:
        grp["all_reply_files"].append((reply_base / rel_within, rel_within))
    if qfile:
        grp["question_files"].append((qdir / qfile, qfile))


def finalize_group(out_key: str, grp: dict, report: Report, out_root: Path,
                   dry_run: bool):
    qout = out_root / out_key
    reset_question_output(qout, dry_run)

    flags = set(grp["flags"])
    if len(grp["source_paths"]) > 1:
        flags.add("merged_from_multiple_sources")
    if grp["archives"]:
        flags.add("contains_archive")

    selected, reason, ambiguous = select_latest(grp["candidates"])
    if ambiguous:
        flags.add("ambiguous_latest_reply")
    if not selected:
        flags.add("no_reply_selected")

    # --- copy question original(s) ---
    question_file_out = None
    if grp["question_files"]:
        # prefer the highest-scoring; if several sources, keep first, others archived
        src, fname = grp["question_files"][0]
        ext = os.path.splitext(fname)[1]
        question_file_out = f"question_original{ext}"
        safe_copy(src, qout / question_file_out, dry_run)
        for i, (s2, fn2) in enumerate(grp["question_files"][1:], start=2):
            safe_copy(s2, qout / "question_other_versions" / fn2, dry_run)
    else:
        flags.add("no_question_file_found")

    # --- copy selected answer(s) ---
    answer_selected_names = []
    if selected:
        if len(selected) == 1:
            c = selected[0]
            ext = os.path.splitext(c.name)[1]
            safe_copy(c.path, qout / f"answer_latest{ext}", dry_run)
            answer_selected_names.append(c.name)
        else:
            # ambiguous: copy all tied candidates, don't pick one
            for c in selected:
                safe_copy(c.path, qout / "answer_latest_candidates" / c.name,
                          dry_run)
                answer_selected_names.append(c.name)

    # --- copy ALL reply versions untouched (dedupe by relative name) ---
    seen = set()
    for src, rel_within in grp["all_reply_files"]:
        dest_rel = rel_within
        base_dest = qout / "answer_all_versions" / dest_rel
        key = str(base_dest).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            safe_copy(src, base_dest, dry_run)
        except OSError:
            flags.add("copy_error")

    # (archives are already copied via all_reply_files and flagged above.)

    review_flags = sorted(flags)
    if any(f not in ("merged_from_multiple_sources",) for f in flags):
        status = "needs_review"
    else:
        status = "ok"

    candidate_names = sorted({c.name for c in grp["candidates"]})

    meta = {
        "question_id": grp["question_id"],
        "original_folder_name": grp["original_folder_names"][0]
            if len(grp["original_folder_names"]) == 1
            else grp["original_folder_names"],
        "session": grp["session"],
        "house": grp["house"],
        "state": grp["state"],
        "source_path": grp["source_paths"][0]
            if len(grp["source_paths"]) == 1 else grp["source_paths"],
        "question_file": question_file_out,
        "answer_file_selected": answer_selected_names[0]
            if len(answer_selected_names) == 1 else answer_selected_names,
        "answer_file_selection_reason": (
            "ambiguous" if ambiguous else
            "date_parsed" if reason == "date_parsed" else
            "version_parsed" if reason == "version_parsed" else
            "mtime_fallback" if reason == "mtime_fallback" else
            reason
        ),
        "all_reply_candidates": candidate_names,
        "archives_flagged": grp["archives"],
        "status": status,
        "review_reason": review_flags if status == "needs_review" else None,
    }

    if not dry_run:
        (qout / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    report.question_folders_seen += 1
    if status == "ok":
        report.organized_ok += 1
    else:
        report.needs_review += 1
        for r in review_flags:
            report.review_reasons[r] += 1
    report.questions.append({
        "question_id": meta["question_id"],
        "session": meta["session"],
        "house": meta["house"],
        "state": meta["state"] or "",
        "status": status,
        "review_reason": ";".join(review_flags),
        "selection_reason": meta["answer_file_selection_reason"],
        "n_candidates": len(candidate_names),
        "n_source_folders": len(grp["source_paths"]),
        "output_path": out_key,
        "source_path": grp["source_paths"][0]
            if len(grp["source_paths"]) == 1 else " | ".join(grp["source_paths"]),
    })


def crawl(source_root: Path, out_root: Path, dry_run: bool) -> Report:
    report = Report(
        started=datetime.now(timezone.utc).isoformat(),
        source_root=str(source_root), out_root=str(out_root), dry_run=dry_run,
    )
    merged_group: dict = {}

    for session_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        if _skip_top_level(session_dir, out_root):
            continue
        report.session_folders_seen += 1
        session_slug, sess_reason = normalize_session(session_dir.name)

        if session_slug is None:
            report.orphans.append({
                "type": "unrecognized_session", "path":
                    os.path.relpath(session_dir, source_root),
                "reason": sess_reason,
            })
            continue

        # LEGACY SESSIONS (2014-2017) have no house folder: the house is in the question
        # folder's name (LSQ/RSQ/ASQ/AQ). Harvest those first, wherever they sit under the
        # session, so the loop below only has to deal with real house folders.
        _harvest_prefixed_questions(session_dir, session_slug, sess_reason,
                                    merged_group, report, out_root, source_root, dry_run)

        for house_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
            house_slug, is_cat, is_orphan = normalize_house(house_dir.name)
            if is_orphan:
                # Not a house folder — but it may be a legacy question folder, or a wrapper
                # holding them ("Due up to FEB- MAR 15"). Those were already taken by
                # _harvest_prefixed_questions; only report what it did not claim.
                if _contains_prefixed_questions(house_dir):
                    continue
                report.orphans.append({
                    "type": "non_house_folder",
                    "path": os.path.relpath(house_dir, source_root),
                    "reason": "unrecognized_house_level_folder",
                })
                continue
            if is_cat:
                # category at house level is unexpected -> flag but skip
                report.orphans.append({
                    "type": "category_at_house_level",
                    "path": os.path.relpath(house_dir, source_root),
                    "reason": "category_folder_where_house_expected",
                })
                continue

            base_ctx = {"session": session_slug, "house": house_slug}
            if sess_reason:
                # attach a session-level flag via a sentinel question flag later
                base_ctx["session_flag"] = sess_reason

            _descend_for_questions(house_dir, base_ctx, house_slug,
                                   merged_group, report, out_root, source_root,
                                   dry_run, state=None, depth=0)

    # record loose supporting files at session & house levels
    _collect_loose_files(source_root, out_root, report)

    # finalize all merged groups
    for out_key, grp in merged_group.items():
        try:
            finalize_group(out_key, grp, report, out_root, dry_run)
        except LockedOutputError as e:
            # A locked output folder must not abort the whole crawl. Skip it,
            # record it so a human can re-run after closing the locking process.
            report.review_reasons["output_locked_skipped"] += 1
            report.orphans.append({
                "type": "output_locked_skipped",
                "path": out_key,
                "reason": str(e),
            })
            print(f"[LOCKED] skipped {out_key}: {e}", file=sys.stderr)

    return report


_LEGACY_SCAN_DEPTH = 3          # session/<wrapper>/<wrapper>/<LSQ …> is the deepest seen


def _contains_prefixed_questions(node: Path, depth: int = 0) -> bool:
    """True if `node` is, or contains, a legacy house-prefixed question folder (LSQ/RSQ/…)."""
    if depth > _LEGACY_SCAN_DEPTH:
        return False
    if house_from_question_name(node.name):
        return True
    try:
        children = [p for p in node.iterdir() if p.is_dir()]
    except OSError:
        return False
    return any(_contains_prefixed_questions(c, depth + 1) for c in children)


def _harvest_prefixed_questions(node: Path, session_slug: str, sess_reason,
                                merged_group, report, out_root, source_root, dry_run,
                                depth: int = 0):
    """
    Walk a legacy session and process every house-prefixed question folder found.

    The house comes from the folder NAME (LSQ/RSQ/ASQ/AQ) rather than a parent directory, so
    these sessions need no house level at all. Wrapper folders ("Due up to …", "Assembly
    Question", "Assurance") are passed through. A folder that both carries a prefix and is a
    real question folder is processed and not descended into further.
    """
    if depth > _LEGACY_SCAN_DEPTH:
        return
    try:
        children = sorted(p for p in node.iterdir() if p.is_dir())
    except OSError:
        return

    for child in children:
        house = house_from_question_name(child.name)
        if house:
            try:
                if not is_question_folder(child):
                    # a prefixed wrapper (e.g. "LSQ 10622" holding "REPLY FROM HR") —
                    # descend so the real question folder inside is still picked up
                    _harvest_prefixed_questions(child, session_slug, sess_reason,
                                                merged_group, report, out_root,
                                                source_root, dry_run, depth + 1)
                    continue
            except OSError:
                continue
            ctx = {"session": session_slug, "house": house, "state": None}
            try:
                process_question(child, ctx, merged_group, report,
                                 out_root, source_root, dry_run)
                report.review_reasons["legacy_prefixed_question_folder"] += 1
            except Exception as e:      # noqa: BLE001 — one bad folder must not stop the crawl
                report.orphans.append({
                    "type": "legacy_question_failed",
                    "path": os.path.relpath(child, source_root),
                    "reason": f"{type(e).__name__}: {e}",
                })
            continue

        # not prefixed itself: descend only if a prefixed question lives somewhere below
        if normalize_house(child.name)[0] is None and _contains_prefixed_questions(child):
            _harvest_prefixed_questions(child, session_slug, sess_reason,
                                        merged_group, report, out_root, source_root,
                                        dry_run, depth + 1)


def _descend_for_questions(node: Path, ctx: dict, house_slug: str,
                           merged_group, report, out_root, source_root,
                           dry_run, state, depth):
    """
    From a house folder, find question folders, passing through category
    folders and (for vidhan) an optional state level.
    """
    for child in sorted(p for p in node.iterdir() if p.is_dir()):
        name_low = norm_ws(child.name).lower()

        if name_low in REPLY_FOLDER_NAMES:
            continue  # replies handled inside their question folder

        # Vidhan state level (only one level deep, only for vidhan)
        if (house_slug == "vidhan_sabha" and state is None
                and depth == 0 and looks_like_state(child.name)
                and not is_question_folder(child)):
            _descend_for_questions(child, ctx, house_slug, merged_group,
                                   report, out_root, source_root, dry_run,
                                   state=child.name, depth=depth + 1)
            continue

        # Category / grouping folder -> pass through
        if name_low in CATEGORY_FOLDERS and not is_question_folder(child):
            _descend_for_questions(child, ctx, house_slug, merged_group,
                                   report, out_root, source_root, dry_run,
                                   state=state, depth=depth + 1)
            continue

        if is_question_folder(child):
            qctx = {"session": ctx["session"], "house": house_slug,
                    "state": state}
            process_question(child, qctx, merged_group, report, out_root,
                             source_root, dry_run)
            # propagate session-year ambiguity flag onto the group
            if ctx.get("session_flag"):
                for key, grp in merged_group.items():
                    if os.path.relpath(child, source_root) in grp["source_paths"]:
                        grp["flags"].add(ctx["session_flag"])
        else:
            # a dir that's neither reply/category/state/question
            if any(True for _ in child.rglob("*")):
                report.orphans.append({
                    "type": "unmatched_folder",
                    "path": os.path.relpath(child, source_root),
                    "reason": "did_not_match_question_pattern",
                })
            else:
                report.orphans.append({
                    "type": "empty_folder",
                    "path": os.path.relpath(child, source_root),
                    "reason": "empty",
                })


def _collect_loose_files(source_root: Path, out_root: Path, report: Report):
    """List loose supporting files sitting at session/house level (not inside
    a question folder). These are copied to organized/_supporting/ and listed."""
    for session_dir in (p for p in source_root.iterdir() if p.is_dir()):
        if _skip_top_level(session_dir, out_root):
            continue
        # session-level loose files
        for f in session_dir.iterdir():
            if f.is_file() and not is_noise(f.name):
                report.orphans.append({
                    "type": "loose_session_file",
                    "path": os.path.relpath(f, source_root),
                    "reason": "supporting_file_at_session_level",
                })
        for house_dir in (p for p in session_dir.iterdir() if p.is_dir()):
            for f in house_dir.iterdir():
                if f.is_file() and not is_noise(f.name):
                    report.orphans.append({
                        "type": "loose_house_file",
                        "path": os.path.relpath(f, source_root),
                        "reason": "supporting_file_at_house_level",
                    })


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_reports(report: Report, out_root: Path, dry_run: bool):
    rep_dir = out_root / "_reports"
    if not dry_run:
        rep_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "started": report.started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "source_root": report.source_root,
        "out_root": report.out_root,
        "dry_run": report.dry_run,
        "session_folders_seen": report.session_folders_seen,
        "question_folders_seen": report.question_folders_seen,
        "organized_ok": report.organized_ok,
        "needs_review": report.needs_review,
        "review_reasons": dict(report.review_reasons.most_common()),
        "orphan_count": len(report.orphans),
        "orphans_by_type": dict(Counter(o["type"] for o in report.orphans)),
    }
    full = {"summary": summary, "questions": report.questions,
            "orphans": report.orphans}

    if not dry_run:
        (rep_dir / "report.json").write_text(
            json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8")

        with (rep_dir / "report.csv").open("w", newline="", encoding="utf-8") as fh:
            if report.questions:
                w = csv.DictWriter(fh, fieldnames=list(report.questions[0].keys()))
                w.writeheader()
                w.writerows(report.questions)

        with (rep_dir / "orphans.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["type", "path", "reason"])
            w.writeheader()
            w.writerows(report.orphans)

    return summary, rep_dir


def print_summary(summary: dict, rep_dir: Path, dry_run: bool):
    line = "=" * 60
    print(line)
    print("NHPC Parliament Data — Phase 1 reorganization" +
          ("  [DRY RUN]" if dry_run else ""))
    print(line)
    print(f"Session folders seen      : {summary['session_folders_seen']}")
    print(f"Question folders organized: {summary['question_folders_seen']}")
    print(f"  ok                      : {summary['organized_ok']}")
    print(f"  needs_review            : {summary['needs_review']}")
    print(f"Orphan/supporting items   : {summary['orphan_count']}")
    print()
    if summary["review_reasons"]:
        print("Top review reasons:")
        for reason, n in list(summary["review_reasons"].items())[:12]:
            print(f"  {n:>4}  {reason}")
    print()
    if summary["orphans_by_type"]:
        print("Orphans by type:")
        for t, n in summary["orphans_by_type"].items():
            print(f"  {n:>4}  {t}")
    print()
    print(f"Full report: {rep_dir / 'report.json'}")
    print(f"             {rep_dir / 'report.csv'}")
    print(f"             {rep_dir / 'orphans.csv'}")
    print(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description="NHPC parliament data Phase-1 reorg")
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true",
                    help="scan and report without copying any files")
    args = ap.parse_args(argv)

    # UTF-8 stdout for Devanagari on Windows consoles
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    source_root = Path(args.source).resolve()
    out_root = Path(args.out).resolve()

    if out_root == source_root or out_root in source_root.parents:
        print("ERROR: output directory must not be the source root.",
              file=sys.stderr)
        return 2

    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    report = crawl(source_root, out_root, args.dry_run)
    summary, rep_dir = write_reports(report, out_root, args.dry_run)
    print_summary(summary, rep_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
