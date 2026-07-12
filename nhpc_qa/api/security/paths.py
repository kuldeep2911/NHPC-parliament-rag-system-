"""
File resolution — ID to path, NEVER path to path.

THE THREAT: if the API accepted a filesystem path from the client, an officer (or an
attacker with an officer's token) could ask for `../../../../etc/passwd` or any file on
the server. So the endpoint takes a DOCUMENT ID (doc_key) plus a file kind
('reply' | 'annexure' + ref_label) and this module maps that to a path SERVER-SIDE,
from the database, and then proves the result is inside the organized/ root.

TWO INDEPENDENT DEFENCES, both required:
  1. The path is never taken from the client. It is looked up in the DB by doc_key.
  2. The resolved path is realpath()'d and asserted to be inside the realpath()'d root.
     Defence 2 is not redundant: a bad Phase-2 parse, a symlink, or a future code change
     could still produce an escaping path, and the officer's input is not the only way
     for one to appear.

Anything that fails either check raises FileAccessDenied and is AUDITED as a denial.
"""

from __future__ import annotations

import os


class FileAccessDenied(Exception):
    """Raised when a file cannot be served. Always audited."""


class FileNotAvailable(Exception):
    """The file is legitimately referenced but was never found on disk
    (annexure with file_present=false). This is an honest 'unavailable', not a denial."""


_REPLY_SQL = """
SELECT d.answer_file_path
FROM diaries d WHERE d.doc_key = %(doc_key)s
"""

_ANNEX_SQL = """
SELECT a.file_path, a.file_present
FROM annexures a WHERE a.doc_key = %(doc_key)s AND a.ref_label = %(ref_label)s
"""


def _jail(root: str, rel_path: str) -> str:
    """
    Resolve `rel_path` under `root` and PROVE the result stays inside it.

    os.path.realpath resolves '..' AND symlinks, so this defeats both `../..` traversal
    and a symlink planted inside organized/ that points outside it. commonpath() is used
    rather than str.startswith(), because startswith('/data/organized') would wrongly
    accept '/data/organized-evil/secrets'.
    """
    root_real = os.path.realpath(root)
    target = os.path.realpath(os.path.join(root_real, rel_path))
    try:
        inside = os.path.commonpath([root_real, target]) == root_real
    except ValueError:
        inside = False          # different drives on Windows
    if not inside:
        raise FileAccessDenied(
            f"resolved path escapes the document root (blocked): {rel_path!r}")
    if not os.path.isfile(target):
        raise FileNotAvailable(f"file not found on disk: {rel_path!r}")
    return target


def resolve(conn, cfg, doc_key: str, file_kind: str, ref_label: str | None = None):
    """
    Map (doc_key, file_kind[, ref_label]) -> an absolute path guaranteed inside the root.

    Returns (abs_path, rel_path). Raises FileAccessDenied / FileNotAvailable.
    """
    if file_kind not in ("reply", "annexure"):
        raise FileAccessDenied(f"unknown file_kind {file_kind!r}")

    with conn.cursor() as cur:
        if file_kind == "reply":
            cur.execute(_REPLY_SQL, {"doc_key": doc_key})
            row = cur.fetchone()
            if not row:
                raise FileAccessDenied(f"unknown doc_key {doc_key!r}")
            rel = row[0]
            if not rel:
                raise FileNotAvailable(f"{doc_key} has no reply file recorded")
        else:
            if not ref_label:
                raise FileAccessDenied("annexure requested without a ref_label")
            cur.execute(_ANNEX_SQL, {"doc_key": doc_key, "ref_label": ref_label})
            row = cur.fetchone()
            if not row:
                raise FileAccessDenied(
                    f"{doc_key} does not reference an annexure {ref_label!r}")
            rel, present = row
            if not present or not rel:
                # honest: the reply cites it, but Phase 1/2 never found the file
                raise FileNotAvailable(
                    f"{ref_label} is referenced by {doc_key} but the file is unavailable")

    return _jail(cfg.organized_root, rel), rel


def content_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
        ".txt": "text/plain; charset=utf-8",
    }.get(ext, "application/octet-stream")
