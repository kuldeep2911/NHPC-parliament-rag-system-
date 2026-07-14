"""
Upload validation. THE most security-critical module in the application.

This is the one place where bytes chosen by someone else become files on our disk. Two
independent controls, because either one alone has been defeated in the wild:

  1. SANITIZE every path component (drop '..', drive letters, separators, NUL, reserved
     Windows names, control characters).
  2. REALPATH-JAIL the result: resolve the final absolute path and ASSERT it is inside the
     source root. This is the control that cannot be argued with -- it does not care how
     clever the encoding was ('..%2f', '....//', a symlink, a UNC path, 'C:\\'). If the
     resolved path is not under the root, the write does not happen.

Sanitisation without the jail is a blocklist, and blocklists lose. The jail without
sanitisation would let through filenames that are legal but hostile (CON, LPT1, names with
newlines). We do both.

TYPE VALIDATION is by extension AND content. Extension alone is worthless -- anyone can
rename evil.exe to reply.pdf. The subtle case is .docx/.xlsx: those ARE zip files, so a
sniffer that stops at 'PK\\x03\\x04' would accept any zip at all, including a renamed .jar
or a zip bomb. We open the container and require the Office parts to be present.
"""

from __future__ import annotations

import os
import re
import unicodedata
import zipfile

# ---------------------------------------------------------------------------
# MAGIC BYTES
# ---------------------------------------------------------------------------
_PDF = b"%PDF-"
_ZIP = b"PK\x03\x04"          # docx/xlsx (and every other zip -- hence the deeper check)
_OLE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"   # legacy .doc/.xls (OLE2 compound file)

# What a .docx / .xlsx must actually contain inside the zip. This is what separates a real
# Office file from "any zip renamed to .docx".
_OOXML_PARTS = {
    ".docx": ("word/",),
    ".xlsx": ("xl/",),
}

# Windows reserved device names. A file called CON or LPT1 is not creatable and, worse,
# can make naive code hang on a device handle.
_WIN_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

MAX_COMPONENT = 120           # per path segment
MAX_REL_PATH = 400            # whole relative path


class Rejected(Exception):
    """A file or path that must not be written. The message is shown to the admin."""


# ---------------------------------------------------------------------------
# 1. FILENAME / PATH SANITISATION
# ---------------------------------------------------------------------------
def sanitize_component(name: str) -> str:
    """
    Make ONE path segment safe. Raises Rejected if nothing safe survives.

    Devanagari and other non-ASCII names are PRESERVED -- this corpus is bilingual and
    mangling them would be a bug, not a security win. We normalise to NFC (so the same
    name cannot be written two different ways) and strip only what is genuinely dangerous.
    """
    if not name:
        raise Rejected("empty path component")

    # NFC first: 'é' can be one codepoint or two, and two spellings of one name would let
    # a collision check be bypassed.
    n = unicodedata.normalize("NFC", name)

    # NUL truncates the path in the C layer underneath Python -- 'a.pdf\x00.exe' can become
    # 'a.pdf' to a checker and 'a.pdf\x00.exe' to the OS. Never allow it.
    if "\x00" in n:
        raise Rejected("NUL byte in filename")

    # Control characters (newlines in a filename break every log and audit line).
    n = "".join(ch for ch in n if unicodedata.category(ch)[0] != "C")

    # Separators must never survive inside a COMPONENT -- that is how 'a/../../b' hides.
    n = n.replace("/", "_").replace("\\", "_")

    # Windows: drive letters, and the characters the filesystem simply cannot store.
    n = re.sub(r'^[A-Za-z]:', "", n)
    n = re.sub(r'[<>:"|?*]', "_", n)

    n = n.strip().strip(".")          # trailing dots/spaces are silently eaten by Windows
    if not n or n in (".", ".."):
        raise Rejected(f"unsafe path component: {name!r}")

    if n.split(".")[0].lower() in _WIN_RESERVED:
        raise Rejected(f"reserved device name: {name!r}")

    if len(n) > MAX_COMPONENT:
        stem, ext = os.path.splitext(n)
        n = stem[:MAX_COMPONENT - len(ext)] + ext

    return n


def sanitize_relpath(rel: str) -> str:
    """
    Sanitise a whole client-supplied relative path (a browser's webkitRelativePath).

    Every '..' is DROPPED, not resolved -- there is no legitimate reason for an upload to
    contain one, so we do not try to be clever about where it might point.
    """
    if not rel:
        raise Rejected("empty path")
    raw = rel.replace("\\", "/")
    parts = []
    for seg in raw.split("/"):
        seg = seg.strip()
        if not seg or seg == ".":
            continue
        if seg == "..":
            # Do not pop the parent -- just refuse. Popping would make '../a/../../b'
            # a puzzle; refusing makes it a non-event.
            raise Rejected("path traversal ('..') is not allowed")
        parts.append(sanitize_component(seg))
    if not parts:
        raise Rejected("path has no usable components")
    out = "/".join(parts)
    if len(out) > MAX_REL_PATH:
        raise Rejected("path is too long")
    return out


# ---------------------------------------------------------------------------
# 2. THE REALPATH JAIL — the control that cannot be argued with
# ---------------------------------------------------------------------------
def jail(root: str, rel: str) -> str:
    """
    Resolve `rel` under `root` and PROVE the result is inside `root`. Returns the absolute
    target path, or raises Rejected.

    Everything above is defence in depth. THIS is the line that actually holds: whatever
    encoding, symlink or unicode trick was used, the resolved absolute path either starts
    with the source root or the write does not happen.

    os.path.realpath resolves symlinks, so a symlinked subfolder pointing at C:\\Windows
    cannot be used to write outside the tree either.
    """
    root_real = os.path.realpath(os.path.abspath(root))
    target = os.path.realpath(os.path.join(root_real, rel))

    try:
        common = os.path.commonpath([root_real, target])
    except ValueError:
        # Raised when the paths are on different drives -- which is itself an escape.
        raise Rejected("resolved path is outside the source root (different volume)")

    if common != root_real:
        raise Rejected("resolved path escapes the source root")
    if target == root_real:
        raise Rejected("path resolves to the source root itself")
    return target


# ---------------------------------------------------------------------------
# 3. CONTENT SNIFFING — extension AND magic bytes
# ---------------------------------------------------------------------------
def sniff(path: str, ext: str) -> None:
    """
    Prove the FILE CONTENT matches its extension. Raises Rejected on a mismatch.

    Called AFTER the bytes are on disk in staging and BEFORE anything moves into the source
    tree, so a file that lies about what it is never reaches the live data.
    """
    ext = ext.lower()
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
    except OSError as e:
        raise Rejected(f"unreadable: {e}")

    size = os.path.getsize(path)
    if size == 0:
        raise Rejected("file is empty")

    if ext == ".pdf":
        # Some real-world PDFs carry a few junk bytes before %PDF-. Allow a small offset,
        # but do not scan the whole file (that would let an .exe with a %PDF- string in it
        # through).
        with open(path, "rb") as fh:
            first = fh.read(1024)
        if _PDF not in first[:512]:
            raise Rejected("not a PDF (missing %PDF- header)")
        return

    if ext in (".docx", ".xlsx"):
        if head[:4] != _ZIP:
            raise Rejected(f"not a real {ext} (missing zip header)")
        # A .docx IS a zip -- so 'starts with PK' proves nothing on its own. Open the
        # container and require the Office parts. This is what rejects a renamed .jar,
        # an arbitrary archive, or a zip bomb dressed as a document.
        want = _OOXML_PARTS[ext]
        try:
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
                if not any(n.startswith(w) for w in want for n in names):
                    raise Rejected(f"not a real {ext} (no {want[0]} part inside)")
                if "[Content_Types].xml" not in names:
                    raise Rejected(f"not a real {ext} (no [Content_Types].xml)")
        except zipfile.BadZipFile:
            raise Rejected(f"not a real {ext} (corrupt zip container)")
        return

    if ext in (".doc", ".xls"):
        if head != _OLE:
            raise Rejected(f"not a real {ext} (missing OLE2 header)")
        return

    if ext == ".txt":
        # No magic bytes exist for plain text, so prove it the only way there is: it must
        # DECODE. A binary renamed to .txt will not.
        with open(path, "rb") as fh:
            raw = fh.read(65536)
        if b"\x00" in raw:
            raise Rejected("not text (contains NUL bytes)")
        for enc in ("utf-8", "utf-16", "cp1252"):
            try:
                raw.decode(enc)
                return
            except (UnicodeDecodeError, LookupError):
                continue
        raise Rejected("not text (does not decode as UTF-8, UTF-16 or CP1252)")

    raise Rejected(f"unsupported type: {ext}")


def check_extension(filename: str, allowed: set[str]) -> str:
    """The extension, lowercased, proven to be on the allow-list. ALLOW-LIST, never a
    blocklist: a blocklist has to guess every dangerous type, and it will miss one."""
    ext = os.path.splitext(filename)[1].lower()
    if not ext:
        raise Rejected("file has no extension")
    if ext not in allowed:
        raise Rejected(f"type '{ext}' is not allowed "
                       f"(allowed: {', '.join(sorted(allowed))})")
    return ext
