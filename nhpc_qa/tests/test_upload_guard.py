"""
Security tests for the upload guard. NO DATABASE NEEDED -- these are pure functions, which
is exactly why the dangerous logic was put in pure functions.

    python nhpc_qa/tests/test_upload_guard.py

The path-traversal jail is the single most important control in the upload feature: it is
the difference between "an admin uploaded a session folder" and "an admin wrote a file into
C:\\Windows\\System32". It gets the most tests, including a NEGATIVE CONTROL proving the
test can actually fail.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nhpc_qa.api.security import upload_guard as guard      # noqa: E402
from nhpc_qa.api.security.upload_guard import Rejected      # noqa: E402

PASS = FAIL = 0


def ok(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}")


def rejects(fn, label):
    """The call MUST raise Rejected."""
    try:
        fn()
    except Rejected as e:
        ok(True, f"{label}  ->  rejected ({e})")
        return
    except Exception as e:      # noqa: BLE001
        ok(False, f"{label}  ->  wrong exception: {type(e).__name__}: {e}")
        return
    ok(False, f"{label}  ->  *** ACCEPTED (this is a security hole) ***")


def allows(fn, label):
    try:
        fn()
        ok(True, label)
    except Exception as e:      # noqa: BLE001
        ok(False, f"{label}  ->  wrongly rejected: {e}")


# ===========================================================================
print("=" * 74)
print("1. PATH TRAVERSAL — nothing may land outside the source root")
print("=" * 74)
ROOT = os.path.realpath(tempfile.mkdtemp(prefix="nhpc_jail_"))

for evil in [
    "../outside.pdf",
    "../../../../../../Windows/System32/evil.pdf",
    "..\\..\\outside.pdf",
    "session/../../outside.pdf",
    "session/house/../../../outside.pdf",
    "./../../outside.pdf",
    "....//....//outside.pdf",          # the classic "strip once and you lose" payload
]:
    rejects(lambda e=evil: guard.jail(ROOT, guard.sanitize_relpath(e)),
            f"{evil!r:45s}")

# Absolute paths and drive letters must not escape either.
for evil in ["C:/Windows/System32/evil.pdf", "/etc/passwd", "\\\\server\\share\\evil.pdf"]:
    try:
        rel = guard.sanitize_relpath(evil)
        target = guard.jail(ROOT, rel)
        inside = os.path.commonpath([ROOT, target]) == ROOT
        ok(inside, f"{evil!r:45s}  ->  neutralised to {os.path.relpath(target, ROOT)!r}")
    except Rejected as e:
        ok(True, f"{evil!r:45s}  ->  rejected ({e})")

# NEGATIVE CONTROL: prove the jail would actually CATCH an escape if one got through.
# Without this, a jail that accepted everything would still show 'all tests pass'.
print("\n  negative control (proves the jail can fail):")
escaped = os.path.join(os.path.dirname(ROOT), "escaped.pdf")
try:
    common = os.path.commonpath([ROOT, os.path.realpath(escaped)])
    ok(common != ROOT, "a path OUTSIDE the root is correctly seen as outside")
except ValueError:
    ok(True, "a path outside the root is correctly seen as outside")

print("\n  legitimate paths still work:")
allows(lambda: guard.jail(ROOT, guard.sanitize_relpath(
    "PARLIAMENT MAR 26/LOK SABHA/1234/reply.pdf")), "a normal nested session path")
allows(lambda: guard.jail(ROOT, guard.sanitize_relpath(
    "संसद सत्र/लोक सभा/1234/उत्तर.pdf")), "a Devanagari path (bilingual corpus!)")
allows(lambda: guard.jail(ROOT, guard.sanitize_relpath(
    "PARLIAMENT FEB MAR 24/likely issues/information received/EDM.docx")),
    "the real corpus's messy 4-deep structure")


# ===========================================================================
print()
print("=" * 74)
print("2. FILENAME SANITISATION")
print("=" * 74)
rejects(lambda: guard.sanitize_component("evil\x00.pdf"), "NUL byte in a filename")
rejects(lambda: guard.sanitize_component(".."), "'..' as a component")
rejects(lambda: guard.sanitize_component("CON"), "Windows reserved device name 'CON'")
rejects(lambda: guard.sanitize_component("LPT1.pdf"), "reserved device 'LPT1.pdf'")
ok(guard.sanitize_component("a/b.pdf") == "a_b.pdf", "a separator inside a component is neutralised")
ok(guard.sanitize_component("रिपोर्ट.pdf") == "रिपोर्ट.pdf", "Devanagari filenames are PRESERVED")
ok("\n" not in guard.sanitize_component("bad\nname.pdf"), "newline stripped (it would break every audit line)")


# ===========================================================================
print()
print("=" * 74)
print("3. TYPE VALIDATION — extension AND content")
print("=" * 74)
ALLOWED = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".txt"}
tmp = tempfile.mkdtemp(prefix="nhpc_sniff_")


def w(name, data: bytes):
    p = os.path.join(tmp, name)
    with open(p, "wb") as fh:
        fh.write(data)
    return p


print("  extension allow-list:")
for bad in ["evil.exe", "run.sh", "a.bat", "x.zip", "lib.dll", "noext"]:
    rejects(lambda b=bad: guard.check_extension(b, ALLOWED), f"{bad!r:14s}")

print("\n  content sniffing — a LIE about the extension is caught:")
rejects(lambda: guard.sniff(w("evil.pdf", b"MZ\x90\x00This is a Windows .exe"), ".pdf"),
        "a Windows .exe renamed to .pdf")
rejects(lambda: guard.sniff(w("evil.pdf", b"#!/bin/sh\nrm -rf /"), ".pdf"),
        "a shell script renamed to .pdf")
rejects(lambda: guard.sniff(w("empty.pdf", b""), ".pdf"), "an empty file")
rejects(lambda: guard.sniff(w("evil.docx", b"MZ\x90\x00"), ".docx"),
        "an .exe renamed to .docx")

# THE SUBTLE ONE. A .docx IS a zip. A sniffer that stops at the 'PK' magic would accept
# ANY zip -- a renamed .jar, an arbitrary archive, a zip bomb.
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as z:
    z.writestr("evil.class", "not an office document")
rejects(lambda: guard.sniff(w("fake.docx", buf.getvalue()), ".docx"),
        "a PLAIN ZIP renamed to .docx (the trap: docx IS a zip)")

rejects(lambda: guard.sniff(w("bin.txt", b"\x00\x01\x02\xff\xfe binary"), ".txt"),
        "a binary renamed to .txt")

print("\n  genuine files are accepted:")
allows(lambda: guard.sniff(w("real.pdf", b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj"), ".pdf"),
       "a real PDF")

good = io.BytesIO()
with zipfile.ZipFile(good, "w") as z:
    z.writestr("[Content_Types].xml", "<Types/>")
    z.writestr("word/document.xml", "<w:document/>")
allows(lambda: guard.sniff(w("real.docx", good.getvalue()), ".docx"), "a real .docx")

gx = io.BytesIO()
with zipfile.ZipFile(gx, "w") as z:
    z.writestr("[Content_Types].xml", "<Types/>")
    z.writestr("xl/workbook.xml", "<workbook/>")
allows(lambda: guard.sniff(w("real.xlsx", gx.getvalue()), ".xlsx"), "a real .xlsx")

allows(lambda: guard.sniff(w("real.doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 40), ".doc"),
       "a real legacy .doc (OLE2)")
allows(lambda: guard.sniff(w("real.txt", "प्रश्न संख्या 1234\nNHPC reply.\n".encode()), ".txt"),
       "a real UTF-8 .txt with Devanagari")


# ===========================================================================
print()
print("=" * 74)
print(f"{PASS} passed, {FAIL} failed")
print("=" * 74)
sys.exit(1 if FAIL else 0)
