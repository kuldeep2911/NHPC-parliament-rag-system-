"""
Upload config. Every knob from the environment; validated at startup; nothing hardcoded.

The defaults are deliberately conservative. An upload endpoint is the highest-risk surface
in the application -- it is the one place where an outsider's bytes become files on our
disk -- so the limits start tight and are raised deliberately, not the other way round.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(k, d=""):     return os.getenv(k, d).strip()
def _env_int(k, d):
    try:    return int(os.getenv(k, str(d)))
    except ValueError: return d
def _env_bool(k, d):   return os.getenv(k, str(d)).strip().lower() in ("1", "true", "yes", "on")


# The ONLY types that may enter the source tree. Everything the live corpus actually
# contains (pdf 2559, docx 2164, xlsx 236, doc 79, xls 9, txt 2) -- and nothing else.
# Notably NOT: .zip/.db/.tmp (present in the corpus but never wanted from an upload),
# and nothing executable or scriptable.
DEFAULT_ALLOWED = "pdf,doc,docx,xls,xlsx,txt"


@dataclass
class UploadConfig:
    upload_enabled: bool = field(default_factory=lambda: _env_bool("UPLOAD_ENABLED", True))

    # Staging MUST be on the same volume as the source root, or os.replace() degrades from
    # an atomic rename into a copy -- and a copy can be observed half-written by the
    # watcher. Defaults to a dotted dir INSIDE the source root, which guarantees the same
    # volume. The watcher ignores dot-directories (see watcher/runner.py:_ignored).
    upload_staging_root: str = field(
        default_factory=lambda: _env("UPLOAD_STAGING_ROOT", ""))

    upload_max_file_mb:  int = field(default_factory=lambda: _env_int("UPLOAD_MAX_FILE_MB", 50))
    upload_max_total_mb: int = field(default_factory=lambda: _env_int("UPLOAD_MAX_TOTAL_MB", 500))
    upload_max_files:    int = field(default_factory=lambda: _env_int("UPLOAD_MAX_FILES", 500))

    upload_allowed_ext: str = field(
        default_factory=lambda: _env("UPLOAD_ALLOWED_EXT", DEFAULT_ALLOWED))

    def allowed_exts(self) -> set[str]:
        """Normalised to '.pdf' form, lowercase."""
        out = set()
        for e in self.upload_allowed_ext.split(","):
            e = e.strip().lower().lstrip(".")
            if e:
                out.add("." + e)
        return out

    def staging_root(self) -> str:
        """Absolute staging dir. Defaults inside the source root (same volume => atomic
        move). The leading dot keeps it out of the watcher's sight."""
        s = self.upload_staging_root
        if s:
            return os.path.abspath(s)
        src = os.path.abspath(getattr(self, "source_root", None) or "Original Data")
        return os.path.join(src, ".upload_staging")

    def validate_upload(self):
        errs = []
        if not self.upload_enabled:
            return errs
        if not self.allowed_exts():
            errs.append("UPLOAD_ALLOWED_EXT is empty — no file type could ever be uploaded")
        if self.upload_max_file_mb < 1:
            errs.append("UPLOAD_MAX_FILE_MB must be >= 1")
        if self.upload_max_total_mb < self.upload_max_file_mb:
            errs.append("UPLOAD_MAX_TOTAL_MB must be >= UPLOAD_MAX_FILE_MB")
        if self.upload_max_files < 1:
            errs.append("UPLOAD_MAX_FILES must be >= 1")

        # A staging dir on a DIFFERENT volume silently turns the atomic move into a
        # copy+delete, which reintroduces exactly the half-written-file race that staging
        # exists to prevent. Catch it at boot, not in production.
        src = os.path.abspath(getattr(self, "source_root", None) or "Original Data")
        stage = self.staging_root()
        if os.path.splitdrive(src)[0].lower() != os.path.splitdrive(stage)[0].lower():
            errs.append(
                f"UPLOAD_STAGING_ROOT ({stage}) is on a different volume from "
                f"NHPC_SOURCE_ROOT ({src}). The move into the source tree would no longer "
                f"be atomic, and the watcher could read a half-written file.")
        return errs
