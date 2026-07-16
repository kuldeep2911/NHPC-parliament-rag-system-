"""
Supporting-documents config. Reference files an officer pulls into a draft.

Separate from UploadConfig on purpose: this is a DIFFERENT document class with its own
root, its own categories, and its own on/off switch. It REUSES upload's security limits
(size caps, allowed extensions) rather than redefining them -- an upload is an upload.

The category list is a REGISTRY, not a hardcoded enum: adding "annual_reports" later is one
env entry plus a folder, never a code change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(k, d=""):     return os.getenv(k, d).strip()
def _env_bool(k, d):   return os.getenv(k, str(d)).strip().lower() in ("1", "true", "yes", "on")


# slug:Label, comma-separated. The slug is the folder name and the DB category; the label is
# what the officer sees in the dropdown.
DEFAULT_CATEGORIES = ("financial_reports:Financial Reports,"
                      "projects_progress:Projects Progress,"
                      "csr:CSR")


@dataclass
class SupportingConfig:
    # OFF by default: ships dark, enabled deliberately after review. When off, the endpoints
    # 503 and the draft dropdown is absent -- the draft behaves exactly as it does today.
    supporting_enabled: bool = field(
        default_factory=lambda: _env_bool("SUPPORTING_ENABLED", True))

    # The tree lives BESIDE the Q&A source data under the same root, never inside it, so the
    # Q&A crawler never sees a supporting file. Default: '<root>/supporting_documents'.
    supporting_root: str = field(
        default_factory=lambda: _env("SUPPORTING_ROOT", ""))

    supporting_categories_raw: str = field(
        default_factory=lambda: _env("SUPPORTING_CATEGORIES", DEFAULT_CATEGORIES))

    # Capture the as-of date at ingest with the LLM, then let the admin confirm it. Off ->
    # the admin types it by hand (the field stays mandatory either way).
    supporting_llm_asof: bool = field(
        default_factory=lambda: _env_bool("SUPPORTING_LLM_ASOF", True))

    def supporting_categories(self) -> dict:
        """{slug: label}. Order preserved for a stable dropdown."""
        out = {}
        for pair in self.supporting_categories_raw.split(","):
            pair = pair.strip()
            if not pair:
                continue
            slug, _, label = pair.partition(":")
            slug = slug.strip().lower()
            if slug:
                out[slug] = (label.strip() or slug.replace("_", " ").title())
        return out

    def supporting_root_abs(self) -> str:
        """
        Absolute supporting-documents root: organized/supporting_documents/ by default.

        This sits INSIDE organized/, but it is SAFE from the Q&A pipeline for two reasons,
        both load-bearing:
          1. The Q&A loader globs organized/*/*/*/parsed.json (exactly three levels) -- a
             supporting file at organized/supporting_documents/<category>/<file> never
             matches that shape, so it is never loaded as a parliamentary question.
          2. The crawler EXPLICITLY skips this subtree (see crawler._SKIP_DIRS), so a full
             re-crawl -- which rewrites organized/ -- never touches or deletes it.
        Change the location with SUPPORTING_ROOT if you must, but the two guarantees above
        assume this default.
        """
        if self.supporting_root:
            return os.path.abspath(self.supporting_root)
        org = os.path.abspath(getattr(self, "organized_root", None) or "organized")
        return os.path.join(org, "supporting_documents")

    def validate_supporting(self):
        errs = []
        if not self.supporting_enabled:
            return errs
        if not self.supporting_categories():
            errs.append("SUPPORTING_CATEGORIES is empty — at least one 'slug:Label' needed")
        # category slugs become folder names -> must be path-safe
        for slug in self.supporting_categories():
            if not slug.replace("_", "").isalnum():
                errs.append(f"supporting category slug {slug!r} must be alphanumeric/underscore")
        return errs
