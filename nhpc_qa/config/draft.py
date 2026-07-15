"""
Draft-assist config. Every knob from the environment; nothing hardcoded.

DRAFT_ENABLED is separate from GENERATION_ENABLED on purpose:

    GENERATION_ENABLED  wires the OLD generate node INTO THE LANGGRAPH, so every /query
                        waits for the LLM before returning any results. Still false.
    DRAFT_ENABLED       enables the /draft ENDPOINT, which the UI calls AFTER results are
                        already on screen. Retrieval never waits for it.

They are different mechanisms with different latency consequences, so they get different
switches. Turning drafting on must not silently slow down search.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(k, d=""):
    return os.getenv(k, d).strip()


def _env_int(k, d):
    try:
        return int(os.getenv(k, str(d)))
    except ValueError:
        return d


def _env_bool(k, d):
    return os.getenv(k, str(d)).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class DraftConfig:
    # The /draft endpoint. On -- but the officer still has to CLICK "Generate draft":
    # most searches are exploratory ("has this been asked before?"), and drafting every
    # one of them would burn LLM calls on drafts nobody reads. On-prem that is real GPU
    # time.
    draft_enabled: bool = field(default_factory=lambda: _env_bool("DRAFT_ENABLED", True))

    # How many retrieved answers feed the draft. The whole point is grounding, so more
    # context is not automatically better: past a handful, weaker matches start diluting
    # the strong ones and the model drifts toward the average of them.
    draft_context_k: int = field(default_factory=lambda: _env_int("DRAFT_CONTEXT_K", 5))

    # Enough for a point-wise reply plus key points and gaps. Generous rather than tight:
    # a truncated draft is worse than a slow one, because the officer cannot tell which
    # parts were cut.
    draft_max_tokens: int = field(default_factory=lambda: _env_int("DRAFT_MAX_TOKENS", 2400))

    # ZERO, and it should stay zero. This is a parliamentary reply: we want the same draft
    # from the same evidence every time. Temperature is creativity, and creativity here is
    # a synonym for invention.
    draft_temperature: float = field(
        default_factory=lambda: float(_env("DRAFT_TEMPERATURE", "0") or 0))

    def validate_draft(self):
        errs = []
        if not self.draft_enabled:
            return errs
        if self.draft_context_k < 1:
            errs.append("DRAFT_CONTEXT_K must be >= 1")
        if self.draft_max_tokens < 400:
            errs.append("DRAFT_MAX_TOKENS is too small for a point-wise reply (>= 400)")
        if self.draft_temperature > 0.3:
            errs.append("DRAFT_TEMPERATURE above 0.3 invites invented facts in a "
                        "parliamentary reply; keep it at 0")
        return errs
