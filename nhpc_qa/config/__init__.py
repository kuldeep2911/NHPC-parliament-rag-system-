"""
ONE configuration object for the whole application.

WHY THIS EXISTS. There used to be three config classes:

    phase2.Config          (59 fields: parser, LLM, trace, Langfuse)
    phase3.Phase3Config    (17 fields: DB, embeddings)   -- SEPARATE tree
    phase4.Phase4Config    (31 fields: retrieval, rerank, API)

phase2's tree and phase3/4's tree did not know about each other, and the `_env_str /
_env_int / _env_bool` helpers were defined three times. That split caused real bugs:

  * Phase4Config had no `resolve_backends()` (a phase-2 method), so the generation node
    crashed at runtime with AttributeError.
  * Phase4Config had no `langfuse_enabled`, so four Langfuse fields had to be redeclared.
  * generation/draft.py had to build a SECOND config object at runtime just to get an LLM:
        from nhpc_qa.config.parse import Config as Phase2Config
        llm = get_llm(Phase2Config())

`Settings` unifies all three by INHERITANCE, not by retyping. That matters: there are 78
env vars in the contract, and re-declaring them by hand is how you silently drop one.
Every field, default, and method of all three classes is inherited unchanged, so:

  * EVERY EXISTING ENV VAR NAME STILL WORKS. NHPC_LLM_BACKEND, EMBED_MODEL,
    PHASE3_DB_DSN, RERANK_ENABLED, PHASE4_API_PORT ... all unchanged. Your .env and the
    deployment need no edits. This is an internal cleanup, not a rename.
  * One object now satisfies every provider factory (parser, llm, embedder, reranker),
    so the runtime workaround above is gone.

MRO note: Settings(RetrievalConfig, ParseConfig) -- RetrievalConfig already extends
IndexConfig, so the chain is Settings -> RetrievalConfig -> IndexConfig -> ParseConfig.
Field names do not collide across the trees except for the four Langfuse fields and
`organized_root`, which are IDENTICAL declarations (same env var, same default), so the
MRO picking either one is correct.
"""

from __future__ import annotations

from dataclasses import dataclass

# The three original config classes, still in their original modules (logic untouched).
# They are the source of truth for all 78 env vars; Settings only composes them.
from nhpc_qa.config.parse import Config as ParseConfig       # parser, LLM, trace, Langfuse
from nhpc_qa.config.index import Phase3Config as IndexConfig  # DB, embeddings
from nhpc_qa.config.retrieval import Phase4Config as RetrievalConfig   # retrieval, rerank, API

from nhpc_qa.config.parse import load_dotenv                  # noqa: F401 (re-export)


@dataclass
class Settings(RetrievalConfig, ParseConfig):
    """
    The single application config. Inherits every field of all three trees.

        from nhpc_qa.config import Settings, load_dotenv
        load_dotenv()
        cfg = Settings()
        errs = cfg.validate_all()      # fail fast, with every problem at once
    """

    def validate_all(self, need_db=True, need_embed=True, need_rerank=None):
        """
        Every check from every layer, in one call. Returns a list of human-readable
        errors ([] = OK) so the caller can print them all rather than dying on the first.
        """
        errs = []
        # phase-2 backends (parser + LLM)
        errs += list(self.validate())
        # phase-3 (DB + embeddings) -- validate() on Phase3Config takes these flags
        errs += list(IndexConfig.validate(self, need_db=need_db, need_embed=need_embed))
        # phase-4 (retrieval + rerank), which also re-runs the phase-3 checks
        if need_rerank is None:
            need_rerank = bool(getattr(self, "rerank_enabled", False))
        if need_rerank:
            errs += [e for e in self.validate_phase4() if e not in errs]
        return _dedup(errs)

    def describe_all(self) -> dict:
        """Redacted snapshot for logs/reports. Never contains a secret."""
        d = dict(self.describe())          # phase-3/4 describe (already redacts the DSN)
        d.update({
            "parser_backend": self.resolve_backends()[0],
            "llm_backend": self.resolve_backends()[1],
            "llm_grouping": self.llm_grouping,
            "langfuse_enabled": self.langfuse_enabled,
        })
        return d


def _dedup(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


__all__ = ["Settings", "load_dotenv"]
