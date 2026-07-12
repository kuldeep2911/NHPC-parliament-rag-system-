"""
Phase-3 configuration. Everything from env; nothing hardcoded; validated at startup.

Secrets (DB password inside the DSN, NIM API key) are read from the environment only
and never logged. `describe()` prints a redacted view safe for the load report.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# Reuse Phase 2's .env loader so both phases read the same project-root .env.
from nhpc_qa.config.parse import load_dotenv  # noqa: F401  (re-exported for callers)


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return v if v is not None and v.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


# The embedding model actually in use. The spec named
# nvidia/llama-3.2-nv-embedqa-1b-v2, which reached END OF LIFE on 2026-05-18 and now
# returns HTTP 410 Gone. Its live successor is llama-nemotron-embed-1b-v2, measured:
#   dim 2048, L2-normalised output (=> COSINE), Devanagari OK, passage/query modes.
DEFAULT_EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
DEFAULT_EMBED_DIM = 2048


@dataclass
class Phase3Config:
    # --- input (READ-ONLY) ---------------------------------------------------
    organized_root: str = field(default_factory=lambda: _env("NHPC_ORGANIZED_ROOT", "organized"))
    reports_dir: str = field(default_factory=lambda: _env("PHASE3_REPORTS_DIR", "phase3/_reports"))

    # --- database ------------------------------------------------------------
    db_dsn: str = field(default_factory=lambda: _env("PHASE3_DB_DSN", ""))
    db_statement_timeout_ms: int = field(default_factory=lambda: _env_int("PHASE3_DB_TIMEOUT_MS", 60000))

    # --- embedding provider --------------------------------------------------
    #   nvidia_nim_api    -> NVIDIA-hosted NIM (dev/now). Text leaves the network.
    #   nvidia_selfhosted -> on-prem HF model / self-hosted NIM (server/later).
    embed_backend: str = field(default_factory=lambda: _env("EMBED_BACKEND", "nvidia_nim_api"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", DEFAULT_EMBED_MODEL))
    embed_dim: int = field(default_factory=lambda: _env_int("EMBED_DIM", DEFAULT_EMBED_DIM))
    embed_url: str = field(default_factory=lambda: _env(
        "EMBED_URL", "https://integrate.api.nvidia.com/v1/embeddings"))
    embed_api_key_env: str = field(default_factory=lambda: _env("EMBED_API_KEY_ENV", "NVIDIA_EMBED_API_KEY"))
    embed_batch_size: int = field(default_factory=lambda: _env_int("EMBED_BATCH_SIZE", 32))
    embed_timeout_s: int = field(default_factory=lambda: _env_int("EMBED_TIMEOUT_S", 120))
    embed_max_retries: int = field(default_factory=lambda: _env_int("EMBED_MAX_RETRIES", 5))
    # 'passage' when INDEXING sub-questions; Phase 4 uses 'query' at search time.
    embed_input_type: str = field(default_factory=lambda: _env("EMBED_INPUT_TYPE", "passage"))
    # on-prem self-hosted model (used when embed_backend=nvidia_selfhosted)
    embed_selfhosted_url: str = field(default_factory=lambda: _env("EMBED_SELFHOSTED_URL", ""))
    embed_selfhosted_model_path: str = field(default_factory=lambda: _env("EMBED_SELFHOSTED_MODEL_PATH", ""))

    # --- indexing ------------------------------------------------------------
    # hnsw = approximate (default) | none = exact scan (viable at this corpus size)
    vector_index: str = field(default_factory=lambda: _env("PHASE3_VECTOR_INDEX", "hnsw"))

    def api_key(self):
        """Embedding API key from env. None if unset. Never logged."""
        return os.environ.get(self.embed_api_key_env)

    def validate(self, need_db=True, need_embed=True):
        """Fail-fast checks. Returns a list of human-readable errors ([] = OK)."""
        errs = []
        if need_db and not self.db_dsn:
            errs.append("PHASE3_DB_DSN is not set (postgresql://user:pass@host:port/db)")
        if not os.path.isdir(self.organized_root):
            errs.append(f"organized root not found: {self.organized_root}")
        if need_embed:
            if self.embed_backend not in ("nvidia_nim_api", "nvidia_selfhosted"):
                errs.append(f"EMBED_BACKEND must be nvidia_nim_api|nvidia_selfhosted, "
                            f"got {self.embed_backend!r}")
            if self.embed_backend == "nvidia_nim_api" and not self.api_key():
                errs.append(f"EMBED_BACKEND=nvidia_nim_api but ${self.embed_api_key_env} is unset")
            if self.embed_backend == "nvidia_selfhosted" and not self.embed_selfhosted_url:
                errs.append("EMBED_BACKEND=nvidia_selfhosted but EMBED_SELFHOSTED_URL is unset")
            if self.embed_dim <= 0:
                errs.append(f"EMBED_DIM must be positive, got {self.embed_dim}")
        if self.vector_index not in ("hnsw", "none"):
            errs.append(f"PHASE3_VECTOR_INDEX must be hnsw|none, got {self.vector_index!r}")
        return errs

    def describe(self) -> dict:
        """Redacted snapshot for the load report (no secrets, ever)."""
        return {
            "organized_root": self.organized_root,
            "db_dsn": _redact_dsn(self.db_dsn),
            "embed_backend": self.embed_backend,
            "embed_model": self.embed_model,
            "embed_dim": self.embed_dim,
            "embed_url": self.embed_url,
            "embed_input_type": self.embed_input_type,
            "embed_batch_size": self.embed_batch_size,
            "embed_api_key_set": bool(self.api_key()),
            "vector_index": self.vector_index,
        }


def _redact_dsn(dsn: str) -> str:
    """postgresql://user:secret@host/db -> postgresql://user:***@host/db"""
    if not dsn:
        return ""
    return re.sub(r"://([^:/@]+):([^@]*)@", r"://\1:***@", dsn)


def session_year(session: str):
    """'2020-feb-mar' -> 2020. None when no leading 4-digit year."""
    m = re.match(r"\s*(\d{4})", session or "")
    return int(m.group(1)) if m else None
