"""
Backward-compatible shim over the provider interface.

The model seam now lives in providers.py (LocalProvider / NvidiaProvider), which
covers LLM + OCR + VLM behind one config switch (cfg.backend = local|nvidia).
This module re-exports the pieces the rest of the pipeline imported historically
so existing imports keep working:

    from .llm import BackendError, get_backend

`get_backend(cfg)` returns the selected provider; the provider exposes
`.name`, `.complete_json(...)`, `.ocr_image(...)`, `.parse_visual(...)`.
"""

from __future__ import annotations

from .providers import (  # noqa: F401  (re-exported for compatibility)
    BackendError,
    NotWiredError,
    get_provider,
    extract_json as _extract_json,
)


def get_backend(cfg):
    """Compatibility alias — returns the provider for cfg.backend."""
    return get_provider(cfg)
