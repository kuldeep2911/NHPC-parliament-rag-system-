"""
The provider registry — ONE place that hands out every model client.

There used to be four factories scattered across three phases:

    phase2/providers.py     get_parser(cfg), get_llm(cfg)
    phase3/embeddings.py    get_embedder(cfg)
    phase4/rerank/...       get_reranker(cfg)

They are now all reachable from here. The implementations are UNCHANGED -- same classes,
same backends, same request/response contracts, same fail-fast dim check. This module only
gives them one front door, so a caller never has to know which phase a model client
happened to be born in.

    from nhpc_qa.core.providers import get_llm, get_embedder, get_reranker, get_parser

Every factory takes the single `Settings` object (nhpc_qa.config). Backend selection stays
config-only:

    parser    NHPC_PARSER_BACKEND   nemotron | docling
    llm       NHPC_LLM_BACKEND      gemini | ollama | groq | deterministic
    embedder  EMBED_BACKEND         nvidia_nim_api | nvidia_selfhosted
    reranker  RERANK_BACKEND        nvidia_nim_api | nvidia_selfhosted

Switching any of them to an on-prem/self-hosted backend is a config change, never a code
change -- which is the whole point of the seam.
"""

from __future__ import annotations

from nhpc_qa.core.providers.embeddings import EmbeddingError, get_embedder
from nhpc_qa.core.providers.models import (
    BackendError,
    NotWiredError,
    get_llm,
    get_parser,
)
from nhpc_qa.core.providers.rerank import RerankError, get_reranker

__all__ = [
    "get_parser", "get_llm", "get_embedder", "get_reranker",
    "BackendError", "NotWiredError", "EmbeddingError", "RerankError",
]
