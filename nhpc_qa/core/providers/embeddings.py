"""
Embedding provider interface — one contract, two backends, chosen by config.

    get_embedder(cfg).embed_passages([...]) -> list[list[float]]

  EMBED_BACKEND=nvidia_nim_api     NVIDIA-hosted NIM. Dev/now. Text LEAVES the network.
  EMBED_BACKEND=nvidia_selfhosted  On-prem model / self-hosted NIM. Server/later.
                                   Nothing leaves the NHPC network.

Switching is a CONFIG change only -- identical input/output contract, no caller change.
Keys and URLs come from env; nothing is hardcoded.

WHAT IS EMBEDDED: only sub_question.question_text (parsed.json declares
embedding_unit = 'sub_question.question_text'). Answers, tables and annexures are
DISPLAY PAYLOAD fetched after a question matches -- never embedded.

MODEL: nvidia/llama-nemotron-embed-1b-v2. The model the spec named
(llama-3.2-nv-embedqa-1b-v2) reached END OF LIFE on 2026-05-18 and now returns HTTP
410 Gone. Measured facts for the replacement:
    dim 2048 | output L2-normalised => COSINE | Devanagari OK | passage/query modes

INPUT TYPE: 'passage' when indexing sub-questions (here). Phase-4 search must embed the
user's query with input_type='query' -- the model is asymmetric and mixing the two
degrades retrieval.

FAIL-FAST: a vector whose length != cfg.embed_dim raises EmbeddingError rather than
being written, so a model swap can never silently corrupt the index.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class EmbeddingError(RuntimeError):
    pass


class _Base:
    """Common contract. Subclasses implement _embed(texts, input_type)."""

    name = "base"

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = cfg.embed_model
        self.dim = int(cfg.embed_dim)

    # -- public API ---------------------------------------------------------
    def embed_passages(self, texts):
        """Embed sub-question text for INDEXING (input_type=passage)."""
        return self._embed_checked(texts, "passage")

    def embed_queries(self, texts):
        """Embed a user query for SEARCH (input_type=query). Phase 4 uses this."""
        return self._embed_checked(texts, "query")

    # -- internals ----------------------------------------------------------
    def _embed_checked(self, texts, input_type):
        if not texts:
            return []
        vecs = self._embed(list(texts), input_type)
        if len(vecs) != len(texts):
            raise EmbeddingError(
                f"{self.name}: asked for {len(texts)} vectors, got {len(vecs)}")
        for i, v in enumerate(vecs):
            if len(v) != self.dim:
                raise EmbeddingError(
                    f"{self.name}: vector {i} has dim {len(v)} but the column is "
                    f"vector({self.dim}). Model/config mismatch — refusing to write. "
                    f"(model={self.model})")
        return vecs

    def _embed(self, texts, input_type):
        raise NotImplementedError


class NvidiaNimApiEmbedder(_Base):
    """
    NVIDIA-hosted NIM (integrate.api.nvidia.com), OpenAI-compatible /v1/embeddings.

    Build/dev phase only: the sub-question text is sent to NVIDIA's cloud. For real
    NHPC data on the server, switch EMBED_BACKEND=nvidia_selfhosted -- same contract.
    """

    name = "nvidia_nim_api"

    def _key(self):
        k = self.cfg.api_key()
        if not k:
            raise EmbeddingError(
                f"EMBED_BACKEND=nvidia_nim_api needs ${self.cfg.embed_api_key_env}")
        return k

    def _embed(self, texts, input_type):
        body = {
            "input": texts,
            "model": self.model,
            "input_type": input_type,      # passage (index) | query (search)
            "encoding_format": "float",
            "truncate": "END",             # never fail on an over-long question
        }
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        last = None
        for attempt in range(self.cfg.embed_max_retries):
            req = urllib.request.Request(
                self.cfg.embed_url, data=payload,
                headers={"Content-Type": "application/json",
                         "Accept": "application/json",
                         "Authorization": f"Bearer {self._key()}",
                         "User-Agent": "NHPC-parliament-rag/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.embed_timeout_s) as r:
                    data = json.loads(r.read().decode("utf-8"))
                # the API may return items out of order; sort by index to be safe
                items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
                return [it["embedding"] for it in items]
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:200]
                except Exception:
                    pass
                last = f"HTTP {e.code} {detail}"
                if e.code == 410:
                    # model retired — retrying cannot help, and a silent fallback
                    # would corrupt the index with vectors from a different model.
                    raise EmbeddingError(
                        f"model {self.model!r} is GONE (HTTP 410): {detail}. "
                        f"Pick a live model (GET /v1/models) and re-embed with --force.")
                if e.code == 429 or 500 <= e.code < 600:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise EmbeddingError(f"{self.name}: {last}")
            except urllib.error.URLError as e:
                last = f"{type(e).__name__}: {e}"
                time.sleep(min(2 ** attempt, 30))
            except Exception as e:      # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
                time.sleep(min(2 ** attempt, 30))
        raise EmbeddingError(f"{self.name}: failed after "
                             f"{self.cfg.embed_max_retries} attempts — {last}")


class NvidiaSelfHostedEmbedder(_Base):
    """
    On-prem embedder for the NHPC server. NOTHING leaves the network.

    Two ways to run it; both satisfy this same contract, so nothing else changes:

    A) Self-hosted NIM container (recommended — identical API to the cloud):
           docker run --gpus all -p 8000:8000 \
             nvcr.io/nim/nvidia/llama-nemotron-embed-1b-v2:latest
       then set:
           EMBED_BACKEND=nvidia_selfhosted
           EMBED_SELFHOSTED_URL=http://localhost:8000/v1/embeddings
       The request/response shape is the same OpenAI-style /v1/embeddings, so this
       class simply POSTs to that URL (implemented below).

    B) Hugging Face weights loaded in-process (no container). Download the model to
       EMBED_SELFHOSTED_MODEL_PATH on a machine with internet, copy it to the server,
       and implement _embed with sentence-transformers:

           from sentence_transformers import SentenceTransformer
           m = SentenceTransformer(cfg.embed_selfhosted_model_path, trust_remote_code=True)
           return m.encode(texts, prompt_name=input_type, normalize_embeddings=True).tolist()

       This is left as a documented stub -- it needs the GPU box to validate, and the
       dim/normalisation must be re-measured there (the fail-fast dim check in _Base
       will catch a mismatch before anything is written).

    Whichever you use, the model must be the SAME one the vectors were built with, or
    re-embed everything with --force: vectors from different models are not comparable.
    """

    name = "nvidia_selfhosted"

    def _embed(self, texts, input_type):
        url = self.cfg.embed_selfhosted_url
        if not url:
            raise EmbeddingError(
                "EMBED_BACKEND=nvidia_selfhosted requires EMBED_SELFHOSTED_URL "
                "(a self-hosted NIM at http://host:8000/v1/embeddings), or implement "
                "the sentence-transformers path documented in this class.")
        body = {
            "input": texts,
            "model": self.model,
            "input_type": input_type,
            "encoding_format": "float",
            "truncate": "END",
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "User-Agent": "NHPC-parliament-rag/1.0"}
        token = self.cfg.api_key()          # optional on-prem bearer token
        if token:
            headers["Authorization"] = f"Bearer {token}"
        last = None
        for attempt in range(self.cfg.embed_max_retries):
            req = urllib.request.Request(
                url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.embed_timeout_s) as r:
                    data = json.loads(r.read().decode("utf-8"))
                items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
                return [it["embedding"] for it in items]
            except Exception as e:          # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
                time.sleep(min(2 ** attempt, 30))
        raise EmbeddingError(f"{self.name}: failed after "
                             f"{self.cfg.embed_max_retries} attempts — {last}")


def get_embedder(cfg):
    """Config-selected embedder. Adding a backend = adding a class here."""
    backend = (cfg.embed_backend or "").strip()
    if backend == "nvidia_nim_api":
        return NvidiaNimApiEmbedder(cfg)
    if backend == "nvidia_selfhosted":
        return NvidiaSelfHostedEmbedder(cfg)
    raise EmbeddingError(
        f"unknown EMBED_BACKEND {backend!r} (nvidia_nim_api | nvidia_selfhosted)")
