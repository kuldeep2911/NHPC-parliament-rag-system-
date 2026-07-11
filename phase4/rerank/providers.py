"""
Reranker provider interface — one contract, two backends, chosen by config.

    get_reranker(cfg).rerank(query, passages) -> [(index, logit), ...]  best first

  RERANK_BACKEND=nvidia_nim_api     NVIDIA-hosted NIM. Dev/now. Text LEAVES the network.
  RERANK_BACKEND=nvidia_selfhosted  On-prem NIM. Server/later. Nothing leaves.

Same pattern as phase3/embeddings.py: identical input/output contract, so switching is
a CONFIG change only. LangGraph does NOT own this call -- a graph node calls it.

MODEL: nvidia/llama-nemotron-rerank-1b-v2.
  The model the spec named (nvidia/nv-rerankqa-mistral-4b-v3) DOES NOT EXIST -> HTTP 404,
  and nvidia/llama-3.2-nv-rerankqa-1b-v2 is END OF LIFE -> HTTP 410. Rerankers are not
  listed by GET /v1/models; they are served from ai.api.nvidia.com on a per-model path.

  Measured live before adopting: for "electricity dues owed by J&K power departments" it
  ranked the dues passage top and unrelated R&D funding last, and it is strongly
  MULTILINGUAL -- it placed a Hindi passage ABOVE its English equivalent. That matters:
  the graph must never restrict retrieval by language, and this model is why we can rely
  on cross-lingual matching at the rerank stage.

CONTRACT:
    request  {"model":..., "query":{"text":...}, "passages":[{"text":...}, ...],
              "truncate":"END"}
    response {"rankings":[{"index": i, "logit": f}, ...]}   already sorted best-first
  `logit` is an UNBOUNDED relevance score (roughly -20..+5 observed), NOT a probability.
  Do not present it to officers as a confidence percentage.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class RerankError(RuntimeError):
    pass


class _Base:
    name = "base"

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = cfg.rerank_model

    def rerank(self, query: str, passages: list[str]):
        """
        Score `passages` against `query`.

        Returns [(original_index, logit), ...] sorted best-first. The caller maps the
        index back onto its own candidate list -- we never reorder the caller's objects
        for it, so there is no chance of a silent misalignment.
        """
        if not passages:
            return []
        out = self._rerank(query, passages)
        seen = {i for i, _ in out}
        if len(out) != len(passages) or seen != set(range(len(passages))):
            raise RerankError(
                f"{self.name}: expected a ranking over all {len(passages)} passages, "
                f"got {len(out)} entries covering {len(seen)} distinct indices")
        return out

    def _rerank(self, query, passages):
        raise NotImplementedError

    # -- shared HTTP ---------------------------------------------------------
    def _post(self, url, headers, query, passages):
        body = {
            "model": self.model,
            "query": {"text": query},
            "passages": [{"text": p} for p in passages],
            "truncate": "END",          # never fail on a long passage
        }
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        last = None
        for attempt in range(self.cfg.rerank_max_retries):
            req = urllib.request.Request(url, data=payload, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.rerank_timeout_s) as r:
                    data = json.loads(r.read().decode("utf-8"))
                rankings = data.get("rankings")
                if not isinstance(rankings, list):
                    raise RerankError(f"{self.name}: no 'rankings' in response")
                return [(int(it["index"]), float(it["logit"])) for it in rankings]
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:200]
                except Exception:
                    pass
                last = f"HTTP {e.code} {detail}"
                if e.code in (404, 410):
                    # wrong/retired model — retrying cannot help, and silently ranking
                    # by something else would corrupt the officer's results.
                    raise RerankError(
                        f"reranker {self.model!r} is unavailable ({last}). The NIM "
                        f"catalogue does not list rerankers; check the per-model URL.")
                if e.code == 429 or 500 <= e.code < 600:
                    time.sleep(min(2 ** attempt, 20))
                    continue
                raise RerankError(f"{self.name}: {last}")
            except Exception as e:      # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
                time.sleep(min(2 ** attempt, 20))
        raise RerankError(f"{self.name}: failed after "
                          f"{self.cfg.rerank_max_retries} attempts — {last}")


class NvidiaNimApiReranker(_Base):
    """NVIDIA-hosted reranking NIM (build/dev; passages leave the network)."""

    name = "nvidia_nim_api"

    def _rerank(self, query, passages):
        key = self.cfg.rerank_api_key()
        if not key:
            raise RerankError(
                f"RERANK_BACKEND=nvidia_nim_api needs ${self.cfg.rerank_api_key_env}")
        return self._post(
            self.cfg.rerank_url,
            {"Content-Type": "application/json", "Accept": "application/json",
             "Authorization": f"Bearer {key}", "User-Agent": "NHPC-parliament-rag/1.0"},
            query, passages)


class NvidiaSelfHostedReranker(_Base):
    """
    On-prem reranking NIM. NOTHING leaves the NHPC network.

        docker run --gpus all -p 8001:8000 \
          nvcr.io/nim/nvidia/llama-nemotron-rerank-1b-v2:latest
        RERANK_BACKEND=nvidia_selfhosted
        RERANK_SELFHOSTED_URL=http://localhost:8001/v1/ranking

    The container speaks the same request/response shape, so this class only changes the
    URL and (optionally) the auth header -- nothing else in the pipeline moves.
    """

    name = "nvidia_selfhosted"

    def _rerank(self, query, passages):
        url = self.cfg.rerank_selfhosted_url
        if not url:
            raise RerankError(
                "RERANK_BACKEND=nvidia_selfhosted requires RERANK_SELFHOSTED_URL")
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "User-Agent": "NHPC-parliament-rag/1.0"}
        token = self.cfg.rerank_api_key()      # optional on-prem bearer token
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return self._post(url, headers, query, passages)


def get_reranker(cfg):
    """Config-selected reranker. Adding a backend = adding a class here."""
    backend = (cfg.rerank_backend or "").strip()
    if backend == "nvidia_nim_api":
        return NvidiaNimApiReranker(cfg)
    if backend == "nvidia_selfhosted":
        return NvidiaSelfHostedReranker(cfg)
    raise RerankError(
        f"unknown RERANK_BACKEND {backend!r} (nvidia_nim_api | nvidia_selfhosted)")
