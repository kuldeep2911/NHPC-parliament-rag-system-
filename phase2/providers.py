"""
Provider interface — the SINGLE seam through which every model call passes.

A "provider" is a backend that can do any of the model-shaped operations the
pipeline needs. Selecting a backend is a single config switch:

    cfg.backend = "local"   -> LocalProvider   (functional; the only tested path)
    cfg.backend = "nvidia"  -> NvidiaProvider  (on-prem NIM/NeMo stub, air-gapped)

Every provider implements the SAME methods with the SAME data models, so moving
local -> nvidia changes ONLY the config value and nothing downstream:

    complete_json(system, user, schema_hint) -> dict     # LLM structuring
    ocr_image(image_bytes, lang)             -> str       # OCR a page image
    parse_visual(image_bytes, prompt)        -> dict      # VLM: text-in-image page
    embed(texts)                             -> list[list[float]]   # (future)

Design rules:
  * No model name or API key is hardcoded here — all come from cfg (which reads
    env). Fallback literals are avoided; a missing model raises a clear error.
  * Providers never crash the run: callers catch BackendError and route to review.
  * The DeterministicProvider (no network) backs the LLM path so the pipeline is
    always runnable/testable without any server. It is used automatically when the
    local LLM has no base_url configured.

This module supersedes the old llm.py backends; llm.get_backend delegates here for
backward compatibility.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request


class BackendError(Exception):
    """Raised by a provider when it cannot fulfil a call. Never crashes the run."""


class NotWiredError(BackendError):
    """Raised by stub methods that must be implemented on-site (e.g. air-gapped)."""


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response; tolerate code fences."""
    if not text:
        raise BackendError("empty response")
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise BackendError("no JSON object in response")


def _b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")


# ---------------------------------------------------------------------------
# Deterministic LLM path (no network) — passthrough of a precomputed result
# ---------------------------------------------------------------------------

class DeterministicLLM:
    """
    Rule-based LLM stand-in. The caller computes structure with rules and hands the
    assembled result here encoded as JSON under '__deterministic_result__', so the
    call signature is identical to a real model. Swapping to a model changes config
    only, not the extractor code.
    """
    name = "deterministic"

    def complete_json(self, system: str, user: str, schema_hint=None) -> dict:
        try:
            payload = json.loads(user)
        except Exception:
            raise BackendError("deterministic backend requires JSON payload")
        if "__deterministic_result__" in payload:
            return payload["__deterministic_result__"]
        raise BackendError("deterministic backend requires precomputed result")


# ---------------------------------------------------------------------------
# OLLAMA LLM — the local extraction model (llama3.2:3b), independent of the parser
# ---------------------------------------------------------------------------

class OllamaLLM:
    """
    LLM-only provider for the extraction pass, hitting a local Ollama server's
    OpenAI-compatible /v1/chat/completions. Model name and base URL are single
    config values (cfg.llm_model, cfg.ollama_base_url). Used regardless of which
    parser backend (Nemotron / Docling) is active, so Nemotron-parse + Ollama-LLM
    run together. Falls back to DeterministicLLM if no base URL is configured.
    """
    kind = "ollama"
    llm_is_deterministic = False

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = cfg.llm_model
        self.base_url = cfg.ollama_base_url or cfg.llm_base_url
        self.name = f"ollama:{self.model}" if self.base_url else "ollama:deterministic"
        self._det = DeterministicLLM()
        if not self.base_url:
            self.llm_is_deterministic = True

    def model_for(self, op: str) -> str:
        if op == "llm":
            return self.model if self.base_url else "deterministic"
        return "unknown"

    def complete_json(self, system: str, user: str, schema_hint=None) -> dict:
        if self.llm_is_deterministic:
            return self._det.complete_json(system, user, schema_hint)
        if not self.model:
            raise BackendError("NHPC_LLM_MODEL not set for ollama LLM backend")
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0, "stream": False,
            "format": "json",  # Ollama: constrain output to valid JSON
        }
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.cfg.llm_timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return extract_json(data["choices"][0]["message"]["content"])


class GroqLLM:
    """
    Cloud LLM via Groq's OpenAI-compatible API (llama-3.3-70b for the build phase).
    A 70B model groups multi-part Q&A far more reliably than the local 3B. Key from
    env only (cfg.groq_api_key_env). At deployment, switch llm_backend to a local/
    on-prem GPU running the same model — no downstream change.

    ⚠️ Build-phase only: document text is sent to Groq's cloud. Not for real NHPC data.
    """
    kind = "groq"
    llm_is_deterministic = False

    # Process-wide request timestamps (shared across instances) so the rate limit
    # is enforced globally, not per-object. A simple sliding-window limiter.
    _call_times = []

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = cfg.groq_model
        self.base_url = cfg.groq_base_url
        self.rpm = max(1, int(getattr(cfg, "groq_rpm", 30)))
        self.name = f"groq:{self.model}"

    def model_for(self, op: str) -> str:
        return self.model if op == "llm" else "unknown"

    def _key(self):
        k = self.cfg.groq_api_key()
        if not k:
            raise BackendError(
                f"llm_backend=groq needs an API key in env ${self.cfg.groq_api_key_env}")
        return k

    def _throttle(self):
        """
        Block until sending a request keeps us at/under groq_rpm in the last 60s.
        This CLIENT-SIDE pacing means we never hit Groq's 429, so every file is
        parsed by the LLM instead of falling back to deterministic.
        """
        import time
        while True:
            now = time.time()
            # drop timestamps older than 60s
            GroqLLM._call_times[:] = [t for t in GroqLLM._call_times if now - t < 60.0]
            if len(GroqLLM._call_times) < self.rpm:
                GroqLLM._call_times.append(now)
                return
            # wait until the oldest call ages out of the 60s window (+ small margin)
            wait = 60.0 - (now - GroqLLM._call_times[0]) + 0.25
            time.sleep(max(0.1, wait))

    def complete_json(self, system: str, user: str, schema_hint=None) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0, "stream": False,
            "response_format": {"type": "json_object"},  # OpenAI-style JSON mode
        }
        import time
        # Retry generously; a 429 WAITS (honoring Retry-After) and retries rather
        # than giving up — so a rate hit never causes a file to skip the LLM.
        for attempt in range(8):
            self._throttle()   # pace to stay under groq_rpm BEFORE each attempt
            req = urllib.request.Request(
                self.base_url.rstrip("/") + "/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self._key()}",
                         # Cloudflare blocks the default python-urllib UA (403/1010).
                         "User-Agent": "NHPC-parliament-rag/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.llm_timeout_s) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return extract_json(data["choices"][0]["message"]["content"])
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    ra = e.headers.get("Retry-After") if e.headers else None
                    delay = float(ra) if (ra and ra.replace(".", "").isdigit()) else 5.0
                    time.sleep(min(delay + 0.5, 65))
                    continue
                if e.code in (500, 502, 503) and attempt < 7:
                    time.sleep(2 * (attempt + 1)); continue
                raise BackendError(f"Groq request failed: HTTP {e.code}")
            except urllib.error.URLError as e:
                if attempt < 7:
                    time.sleep(2 * (attempt + 1)); continue
                raise BackendError(f"Groq request failed: {e}")
        raise BackendError("Groq request failed after retries (rate limit)")


class DeterministicLLMProvider:
    """Wrapper so a deterministic LLM backend has the same shape as OllamaLLM."""
    kind = "deterministic"
    llm_is_deterministic = True
    name = "deterministic"

    def __init__(self, cfg):
        self.cfg = cfg
        self._det = DeterministicLLM()

    def model_for(self, op):
        return "deterministic"

    def complete_json(self, system, user, schema_hint=None):
        return self._det.complete_json(system, user, schema_hint)


# ---------------------------------------------------------------------------
# LOCAL provider — OpenAI-compatible on-prem server (Ollama / vLLM / LM Studio)
# ---------------------------------------------------------------------------

class LocalProvider:
    """
    Functional provider for a locally-hosted, OpenAI-compatible inference server.

    LLM  : POST {base}/chat/completions
    VLM  : same endpoint with an image content part (multimodal models)
    OCR  : delegates to parse_visual with an OCR instruction unless a dedicated
           OCR server is configured (kept simple; Docling's bundled OCR handles
           the common scanned case in the reader).

    All of base_url / model / vision_model come from cfg (env-driven). If no LLM
    base_url is set, the LLM path uses DeterministicLLM so the pipeline still runs.
    """
    kind = "local"

    def __init__(self, cfg):
        self.cfg = cfg
        self._det = DeterministicLLM()
        self.llm_model = cfg.llm_model
        self.vision_model = cfg.vision_model or cfg.llm_model
        self.base_url = cfg.llm_base_url
        if self.base_url:
            self.name = f"local:{self.llm_model or 'default'}"
            self.llm_is_deterministic = False
        else:
            # no server configured -> deterministic LLM, but provider kind is still
            # 'local' so per-page OCR/visual routing labels remain consistent.
            self.name = "local:deterministic"
            self.llm_is_deterministic = True

    # --- LLM -------------------------------------------------------------
    def complete_json(self, system: str, user: str, schema_hint=None) -> dict:
        if self.llm_is_deterministic:
            return self._det.complete_json(system, user, schema_hint)
        if not self.llm_model:
            raise BackendError("NHPC_LLM_MODEL not set for local LLM backend")
        body = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "stream": False,
        }
        return extract_json(self._chat(body))

    def model_for(self, op: str) -> str:
        """Report the concrete model that WILL be used for an op (for the trace)."""
        if op == "llm":
            return "deterministic" if self.llm_is_deterministic else (self.llm_model or "unset")
        if op in ("visual", "ocr"):
            return self.vision_model or "unset"
        return "unknown"

    # --- OCR -------------------------------------------------------------
    def ocr_image(self, image_bytes: bytes, lang: str = "en,hi") -> str:
        # For a self-hosted OpenAI-compatible VLM, OCR == "transcribe verbatim".
        if not self.base_url or not self.vision_model:
            raise NotWiredError("local OCR needs base_url + vision_model")
        prompt = (f"Transcribe ALL text in this image verbatim in reading order. "
                  f"Languages may include {lang} (English/Hindi Devanagari). "
                  f"Output only the transcribed text, no commentary.")
        out = self._vision(prompt, image_bytes, self.vision_model)
        return out

    # --- VLM (text-in-image page) ---------------------------------------
    def parse_visual(self, image_bytes: bytes, prompt: str) -> dict:
        if not self.base_url or not self.vision_model:
            raise NotWiredError("local visual parse needs base_url + vision_model")
        text = self._vision(prompt, image_bytes, self.vision_model)
        try:
            return extract_json(text)
        except BackendError:
            # not JSON: return as a text block so caller can still use it
            return {"text": text}

    def embed(self, texts):
        raise NotWiredError("local embeddings not configured (not needed in Phase 2)")

    # --- transport -------------------------------------------------------
    def _chat(self, body) -> str:
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.cfg.llm_timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    def _vision(self, prompt: str, image_bytes: bytes, model: str) -> str:
        body = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{_b64(image_bytes)}"}},
                ],
            }],
            "temperature": 0,
            "stream": False,
        }
        return self._chat(body)


# ---------------------------------------------------------------------------
# NVIDIA provider — AIR-GAPPED, SELF-HOSTED stub (NIM / NeMo Retriever)
# ---------------------------------------------------------------------------

class NvidiaProvider:
    """
    On-prem, air-gapped NVIDIA NeMo Retriever NIM microservices.

    ── DO NOT use any cloud endpoint (integrate.api.nvidia.com). Everything runs on
       NHPC GPU servers with no internet and no data leaving the network. ──

    Three separate NIM microservices, each POST {url}/v1/infer with a base64
    image_url payload (see config.py for the per-service URLs):
      * OCR             nemoretriever-ocr           -> ocr_image()/parse_visual()
      * page-elements   nemoretriever-page-elements -> detect_page_elements()
      * table-structure nemoretriever-table-structure -> detect_table_structure()

    Request  : {"input":[{"type":"image_url","url":"data:image/png;base64,..."}]}
    OCR resp : data[].text_detections[].text_prediction.{text,confidence} + bbox
    Detect   : data[].bounding_boxes.{<class>:[{x_min,y_min,x_max,y_max,confidence}]}
               page-elements classes: table|chart|title|paragraph|header_footer
               table-structure classes: cell|row|column

    Auth: optional Bearer token from env[cfg.nvidia_token_env]; a self-hosted NIM
    may need none, in which case no Authorization header is sent.

    complete_json (LLM) is intentionally NOT served by these OCR/detection NIMs.
    Keep the LLM on the local Ollama backend, or point NVIDIA_MODEL at a separate
    on-prem LLM NIM and implement complete_json against its OpenAI-compatible API.
    """
    kind = "nvidia"

    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = (cfg.nvidia_mode or "cloud").lower()
        self.ocr_url, self.page_elements_url, self.table_structure_url = cfg.nvidia_urls()
        self.llm_model = cfg.nvidia_model
        self.vision_model = cfg.nvidia_vision_model or cfg.nvidia_ocr_model
        self.name = f"nvidia-{self.mode}:{cfg.nvidia_ocr_model}"

    def _token(self, service):
        """
        Bearer token for a service (ocr|page_elements|table_structure). In cloud
        mode this is the build.nvidia.com key (per-model key wins, shared fallback);
        on-prem it is the optional self-hosted token.
        """
        if self.mode == "cloud":
            key = self.cfg.nvidia_api_key(service)
            if not key:
                raise BackendError(
                    f"NVIDIA cloud mode needs an API key for '{service}' in env "
                    f"(per-model NVIDIA_*_API_KEY or shared "
                    f"${self.cfg.nvidia_api_key_env}; build.nvidia.com nvapi-...).")
            return key
        import os
        return os.environ.get(self.cfg.nvidia_token_env) if self.cfg.nvidia_token_env else None

    def model_for(self, op: str) -> str:
        if op == "llm":
            return self.llm_model or "unset"
        if op == "ocr" or op == "visual":
            return self.cfg.nvidia_ocr_model
        if op == "page_elements":
            return "nemoretriever-page-elements"
        if op == "table_structure":
            return "nemoretriever-table-structure"
        return "unknown"

    # --- shared transport (hosted cloud or on-prem NIM) -----------------
    def _infer(self, url: str, image_bytes: bytes, service: str) -> dict:
        """POST one image to a /v1/infer endpoint and return the JSON dict."""
        if not url:
            raise NotWiredError("endpoint URL not configured")
        body = {"input": [{"type": "image_url",
                           "url": f"data:image/png;base64,{_b64(image_bytes)}"}]}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        token = self._token(service)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        # Retry with exponential backoff on throttling / transient errors (the
        # hosted NVIDIA API rate-limits). Honours Retry-After when present.
        import time
        payload = json.dumps(body).encode("utf-8")
        last = None
        for attempt in range(4):
            req = urllib.request.Request(url, data=payload, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.llm_timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (429, 500, 502, 503, 504):
                    retry_after = e.headers.get("Retry-After") if e.headers else None
                    delay = float(retry_after) if retry_after and retry_after.isdigit() \
                        else (1.5 * (2 ** attempt))
                    time.sleep(min(delay, 30))
                    continue
                raise BackendError(f"NIM request to {url} failed: HTTP {e.code}")
            except urllib.error.URLError as e:
                last = e
                time.sleep(1.5 * (2 ** attempt))
        raise BackendError(f"NIM request to {url} failed after retries: {last}")

    # --- OCR (NIM or hosted cloud) --------------------------------------
    def ocr_image(self, image_bytes: bytes, lang: str = "en,hi") -> str:
        """OCR a page image; return text in top-to-bottom reading order."""
        data = self._infer(self.ocr_url, image_bytes, "ocr")
        return _nim_ocr_to_text(data)

    def parse_visual(self, image_bytes: bytes, prompt: str) -> dict:
        """Image-based page: OCR transcribes it; return {"text": ...}."""
        return {"text": self.ocr_image(image_bytes)}

    # --- object detection (page elements / table structure) -------------
    def detect_page_elements(self, image_bytes: bytes) -> dict:
        """Return page-element boxes: {table|chart|title|paragraph|header_footer:[...]}."""
        data = self._infer(self.page_elements_url, image_bytes, "page_elements")
        return _nim_boxes(data)

    def detect_table_structure(self, image_bytes: bytes) -> dict:
        """Return table-structure boxes: {cell|row|column:[...]}."""
        data = self._infer(self.table_structure_url, image_bytes, "table_structure")
        return _nim_boxes(data)

    def complete_json(self, system: str, user: str, schema_hint=None) -> dict:
        raise NotWiredError(
            "The NeMo Retriever OCR/detection NIMs do not serve an LLM. Keep the "
            "LLM on the local Ollama backend, or point NVIDIA_MODEL at a separate "
            "on-prem LLM NIM and implement complete_json against it.")

    def embed(self, texts):
        raise NotWiredError(
            "Embeddings are a later retrieval-phase concern; point at an on-prem "
            "embedding NIM when needed.")


def _nim_ocr_to_text(data: dict) -> str:
    """Assemble OCR NIM response into reading-order text (sorted by y then x)."""
    out_lines = []
    for item in (data or {}).get("data", []):
        dets = item.get("text_detections", [])
        rows = []
        for d in dets:
            text = (d.get("text_prediction") or {}).get("text", "")
            pts = ((d.get("bounding_box") or {}).get("points") or [])
            y = min((p.get("y", 0.0) for p in pts), default=0.0)
            x = min((p.get("x", 0.0) for p in pts), default=0.0)
            if text:
                rows.append((round(y, 3), x, text))
        rows.sort(key=lambda r: (r[0], r[1]))
        out_lines.extend(t for _, _, t in rows)
    return "\n".join(out_lines)


def _nim_boxes(data: dict) -> dict:
    """Extract the {class: [boxes]} mapping from an object-detection NIM response."""
    for item in (data or {}).get("data", []):
        if "bounding_boxes" in item:
            return item["bounding_boxes"]
    return {}


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def get_provider(cfg):
    """Return the provider for cfg.backend. This is the single local<->nvidia seam."""
    backend = (cfg.backend or "local").lower()
    if backend == "nvidia":
        return NvidiaProvider(cfg)
    if backend == "local":
        return LocalProvider(cfg)
    raise BackendError(f"unknown backend '{backend}' (expected 'local' or 'nvidia')")


def get_parser(cfg):
    """
    Parser provider for document parsing/OCR/table extraction. Independent of the
    LLM backend. 'nemotron' -> NvidiaProvider (hosted or on-prem NIMs); 'docling' ->
    None (the reader uses its built-in Docling/TableFormer path when no NIM provider).
    """
    pb, _ = cfg.resolve_backends()
    if pb == "nemotron":
        return NvidiaProvider(cfg)
    return None  # docling: reader's built-in path, no external provider needed


def get_llm(cfg):
    """LLM provider for the extraction pass, independent of the parser backend."""
    _, lb = cfg.resolve_backends()
    if lb == "groq":
        return GroqLLM(cfg)
    if lb == "ollama":
        return OllamaLLM(cfg)
    if lb == "deterministic":
        return DeterministicLLMProvider(cfg)
    # a bare OpenAI-compatible server via the old local path
    return LocalProvider(cfg)
