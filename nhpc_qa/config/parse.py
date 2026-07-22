"""
Configuration for the Phase-2 parsing pipeline.

All tunables live here or come from environment variables. No secrets are ever
hardcoded — API keys are read from the environment only.

Precedence: explicit kwargs > environment variable > default below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def load_dotenv(path=".env"):
    """
    Minimal .env loader: set KEY=VALUE lines into os.environ if not already set.
    No dependency on python-dotenv. Existing env vars win (never overwritten).
    Secret values are never printed or logged. Silently no-ops if the file is
    absent. Call once at startup before load_config().
    """
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except FileNotFoundError:
        pass


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and v.strip() else default


@dataclass
class Config:
    # --- I/O ---
    organized_root: str = "organized"
    reports_subdir: str = "_reports"
    parsed_filename: str = "parsed.json"

    # --- provider backend: the SINGLE local<->nvidia switch --------------
    # "local"  -> on-prem OpenAI-compatible server (Ollama/vLLM/LM Studio); the
    #             only tested path. Falls back to a deterministic (no-network) LLM
    #             when no llm_base_url is set, so the pipeline always runs.
    # "nvidia" -> air-gapped, self-hosted NVIDIA NIM/NeMo stub (see providers.py).
    backend: str = field(default_factory=lambda: _env_str("NHPC_BACKEND", "local"))

    # Independent backend selection (spec: Nemotron parse + Ollama LLM together):
    #   parser_backend : "nemotron" (hosted/on-prem NIMs) | "docling" (CPU fallback)
    #   llm_backend    : "ollama" (local) | "groq" (cloud 70B, build phase) |
    #                    "deterministic" (no-network)
    # Empty -> derived from `backend` for backward-compat.
    parser_backend: str = field(default_factory=lambda: _env_str("NHPC_PARSER_BACKEND", ""))
    llm_backend: str = field(default_factory=lambda: _env_str("NHPC_LLM_BACKEND", ""))

    # --- GROQ (OpenAI-compatible cloud LLM; llama3.3-70b) -------------------------
    # Superseded by Gemini for the build phase (Groq's 30 rpm free tier was the
    # bottleneck), but kept as a fully working backend. Key from env ONLY.
    groq_base_url: str = field(default_factory=lambda: _env_str("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
    groq_model: str = field(default_factory=lambda: _env_str("GROQ_MODEL", "llama-3.3-70b-versatile"))
    groq_api_key_env: str = field(default_factory=lambda: _env_str("GROQ_API_KEY_ENV", "GROQ_API_KEY"))
    # Client-side rate limit so we never hit Groq's 429 (default free tier is
    # 30 req/min). The GroqLLM provider paces calls to stay under this, so EVERY
    # file is parsed by the LLM instead of falling back to deterministic on 429.
    groq_rpm: int = field(default_factory=lambda: int(_env_str("GROQ_RPM", "30")))

    # --- GEMINI (build-phase cloud LLM; Google AI Studio) -------------------------
    # Uses Google's OpenAI-COMPATIBLE endpoint so GeminiLLM reuses the same
    # chat/completions + JSON-mode contract as Groq and Ollama; nothing downstream
    # changes when the backend switches.
    #
    # Build-phase only: document text is sent to Google's cloud. Not for real NHPC
    # data. At deployment set llm_backend=ollama with an on-prem Qwen3 14B.
    gemini_base_url: str = field(default_factory=lambda: _env_str(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"))
    gemini_model: str = field(default_factory=lambda: _env_str("GEMINI_MODEL", "gemini-2.5-flash"))
    gemini_api_key_env: str = field(default_factory=lambda: _env_str("GEMINI_API_KEY_ENV", "GEMINI_API_KEY"))
    # AI Studio's paid tier allows far more than Groq's free 30/min; pace anyway so a
    # burst never 429s and silently drops a file to the deterministic fallback.
    gemini_rpm: int = field(default_factory=lambda: int(_env_str("GEMINI_RPM", "60")))
    # Gemini 2.5 Flash is a THINKING model: it can spend the entire output budget on
    # internal reasoning and return an empty message. Span extraction is extraction,
    # not reasoning, so thinking is off by default. Raise it if you ever want it.
    gemini_thinking_budget: int = field(default_factory=lambda: int(_env_str("GEMINI_THINKING_BUDGET", "0")))
    gemini_max_tokens: int = field(default_factory=lambda: int(_env_str("GEMINI_MAX_TOKENS", "8192")))

    # --- LOCAL provider settings (OpenAI-compatible on-prem server) -------
    # Local Ollama LLM model name — a SINGLE config value, never hardcoded in logic.
    # On CPU, default to a small 3B-class text model. (Note: Llama 3.2 is 1B/3B text
    # or 11B/90B vision; 8B is Llama 3.1 — so 8B is NOT a valid 3.2 tag.)
    llm_model: str = field(default_factory=lambda: _env_str("NHPC_LLM_MODEL", "llama3.2:3b"))
    # Vision/VLM model for parse_visual + OCR (multimodal). Optional; defaults to a
    # small vision model. Empty means "visual routing unavailable -> flag".
    vision_model: str = field(default_factory=lambda: _env_str("NHPC_VISION_MODEL", ""))
    # OpenAI-compatible base URL for the local server. Empty -> deterministic LLM.
    llm_base_url: str = field(default_factory=lambda: _env_str("NHPC_LLM_BASE_URL", ""))
    # Ollama base URL for the LLM extraction pass (OpenAI-compatible /v1). Default is
    # the standard local Ollama port; single config value, model from llm_model.
    ollama_base_url: str = field(default_factory=lambda: _env_str("NHPC_OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    llm_max_retries: int = 1  # one stricter retry on invalid JSON, then review
    llm_timeout_s: int = field(default_factory=lambda: int(_env_str("NHPC_LLM_TIMEOUT_S", "120")))
    # Ollama-style server-side JSON mode ("format":"json"). Ollama honours it; a self-hosted
    # NVIDIA NIM (e.g. Nemotron Super 49B) may reject the field. Default on for Ollama; set
    # NHPC_LLM_JSON_MODE=0 when pointing NHPC_LLM_BASE_URL at a NIM. The extractor already
    # tolerates prose-wrapped JSON, so turning it off never breaks parsing.
    llm_json_mode: bool = field(default_factory=lambda: _env_bool("NHPC_LLM_JSON_MODE", True))
    # Run the LLM as a SECOND-OPINION cross-check on every prose file: it counts
    # questions + distinct answers and we flag llm_crosscheck_disagree if it differs
    # from the deterministic split. The deterministic result is kept either way.
    llm_crosscheck: bool = field(default_factory=lambda: _env_bool("NHPC_LLM_CROSSCHECK", False))
    # LLM decides the question<->answer GROUPING as primary (best with a capable
    # model like groq llama-3.3-70b); deterministic splitter is the fallback.
    llm_grouping: bool = field(default_factory=lambda: _env_bool("NHPC_LLM_GROUPING", False))

    # --- NVIDIA NeMo Retriever (OCR + page-elements + table-structure) ----
    # Two deployment modes, switched by NVIDIA_MODE (default "cloud" for the build
    # phase; set "onprem" at NHPC to hit self-hosted NIMs with no code change):
    #   cloud  -> NVIDIA-hosted endpoints on ai.api.nvidia.com, Bearer nvapi-... key
    #             (build.nvidia.com). Document content leaves the network — build/test
    #             only; switch to onprem for production.
    #   onprem -> self-hosted NIM microservices, each POST {url}/v1/infer, optional
    #             Bearer token; nothing leaves the NHPC network.
    # Every URL is overridable via env so you can paste the exact endpoint per model.
    nvidia_mode: str = field(default_factory=lambda: _env_str("NVIDIA_MODE", "cloud"))

    # API keys: read from env ONLY (never stored/logged). build.nvidia.com nvapi-...
    # You may have a SEPARATE key per model (one per build.nvidia.com endpoint) OR a
    # single shared key. Per-model env vars take precedence; the shared one is the
    # fallback. Set whichever you have.
    nvidia_api_key_env: str = field(default_factory=lambda: _env_str("NVIDIA_API_KEY_ENV", "NVIDIA_API_KEY"))
    nvidia_ocr_key_env: str = field(default_factory=lambda: _env_str("NVIDIA_OCR_KEY_ENV", "NVIDIA_OCR_API_KEY"))
    nvidia_page_elements_key_env: str = field(default_factory=lambda: _env_str("NVIDIA_PAGE_ELEMENTS_KEY_ENV", "NVIDIA_PAGE_ELEMENTS_API_KEY"))
    nvidia_table_structure_key_env: str = field(default_factory=lambda: _env_str("NVIDIA_TABLE_STRUCTURE_KEY_ENV", "NVIDIA_TABLE_STRUCTURE_API_KEY"))

    # Hosted-cloud endpoint URLs (build phase). OCR path is confirmed; the two
    # detection paths follow the ai.api.nvidia.com/v1/cv/nvidia/<model> pattern —
    # override with the exact URL from each model's build.nvidia.com page.
    nvidia_cloud_ocr_url: str = field(default_factory=lambda: _env_str(
        "NVIDIA_CLOUD_OCR_URL", "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr"))
    nvidia_cloud_page_elements_url: str = field(default_factory=lambda: _env_str(
        "NVIDIA_CLOUD_PAGE_ELEMENTS_URL", "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v2"))
    nvidia_cloud_table_structure_url: str = field(default_factory=lambda: _env_str(
        "NVIDIA_CLOUD_TABLE_STRUCTURE_URL", "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1"))

    # On-prem NIM endpoint URLs (production; each a separate microservice).
    nvidia_ocr_url: str = field(default_factory=lambda: _env_str("NVIDIA_OCR_URL", "http://localhost:8010/v1/infer"))
    nvidia_page_elements_url: str = field(default_factory=lambda: _env_str("NVIDIA_PAGE_ELEMENTS_URL", "http://localhost:8011/v1/infer"))
    nvidia_table_structure_url: str = field(default_factory=lambda: _env_str("NVIDIA_TABLE_STRUCTURE_URL", "http://localhost:8012/v1/infer"))

    nvidia_base_url: str = field(default_factory=lambda: _env_str("NVIDIA_BASE_URL", "http://localhost:8000/v1"))
    nvidia_model: str = field(default_factory=lambda: _env_str("NVIDIA_MODEL", ""))
    nvidia_vision_model: str = field(default_factory=lambda: _env_str("NVIDIA_VISION_MODEL", ""))
    nvidia_ocr_model: str = field(default_factory=lambda: _env_str("NVIDIA_OCR_MODEL", "nvidia/nemoretriever-ocr-v1"))
    # On-prem token env var name (optional; a self-hosted NIM may need none).
    nvidia_token_env: str = field(default_factory=lambda: _env_str("NVIDIA_TOKEN_ENV", "NVIDIA_TOKEN"))

    def nvidia_api_key(self, service: str = None):
        """
        build.nvidia.com API key for a service (cloud mode). Per-model key wins,
        shared NVIDIA_API_KEY is the fallback. service in {ocr, page_elements,
        table_structure}. Returns None if none set.
        """
        per = {
            "ocr": self.nvidia_ocr_key_env,
            "page_elements": self.nvidia_page_elements_key_env,
            "table_structure": self.nvidia_table_structure_key_env,
        }.get(service)
        if per and os.environ.get(per):
            return os.environ[per]
        return os.environ.get(self.nvidia_api_key_env)

    def groq_api_key(self):
        """Groq API key from env (build phase). None if unset."""
        return os.environ.get(self.groq_api_key_env)

    def gemini_api_key(self):
        """
        Gemini API key from env (build phase). None if unset.
        Accepts GOOGLE_API_KEY as a fallback — Google AI Studio's own docs use both
        names — so a key pasted under either variable just works.
        """
        return (os.environ.get(self.gemini_api_key_env)
                or os.environ.get("GOOGLE_API_KEY"))

    def langfuse_keys(self):
        """(public_key, secret_key) from env. Either may be None. Read only when
        langfuse_enabled; never logged."""
        return (os.environ.get(self.langfuse_public_key_env),
                os.environ.get(self.langfuse_secret_key_env))

    def nvidia_urls(self):
        """Return (ocr_url, page_elements_url, table_structure_url) for the mode."""
        if (self.nvidia_mode or "cloud").lower() == "cloud":
            return (self.nvidia_cloud_ocr_url, self.nvidia_cloud_page_elements_url,
                    self.nvidia_cloud_table_structure_url)
        return (self.nvidia_ocr_url, self.nvidia_page_elements_url,
                self.nvidia_table_structure_url)

    def resolve_backends(self):
        """
        Resolve (parser_backend, llm_backend). Explicit config wins; otherwise
        derive from the legacy single `backend` for backward-compat:
          backend=nvidia -> parser=nemotron, llm=ollama
          backend=local  -> parser=docling,  llm=ollama (if base_url) else deterministic
          backend=deterministic -> parser=docling, llm=deterministic
        """
        pb = (self.parser_backend or "").lower()
        lb = (self.llm_backend or "").lower()
        if not pb:
            pb = "nemotron" if (self.backend or "").lower() == "nvidia" else "docling"
        if not lb:
            b = (self.backend or "local").lower()
            if b == "deterministic":
                lb = "deterministic"
            else:
                lb = "ollama"
        return pb, lb

    def validate(self):
        """Fail fast with a clear message if a selected backend is misconfigured."""
        pb, lb = self.resolve_backends()
        errs = []
        if pb == "nemotron" and (self.nvidia_mode or "cloud").lower() == "cloud":
            if not self.nvidia_api_key("ocr"):
                errs.append(
                    "parser_backend=nemotron (cloud) but no API key: set a per-model "
                    "NVIDIA_*_API_KEY or shared $" + self.nvidia_api_key_env)
        if lb == "ollama" and not (self.ollama_base_url or self.llm_base_url):
            errs.append("llm_backend=ollama but no NHPC_OLLAMA_BASE_URL set")
        if lb == "groq" and not self.groq_api_key():
            errs.append("llm_backend=groq but no API key in env $" + self.groq_api_key_env)
        if lb == "gemini" and not self.gemini_api_key():
            errs.append("llm_backend=gemini but no API key in env $"
                        + self.gemini_api_key_env + " (or $GOOGLE_API_KEY)")
        return errs

    # --- trace / observability -------------------------------------------
    # Postgres DSN for the trace layer. Empty -> append-only JSONL under
    # _reports/trace/ so the pipeline runs without a DB. One env var flips to PG.
    trace_dsn: str = field(default_factory=lambda: _env_str("NHPC_TRACE_DSN", ""))
    trace_enabled: bool = field(default_factory=lambda: _env_bool("NHPC_TRACE", True))

    # --- Langfuse (OPTIONAL developer-facing trace UI; OFF by default) --------
    # A thin MIRROR on top of the Postgres/JSONL trace layer above -- that layer
    # stays the durable system-of-record; Langfuse only adds a browsable UI. It is
    # DORMANT unless langfuse_enabled is true: when false the SDK is never imported,
    # never connects, and adds no latency. When true (on the on-prem server) it
    # points at a SELF-HOSTED Langfuse instance -- NOT Langfuse cloud, since traces
    # carry document content. Validated only when enabled; missing keys -> log and
    # fall back to disabled rather than crash. One flag flips it on with no code
    # change. See phase2/trace/langfuse_client.py.
    langfuse_enabled: bool = field(default_factory=lambda: _env_bool("LANGFUSE_ENABLED", False))
    langfuse_host: str = field(default_factory=lambda: _env_str("LANGFUSE_HOST", "http://localhost:3000"))
    langfuse_public_key_env: str = field(default_factory=lambda: _env_str("LANGFUSE_PUBLIC_KEY_ENV", "LANGFUSE_PUBLIC_KEY"))
    langfuse_secret_key_env: str = field(default_factory=lambda: _env_str("LANGFUSE_SECRET_KEY_ENV", "LANGFUSE_SECRET_KEY"))

    # --- parser adapters (activate if the library is importable) ---
    prefer_docling: bool = field(default_factory=lambda: _env_bool("NHPC_USE_DOCLING", True))
    enable_ocr: bool = field(default_factory=lambda: _env_bool("NHPC_USE_OCR", True))
    ocr_lang: str = "en,hi"  # OCR language hint (Devanagari)

    # --- table extraction (Docling's built-in IBM TableFormer model) ---
    # TableFormer recovers table cell structure (rows/cols/spans). "accurate" is
    # slower but far better on merged/spanning headers (the NHPC tables); "fast"
    # trades quality for speed. Cell matching aligns PDF text to predicted cells.
    tableformer_mode: str = field(default_factory=lambda: _env_str("NHPC_TABLEFORMER_MODE", "accurate"))
    tableformer_cell_matching: bool = field(default_factory=lambda: _env_bool("NHPC_TABLEFORMER_CELL_MATCHING", True))

    # --- per-page routing heuristics (combined-image PDFs) ---------------
    # A page with fewer than this many extractable chars is NOT treated as digital.
    scanned_char_threshold_per_page: int = 40
    # If image coverage (image area / page area) exceeds this AND text is sparse,
    # the page is image-based/scanned rather than digital.
    image_coverage_threshold: float = 0.55
    # Below this char count a sparse-text page with a big image -> visual/ocr path.
    visual_min_text_chars: int = 40

    # libreoffice/soffice for DOC/RTF -> DOCX conversion (flagged when used)
    libreoffice_bin: str = field(default_factory=lambda: _env_str("NHPC_LIBREOFFICE", ""))

    # --- behaviour ---
    force: bool = False        # re-parse even if parsed.json exists
    dry_run: bool = False      # analyze + report, write nothing
    limit: int = 0             # 0 == no limit


def load_config(**overrides) -> Config:
    cfg = Config()
    for k, v in overrides.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
