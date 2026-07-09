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

    # --- GROQ (OpenAI-compatible cloud LLM; llama3.3-70b for the build phase) -----
    # Build-phase only (data leaves the network); at deployment switch llm_backend
    # to a local/on-prem GPU running the same 70B. Key from env ONLY.
    groq_base_url: str = field(default_factory=lambda: _env_str("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
    groq_model: str = field(default_factory=lambda: _env_str("GROQ_MODEL", "llama-3.3-70b-versatile"))
    groq_api_key_env: str = field(default_factory=lambda: _env_str("GROQ_API_KEY_ENV", "GROQ_API_KEY"))

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
    llm_timeout_s: int = 120
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
        return errs

    # --- trace / observability -------------------------------------------
    # Postgres DSN for the trace layer. Empty -> append-only JSONL under
    # _reports/trace/ so the pipeline runs without a DB. One env var flips to PG.
    trace_dsn: str = field(default_factory=lambda: _env_str("NHPC_TRACE_DSN", ""))
    trace_enabled: bool = field(default_factory=lambda: _env_bool("NHPC_TRACE", True))

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
