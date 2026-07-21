"""
Phase-4 configuration. Everything from env; nothing hardcoded; validated at startup.

Secrets (DB password inside the DSN, NIM keys) are read from the environment only and
never logged. `describe()` returns a redacted view safe for the trace/report.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from nhpc_qa.config.parse import load_dotenv                     # noqa: F401 (re-exported)
from nhpc_qa.config.index import Phase3Config, _redact_dsn


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return v if v is not None and v.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


# The reranker actually in use. The model the spec named
# (nvidia/nv-rerankqa-mistral-4b-v3) does NOT exist -> HTTP 404, and
# nvidia/llama-3.2-nv-rerankqa-1b-v2 is END OF LIFE -> HTTP 410. Rerankers are not
# listed by GET /v1/models; they live on ai.api.nvidia.com under a per-model path.
# Measured live: ranks the relevant passage first and is strongly multilingual (it put
# a Hindi passage above its English equivalent for a Hindi-relevant query).
DEFAULT_RERANK_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
DEFAULT_RERANK_URL = ("https://ai.api.nvidia.com/v1/retrieval/nvidia/"
                      "llama-nemotron-rerank-1b-v2/reranking")


@dataclass
class Phase4Config(Phase3Config):
    """Extends the Phase-3 config (DB + embeddings) with retrieval/rerank/generation."""

    # --- retrieval: candidates per retriever ---------------------------------
    dense_top_n:   int = field(default_factory=lambda: _env_int("RETRIEVE_DENSE_TOP_N", 30))
    keyword_top_n: int = field(default_factory=lambda: _env_int("RETRIEVE_KEYWORD_TOP_N", 30))
    entity_top_n:  int = field(default_factory=lambda: _env_int("RETRIEVE_ENTITY_TOP_N", 30))
    # How many candidates the reranker scores. This is the POOL the sigmoid filter and the
    # LLM verify pass draw from -- it is NOT a display cap. It must be generous, because the
    # output is now uncapped: a question genuinely asked 12 times must have all 12 reach the
    # reranker, or "no cap" is a lie (the cap just moved upstream). Calibration pools were
    # 19-27; 80 leaves ample headroom for a frequent topic.
    # RERANK_CANDIDATE_POOL is the name that describes what it now is; RETRIEVE_FINAL_TOP_K
    # is the old name and still works, so existing .env files do not break.
    final_top_k:   int = field(default_factory=lambda:
        _env_int("RERANK_CANDIDATE_POOL", _env_int("RETRIEVE_FINAL_TOP_K", 80)))

    # ─── SIGMOID RELEVANCE FILTER ────────────────────────────────────────────
    # The reranker emits a RAW LOGIT (unbounded, NOT a probability). sigmoid(logit) rescales
    # it to 0-1 for interpretability: sigmoid(0)=0.5, big positive -> ~1, big negative -> ~0.
    #
    # ⚠️ THIS IS A MONOTONIC RESCALING, NOT A CALIBRATED PROBABILITY. ⚠️ 0.9 does NOT mean
    # "correct 90% of the time". And it does NOT separate matches from noise on its own --
    # calibration (scratchpad/calibrate2.py) proved the two populations overlap completely:
    # the lowest genuine match scored sigmoid 0.003, while boilerplate noise ("details
    # thereof") scored 0.9999. No sigmoid value splits them.
    #
    # So the threshold is deliberately LOW and its job is RECALL, not precision: a cheap
    # pre-filter that bounds how many candidates reach the LLM verify pass, which is what
    # actually judges similarity. 0.05 keeps every genuine match we could retrieve except
    # one rare-topic outlier at 0.003 (a threshold low enough to keep THAT admits nearly all
    # noise), and the LLM removes the boilerplate the filter lets through.
    #
    # MODEL-SPECIFIC. Calibrated for llama-nemotron-rerank-1b-v2. A different reranker emits
    # logits on a different scale -- re-run the calibration before trusting this value.
    similarity_threshold: float = field(
        default_factory=lambda: _env_float("SIMILARITY_THRESHOLD", 0.1))

    # A bound on pathological cases, NOT a relevance cap. The relevance decision is the
    # sigmoid filter + the LLM; this only stops a degenerate query returning hundreds.
    safety_max_results: int = field(
        default_factory=lambda: _env_int("SAFETY_MAX_RESULTS", 50))

    # ─── LLM SIMILARITY VERIFICATION ─────────────────────────────────────────
    # The precision stage. ONE batched call judges every sigmoid-passing candidate for
    # genuine similarity to the query, and non-matches are dropped. It puts an LLM call in
    # the LIVE query path -- accepted deliberately for precision -- so it is ONE batched
    # call, timed separately in the trace, and RESILIENT: if it fails, the sigmoid set is
    # returned unverified with a "verification_unavailable" flag rather than an error.
    llm_verify_enabled: bool = field(
        default_factory=lambda: _env_bool("LLM_VERIFY_ENABLED", True))

    # Run the LLM entity-discovery pass when a file is uploaded (offline, in the watcher --
    # NOT the query path). Off by default: the seed lists + "Full (ABBR)" mining already
    # cover most entities for free; the per-file LLM call is opt-in.
    entities_llm_on_upload: bool = field(
        default_factory=lambda: _env_bool("ENTITIES_LLM_ON_UPLOAD", False))
    llm_verify_max_tokens: int = field(
        default_factory=lambda: _env_int("LLM_VERIFY_MAX_TOKENS", 1500))

    # --- RRF fusion ----------------------------------------------------------
    # score(d) = sum over retrievers r of  weight_r / (rrf_k + rank_r(d))
    # rank is 1-based; a retriever that did not surface d contributes nothing.
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    rrf_weight_dense:   float = field(default_factory=lambda: _env_float("RRF_W_DENSE", 1.0))
    rrf_weight_keyword: float = field(default_factory=lambda: _env_float("RRF_W_KEYWORD", 0.7))
    rrf_weight_entity:  float = field(default_factory=lambda: _env_float("RRF_W_ENTITY", 0.5))

    # ─── ANSWER-EMBEDDING EXPERIMENT (additive, OFF by default) ───────────────
    # When TRUE, dense retrieval ALSO searches answer_group embeddings (migration 020): a
    # record can be retrieved because its QUESTION matched OR its ANSWER matched. The answer
    # hit is mapped back to the sub_question(s) linked to that answer_group and fused into the
    # SAME candidate set as a THIRD retriever ("dense_answer"), deduped by doc_key. Everything
    # downstream (rerank query-vs-question, sigmoid, LLM verify, date order) is unchanged.
    #
    # ⚠️ FALSE MUST REPRODUCE CURRENT BEHAVIOUR EXACTLY. ⚠️ When off, the answer retriever is
    # never run and never imported into the hot path, and fusion sees only dense+entity.
    #
    # ANSWER_EMBED_WEIGHT is the RRF weight of the answer signal — deliberately MODEST by
    # default (below dense's 1.0) so the answer signal boosts recall without letting shared
    # answer boilerplate dominate the fused order. ANSWER_TOP_N bounds the answer groups
    # scanned before expansion to sub-questions.
    use_answer_embeddings: bool = field(
        default_factory=lambda: _env_bool("USE_ANSWER_EMBEDDINGS", True))
    answer_top_n: int = field(
        default_factory=lambda: _env_int("RETRIEVE_ANSWER_TOP_N", 30))
    answer_embed_weight: float = field(
        default_factory=lambda: _env_float("ANSWER_EMBED_WEIGHT", 0.5))

    # --- WIDEN branch (CHANGE 3 + 4) -----------------------------------------
    # RRF SCORES ARE NOT ON A 0-1 SCALE. With rrf_k=60, one retriever at rank 1
    # contributes weight/(60+1): dense 0.0164, keyword 0.0115, entity 0.0082. Scores
    # accumulate across BOTH retrievers and the several sub-questions of one document
    # (we fuse per doc_key), so a strong document can exceed the naive
    # "rank 1 in all three" figure of 0.036.
    #
    # MEASURED on this corpus (phase4/scripts/measure_rrf.py, 517 docs / 1914 parts):
    #     strong + entity queries : top_score 0.027 .. 0.085   gap >= 0.0017
    #     vague / nonsense queries: top_score 0.028 .. 0.032   gap  0.0004 .. 0.0054
    #
    # KEY FINDING: top_score does NOT separate good from bad -- the ranges OVERLAP
    # (a vague query still has a nearest neighbour). The #1-#2 GAP is the discriminating
    # signal: "details thereof" gaps at 0.0004, an order of magnitude below any strong
    # query. So DELTA does the real work and TAU is set low, as a floor that only fires
    # when retrieval is genuinely degenerate -- otherwise we would widen on every query
    # and triple the latency for nothing.
    widen_enabled: bool = field(default_factory=lambda: _env_bool("WIDEN_ENABLED", True))
    widen_tau:   float = field(default_factory=lambda: _env_float("WIDEN_TAU", 0.0150))
    widen_delta: float = field(default_factory=lambda: _env_float("WIDEN_DELTA", 0.0015))
    # What WIDEN actually DOES (re-running node 2 unchanged would be pointless):
    #   1. every retriever's top-N is multiplied by widen_top_n_factor, AND
    #   2. the entity retriever relaxes from FILTER to BOOST-ONLY, AND
    #   3. metadata filters (house/session/is_nhpc_relevant) are dropped.
    # Capped at ONE retry.
    widen_top_n_factor: int = field(default_factory=lambda: _env_int("WIDEN_TOP_N_FACTOR", 3))

    # --- reranker ------------------------------------------------------------
    #   nvidia_nim_api    -> NVIDIA-hosted NIM (dev). Text leaves the network.
    #   nvidia_selfhosted -> on-prem NIM (server). Nothing leaves.
    # DISPLAY ORDER of the final top-K. Retrieval is unaffected either way -- the same
    # documents are shown; only their order changes.
    #   date       (default) relevance selects the top-K, then reply_date DESC orders it,
    #              so the officer reads the most RECENT of the RELEVANT replies first.
    #              Undated documents sort last, marked "date unknown".
    #   relevance  pure cross-encoder order (the pre-date behaviour).
    result_sort: str = field(default_factory=lambda: _env("RESULT_SORT", "date"))

    rerank_enabled: bool = field(default_factory=lambda: _env_bool("RERANK_ENABLED", True))
    rerank_backend: str = field(default_factory=lambda: _env("RERANK_BACKEND", "nvidia_nim_api"))
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", DEFAULT_RERANK_MODEL))
    rerank_url: str = field(default_factory=lambda: _env("RERANK_URL", DEFAULT_RERANK_URL))
    rerank_api_key_env: str = field(default_factory=lambda: _env("RERANK_API_KEY_ENV",
                                                                 "NVIDIA_RERANK_API_KEY"))
    rerank_selfhosted_url: str = field(default_factory=lambda: _env("RERANK_SELFHOSTED_URL", ""))
    rerank_timeout_s: int = field(default_factory=lambda: _env_int("RERANK_TIMEOUT_S", 60))
    rerank_max_retries: int = field(default_factory=lambda: _env_int("RERANK_MAX_RETRIES", 4))

    # --- generation (OPTIONAL, OFF by default) -------------------------------
    # When off the node is skipped entirely. When on and it FAILS, the officer still
    # gets their retrieved results -- generation is wrapped and degrades gracefully.
    generation_enabled: bool = field(default_factory=lambda: _env_bool("GENERATION_ENABLED", False))
    generation_max_tokens: int = field(default_factory=lambda: _env_int("GENERATION_MAX_TOKENS", 1200))

    # --- Langfuse (OPTIONAL developer trace UI; OFF by default) ---------------
    # Phase4Config extends Phase3Config, which does not carry these -- they live on the
    # Phase-2 Config. Redeclared here (same env var names, same defaults) so the Phase-4
    # tracer reads one config object. When false the SDK is never imported and tracing
    # costs one boolean check per node.
    langfuse_enabled: bool = field(default_factory=lambda: _env_bool("LANGFUSE_ENABLED", False))
    langfuse_host: str = field(default_factory=lambda: _env("LANGFUSE_HOST", "http://localhost:3000"))
    langfuse_public_key_env: str = field(default_factory=lambda: _env("LANGFUSE_PUBLIC_KEY_ENV", "LANGFUSE_PUBLIC_KEY"))
    langfuse_secret_key_env: str = field(default_factory=lambda: _env("LANGFUSE_SECRET_KEY_ENV", "LANGFUSE_SECRET_KEY"))

    # --- WATCHER / incremental sync -----------------------------------------
    # AUTO-START WITH THE UI. When true (default), `nhpc serve` also launches the watcher in
    # a background daemon thread, so starting the officer UI automatically starts incremental
    # sync -- no second `nhpc watch` process to remember. Set WATCHER_WITH_SERVE=false to run
    # them separately (e.g. the watcher on a different host). A watcher failure never affects
    # the UI: it is caught and logged, and the API keeps serving.
    # (Jab true ho — jo default hai — to `nhpc serve` watcher ko background daemon thread me
    # bhi chala deta hai, isliye UI start karte hi incremental sync apne aap chalu ho jaata
    # hai; alag `nhpc watch` process yaad rakhne ki zaroorat nahi. Alag chalane ke liye
    # WATCHER_WITH_SERVE=false karein. Watcher fail ho to bhi UI par asar nahi padta.)
    watcher_with_serve: bool = field(
        default_factory=lambda: _env_bool("WATCHER_WITH_SERVE", True))
    # The ORIGINAL source tree the watcher observes. READ-ONLY: the watcher never writes
    # into it; the crawl stage copies OUT of it into organized/.
    source_root: str = field(default_factory=lambda: _env("NHPC_SOURCE_ROOT", "Original Data"))
    # SETTLING: how long a question folder must be QUIET before it is processed. A folder
    # is copied in file by file; parsing it mid-copy would read a reply whose annexure has
    # not landed yet and record "referenced but unavailable" as fact. 10s is a sane default
    # for a local copy; raise it for a slow network share.
    watch_settle_seconds: int = field(default_factory=lambda: _env_int("WATCH_SETTLE_SECONDS", 10))
    # how often the worker looks for jobs whose quiet period has elapsed
    watch_poll_seconds: int = field(default_factory=lambda: _env_int("WATCH_POLL_SECONDS", 5))
    # a job that keeps failing is left for an operator rather than retried forever
    watch_max_attempts: int = field(default_factory=lambda: _env_int("WATCH_MAX_ATTEMPTS", 3))
    # a worker killed mid-job leaves its row claimed; this releases it
    watch_stale_seconds: int = field(default_factory=lambda: _env_int("WATCH_STALE_SECONDS", 900))
    # PURGE grace: how long a soft-deleted record must have been inactive before
    # `nhpc purge` will permanently remove it. Deliberately long -- a purge is not undoable.
    purge_grace_days: int = field(default_factory=lambda: _env_int("PURGE_GRACE_DAYS", 30))

    # --- API / security ------------------------------------------------------
    api_host: str = field(default_factory=lambda: _env("PHASE4_API_HOST", "127.0.0.1"))
    # 8099, not 8080: on Windows, 8080 frequently falls inside a reserved TCP exclusion
    # range (Hyper-V / WinNAT), and binding it fails with WinError 10013 "an attempt was
    # made to access a socket in a way forbidden by its access permissions".
    # Check yours with:  netsh interface ipv4 show excludedportrange protocol=tcp
    api_port: int = field(default_factory=lambda: _env_int("PHASE4_API_PORT", 8099))
    # Roles allowed to query / open files. Comma-separated.
    roles_query: str = field(default_factory=lambda: _env("PHASE4_ROLES_QUERY", "officer,admin"))
    roles_file: str = field(default_factory=lambda: _env("PHASE4_ROLES_FILE", "officer,admin"))

    def rerank_api_key(self):
        """Reranker key from env. None if unset. Never logged."""
        return os.environ.get(self.rerank_api_key_env)

    def langfuse_keys(self):
        """(public_key, secret_key) from env; read only when langfuse_enabled.
        Required by nhpc_qa.core.trace.langfuse_client.LangfuseTracer, which this config is
        passed to -- keeping the same method name means that client is reused as-is."""
        return (os.environ.get(self.langfuse_public_key_env),
                os.environ.get(self.langfuse_secret_key_env))

    # ---- validation ---------------------------------------------------------
    def validate_phase4(self):
        errs = list(self.validate(need_db=True, need_embed=True))
        if self.rerank_enabled:
            if self.rerank_backend not in ("nvidia_nim_api", "nvidia_selfhosted"):
                errs.append(f"RERANK_BACKEND must be nvidia_nim_api|nvidia_selfhosted, "
                            f"got {self.rerank_backend!r}")
            if self.rerank_backend == "nvidia_nim_api" and not self.rerank_api_key():
                errs.append(f"RERANK_ENABLED=1 but ${self.rerank_api_key_env} is unset")
            if self.rerank_backend == "nvidia_selfhosted" and not self.rerank_selfhosted_url:
                errs.append("RERANK_BACKEND=nvidia_selfhosted but RERANK_SELFHOSTED_URL is unset")
        if self.rrf_k <= 0:
            errs.append(f"RRF_K must be positive, got {self.rrf_k}")
        if self.widen_top_n_factor < 2:
            errs.append("WIDEN_TOP_N_FACTOR must be >= 2 or WIDEN cannot broaden anything")
        if self.use_answer_embeddings:
            if self.answer_embed_weight < 0:
                errs.append(f"ANSWER_EMBED_WEIGHT must be >= 0, got {self.answer_embed_weight}")
            if self.answer_top_n <= 0:
                errs.append(f"RETRIEVE_ANSWER_TOP_N must be positive, got {self.answer_top_n}")
        return errs

    def describe(self) -> dict:
        d = super().describe()
        d.update({
            "dense_top_n": self.dense_top_n,
            "keyword_top_n": self.keyword_top_n,
            "entity_top_n": self.entity_top_n,
            "final_top_k": self.final_top_k,
            "rrf_k": self.rrf_k,
            "rrf_weights": {"dense": self.rrf_weight_dense,
                            "keyword": self.rrf_weight_keyword,
                            "entity": self.rrf_weight_entity},
            "use_answer_embeddings": self.use_answer_embeddings,
            "answer_embed_weight": self.answer_embed_weight,
            "answer_top_n": self.answer_top_n,
            "widen": {"enabled": self.widen_enabled, "tau": self.widen_tau,
                      "delta": self.widen_delta, "top_n_factor": self.widen_top_n_factor},
            "rerank_enabled": self.rerank_enabled,
            "rerank_backend": self.rerank_backend,
            "rerank_model": self.rerank_model,
            "rerank_api_key_set": bool(self.rerank_api_key()),
            "generation_enabled": self.generation_enabled,
        })
        return d

    def roles_for(self, action: str):
        raw = self.roles_query if action == "query" else self.roles_file
        return {r.strip() for r in raw.split(",") if r.strip()}
