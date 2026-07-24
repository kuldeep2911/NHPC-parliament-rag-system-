#!/usr/bin/env bash
# NHPC Q&A — connectivity preflight. Run BEFORE bootstrap to confirm every dependency
# the app needs is up and reachable. Read-only: it changes nothing.
#
#   ./deploy/preflight.sh
#
# Reads the localhost URLs from .env so it checks exactly what the app will use.

set -uo pipefail
cd "$(dirname "$0")/.."
set -a; [[ -f .env ]] && . ./.env; set +a

pass=0; fail=0
chk(){ if eval "$2" >/dev/null 2>&1; then printf "  \033[1;32m✓\033[0m %s\n" "$1"; pass=$((pass+1));
       else printf "  \033[1;31m✗\033[0m %s\n" "$1"; fail=$((fail+1)); fi; }

echo "Preflight — checking on-prem dependencies:"

# Postgres
chk "Postgres @ ${PHASE3_DB_DSN%%@*}@..." \
    "python3 -c \"import psycopg,os; psycopg.connect(os.environ['PHASE3_DB_DSN']).close()\""

# Embeddings NIM
EMB="${EMBED_SELFHOSTED_URL:-http://localhost:8801/v1/embeddings}"
chk "Embeddings NIM @ ${EMB}" "curl -sf ${EMB%/v1/*}/v1/health/ready"

# Reranker NIM
RRK="${RERANK_SELFHOSTED_URL:-http://localhost:8802/v1/ranking}"
chk "Reranker NIM   @ ${RRK}" "curl -sf ${RRK%/v1/*}/v1/health/ready"

# LLM server + the model present. Uses the OpenAI-standard /v1/models endpoint, which BOTH
# Ollama AND vLLM expose — so this check is correct whether the LLM is a local Ollama or a
# remote vLLM (NHPC_LLM_BASE_URL takes priority over NHPC_OLLAMA_BASE_URL, matching the code).
LLM_BASE="${NHPC_LLM_BASE_URL:-${NHPC_OLLAMA_BASE_URL:-http://localhost:11434/v1}}"
LLM_MODEL="${NHPC_LLM_MODEL:-qwen3:14b}"
LLM_AUTH=""
[ -n "${NHPC_LLM_API_KEY:-}" ] && LLM_AUTH="-H \"Authorization: Bearer ${NHPC_LLM_API_KEY}\""
chk "LLM server     @ ${LLM_BASE}" \
    "eval curl -sf ${LLM_AUTH} ${LLM_BASE%/}/models"
chk "LLM model '${LLM_MODEL}' available" \
    "eval curl -sf ${LLM_AUTH} ${LLM_BASE%/}/models | grep -q \"${LLM_MODEL}\""

echo
if [[ $fail -eq 0 ]]; then
  printf "\033[1;32mAll %d checks passed — ready to bootstrap.\033[0m\n" "$pass"
else
  printf "\033[1;31m%d check(s) failed.\033[0m Fix these before ./deploy/bootstrap.sh\n" "$fail"
  echo "  - Postgres:  cd deploy/postgres && docker compose up -d"
  echo "  - Models:    cd deploy/models   && docker compose up -d   (wait ~3 min for NIMs)"
  exit 1
fi
