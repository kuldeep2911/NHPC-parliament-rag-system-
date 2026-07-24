#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# NHPC Q&A — one-shot bootstrap. Run ONCE after the DB and model services are up.
# Idempotent and resumable: safe to re-run if a step fails — completed work is skipped.
# ═══════════════════════════════════════════════════════════════════════════
#
#   cd /opt/nhpc && ./deploy/bootstrap.sh
#
# Steps: preflight -> migrate -> seed entities -> load corpus -> embed questions ->
#        embed answers -> dedupe entities -> reseed synonyms -> summary.
#
# Requires: .env present at project root, the venv active OR at ./.venv, Postgres
# reachable at PHASE3_DB_DSN, and (for embedding) the embed model service healthy.

set -euo pipefail
cd "$(dirname "$0")/.."          # project root

# --- pick python ------------------------------------------------------------
if [[ -x ".venv/bin/python" ]]; then PY=".venv/bin/python"
elif command -v python3 >/dev/null; then PY="python3"
else PY="python"; fi
export PYTHONUNBUFFERED=1

say(){ printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }
ok(){  printf "\033[1;32m  ✓ %s\033[0m\n" "$*"; }
die(){ printf "\033[1;31m  ✗ %s\033[0m\n" "$*" >&2; exit 1; }

[[ -f .env ]] || die ".env not found at project root — copy deploy/.env.production.example to .env first"

# Load .env into the environment so a MANUAL run (not just the systemd unit's
# EnvironmentFile) picks up every setting — most importantly NHPC_LLM_GROUPING, without
# which the parse silently falls back to the rule-based path and produces poor Q&A pairs.
set -a; . ./.env; set +a

# Guard the one flag whose absence quietly wrecks extraction quality.
if [[ "${NHPC_LLM_GROUPING:-}" != "1" && "${NHPC_LLM_GROUPING:-}" != "true" ]]; then
  die "NHPC_LLM_GROUPING is not enabled in .env. Parsing would use the rule-based fallback \
(no LLM) and produce bad question/answer pairs. Set NHPC_LLM_GROUPING=1 and re-run."
fi

# ---------------------------------------------------------------------------
say "1/8  Preflight — config + connectivity"
$PY - <<'PYEOF' || die "preflight failed — fix .env and re-run"
import sys
from nhpc_qa.config import Settings, load_dotenv
load_dotenv()
cfg = Settings()
errs = cfg.validate_all() if hasattr(cfg, "validate_all") else cfg.validate(need_db=True, need_embed=True)
if errs:
    print("CONFIG ERRORS:")
    for e in errs: print("   -", e)
    sys.exit(1)
# DB reachable?
from nhpc_qa.core.db.session import connect
with connect(cfg) as conn, conn.cursor() as cur:
    cur.execute("SELECT 1")
print("  config valid, database reachable")
PYEOF
ok "preflight passed"

# ---------------------------------------------------------------------------
say "2/8  Database migrations"
$PY -m nhpc_qa.core.db.migrate
ok "schema up to date"

# ---------------------------------------------------------------------------
# We do NOT copy a database dump or a pre-parsed corpus to the server. Only the RAW source
# PDFs are copied; the pipeline REGENERATES everything (organized/ + all DB rows +
# embeddings) from them. This step is the whole data build and is the slow one: it runs the
# local OCR/parse + the LLM extraction + the embedder over the corpus.
say "3/6  Build the index from raw source  (crawl -> parse -> index + embed)"
if [[ -d "${NHPC_SOURCE_ROOT:-Original Data}" || -n "${NHPC_SOURCE_ROOT:-}" ]]; then
  # Resumable: re-running only redoes what changed. --stages can narrow it (e.g. index only).
  $PY -m nhpc_qa.cli run ${BOOTSTRAP_RUN_ARGS:-} || die "pipeline run failed"
  ok "corpus rebuilt from source"
else
  die "source tree not found (set NHPC_SOURCE_ROOT to the copied raw PDFs)"
fi

# ---------------------------------------------------------------------------
say "4/6  Seed + consolidate the entity dictionary"
# seed states/projects/synonyms; --llm adds offline discovery over every document (local LLM)
$PY -m nhpc_qa.entities.build ${ENTITIES_LLM:+--llm} || die "entity seeding failed"
# answer-group embeddings are not part of `run`; build them here (resumable)
$PY -m nhpc_qa.pipeline.index.embed_answers --stale || die "answer embedding failed"
$PY -m nhpc_qa.entities.dedupe || printf "  (dedupe: nothing to merge)\n"
ok "entities ready, answers embedded"

# ---------------------------------------------------------------------------
# Create the FIRST admin so you can log in. Idempotent: if an admin already exists this is a
# no-op. Set AUTH_ADMIN_EMAIL (+ optionally AUTH_ADMIN_PASSWORD) in .env; with a password
# set you log in with it directly, otherwise a one-time password is printed here ONCE.
say "5/6  Create the administrator account"
$PY -m nhpc_qa.cli create-admin || printf "  (admin already exists — nothing to do)\n"

# ---------------------------------------------------------------------------
say "6/6  Summary"
$PY - <<'PYEOF'
from nhpc_qa.config import Settings, load_dotenv
from nhpc_qa.core.db.session import connect
load_dotenv(); cfg = Settings()
with connect(cfg) as conn, conn.cursor() as cur:
    def c(q):
        try: cur.execute(q); return cur.fetchone()[0]
        except Exception: conn.rollback(); return "?"
    print(f"  active documents      : {c('SELECT count(*) FROM diaries WHERE active')}")
    print(f"  sub-questions embedded: {c('SELECT count(*) FROM sub_questions WHERE embedding IS NOT NULL')}"
          f" / {c('SELECT count(*) FROM sub_questions')}")
    print(f"  answer groups embedded: {c('SELECT count(*) FROM answer_groups WHERE embedding IS NOT NULL')}"
          f" / {c('SELECT count(*) FROM answer_groups')}")
    print(f"  entities              : {c('SELECT count(*) FROM entities')}")
    print(f"  concept synonyms      : {c('SELECT count(*) FROM concept_synonyms')}")
PYEOF

printf "\n\033[1;32m✅ Bootstrap complete.\033[0m Start the service:  sudo systemctl enable --now nhpc-api\n"
printf "   Then watch the logs for the first-admin one-time password:  journalctl -u nhpc-api -f\n\n"
