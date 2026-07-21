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
say "3/8  Seed entity dictionary (states, projects, synonyms)"
# --llm adds offline discovery over every document (uses the local LLM). Drop --llm for a
# faster, seed-only build.
$PY -m nhpc_qa.entities.build ${ENTITIES_LLM:+--llm} || die "entity seeding failed"
ok "entities seeded"

# ---------------------------------------------------------------------------
say "4/8  Load parsed corpus into Postgres (idempotent upsert)"
if [[ -d organized ]]; then
  $PY -m nhpc_qa.pipeline.index.loader
  ok "corpus loaded"
else
  printf "  (no organized/ folder found — skipping; run the parse pipeline first if this is a fresh corpus)\n"
fi

# ---------------------------------------------------------------------------
say "5/8  Embed sub-questions  (resumable: only missing rows)"
$PY -m nhpc_qa.pipeline.index.embedder --stale
ok "question embeddings ready"

# ---------------------------------------------------------------------------
say "6/8  Embed answer groups  (resumable: only missing rows)"
$PY -m nhpc_qa.pipeline.index.embed_answers --stale
ok "answer embeddings ready"

# ---------------------------------------------------------------------------
say "7/8  Consolidate duplicate entities"
$PY -m nhpc_qa.entities.dedupe || printf "  (dedupe skipped or nothing to merge)\n"
ok "entities consolidated"

# ---------------------------------------------------------------------------
say "8/8  Summary"
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
