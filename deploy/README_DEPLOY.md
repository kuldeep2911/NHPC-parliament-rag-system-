# NHPC Parliamentary Q&A — On-Prem Deployment Guide

Deploy the whole system on a single air-gapped NHPC server. Nothing leaves the network:
Postgres, the embedding + reranker models, and the LLM all run locally. Everything is
config-controlled through one `.env` file, and one bootstrap script builds the index.

**You do three things:** (1) build an offline bundle on an internet machine, (2) copy it to
the server and install, (3) run `bootstrap` and start the service. The rest is automatic.

---

## 0. What runs where

| Component | How it runs | Local endpoint | Config backend |
|---|---|---|---|
| Postgres + pgvector | docker (`deploy/postgres`) | `localhost:5433` | `PHASE3_DB_DSN` |
| Embeddings (`llama-nemotron-embed-1b-v2`) | docker NIM (`deploy/models`) | `localhost:8801` | `EMBED_BACKEND=nvidia_selfhosted` |
| Reranker (`llama-nemotron-rerank-1b-v2`) | docker NIM (`deploy/models`) | `localhost:8802` | `RERANK_BACKEND=nvidia_selfhosted` |
| LLM (`qwen3:14b`) | docker Ollama (`deploy/models`) | `localhost:11434` | `NHPC_LLM_BACKEND=ollama` |
| API + officer UI + watcher | systemd (`nhpc-api`) | `localhost:8099` | — |
| Reverse proxy (TLS + auth headers) | nginx / org SSO | :443 | — |

**Hardware:** the two NIMs need an NVIDIA GPU + the NVIDIA Container Toolkit. Ollama uses
the GPU if present (else CPU, slower). Rough sizing: 1× 24 GB GPU comfortably holds both
1B NIMs; Qwen3-14B wants its own ~16 GB (a second GPU, or run it CPU/quantized). No GPU at
all? See **Appendix B** for the CPU/HuggingFace-weights path.

---

## 1. On an internet-connected machine — build the offline bundle

Use a machine with the **same OS + Python 3.12 + CPU architecture** as the server.

```bash
git clone <repo> nhpc && cd nhpc
docker login nvcr.io                      # NGC API key — needed to pull the NIM images
export NHPC_LLM_MODEL=qwen3:14b
./deploy/build-offline-bundle.sh          # downloads wheels + docker images + models
ollama pull qwen3:14b                      # then copy ~/.ollama/models into the bundle
cp -r ~/.ollama/models nhpc-offline-bundle/models/ollama
```

You now have `nhpc-offline-bundle/` (wheels, docker image tarballs, Docling + Ollama model
blobs, `MANIFEST.txt`). Copy **the repo** and **the bundle** to the server (USB / secure
transfer).

---

## 2. On the NHPC server — install (once)

Prereqs on the server: Docker + docker-compose plugin, the NVIDIA Container Toolkit (for
GPU), and Python 3.12. All offline.

```bash
sudo mkdir -p /opt/nhpc && sudo chown "$USER" /opt/nhpc
cp -r nhpc/*  /opt/nhpc/                   # the repo
cd /opt/nhpc

# 2a. Python venv from the local wheelhouse (no internet)
python3.12 -m venv .venv
. .venv/bin/activate
pip install --no-index --find-links /path/to/nhpc-offline-bundle/wheelhouse \
    pip setuptools wheel
pip install --no-index --find-links /path/to/nhpc-offline-bundle/wheelhouse \
    -r requirements.txt                    # or requirements-serve.txt if never parsing
pip install --no-index --no-build-isolation -e .   # installs the `nhpc` command

# 2b. Load the docker images
for t in /path/to/nhpc-offline-bundle/docker-images/*.tar; do docker load -i "$t"; done

# 2c. Docling + Ollama models into place
mkdir -p ~/.cache/huggingface
cp -r /path/to/nhpc-offline-bundle/models/huggingface/. ~/.cache/huggingface/
mkdir -p deploy/models/ollama-import       # (Ollama model imported in step 3)
```

---

## 3. Start Postgres + the models

```bash
# Postgres
cd /opt/nhpc/deploy/postgres
cp .env.postgres.example .env
sed -i 's/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=<strong-db-password>/' .env
docker compose up -d

# Models
cd /opt/nhpc/deploy/models
cp .env.models.example .env                # NGC_API_KEY stays EMPTY on the air-gapped box
# Import the Ollama model from the copied blob:
cp -r /path/to/nhpc-offline-bundle/models/ollama/. ./ollama-import/
docker compose up -d
docker compose exec ollama ollama create qwen3:14b -f /models/Modelfile   # if using a Modelfile
# (or if you copied a populated ~/.ollama volume, the model is already there)

docker compose ps                          # wait until embed/rerank are "healthy" (~3 min)
```

---

## 4. Configure — the one file you edit

```bash
cd /opt/nhpc
cp deploy/.env.production.example .env
```

Edit `.env` and change **only the four «CHANGE ME» values**:

- `DB_PASSWORD` / the password inside `PHASE3_DB_DSN` — must match step 3's Postgres password
- `AUTH_SECRET_KEY` — `python -c "import secrets; print(secrets.token_urlsafe(48))"`
- `AUTH_ADMIN_EMAIL` — the first admin's email

Also point `NHPC_SOURCE_ROOT` at where new question folders will land (default
`/opt/nhpc/data/Original Data`). Everything else is already wired to localhost.

---

## 5. Build the index — one command

```bash
cd /opt/nhpc
./deploy/preflight.sh          # confirms DB + all 3 model services answer
./deploy/bootstrap.sh          # migrate -> seed -> load -> embed (idempotent, resumable)
```

`bootstrap.sh` prints a summary (documents loaded, vectors built, entities). Safe to re-run
if anything fails — completed steps are skipped.

> **Fresh corpus (no `organized/` yet)?** Run the parse pipeline first to turn raw PDFs into
> `organized/*/parsed.json`, then re-run bootstrap. If you copied a pre-built `organized/`
> from the build machine, bootstrap loads it directly.

---

## 6. Start serving

```bash
# install the services (bootstrap oneshot + api)
sudo cp deploy/systemd/*.service deploy/systemd/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nhpc-api        # pulls in nhpc-bootstrap automatically
journalctl -u nhpc-api -f                   # <-- the first-admin ONE-TIME PASSWORD prints here
```

The watcher starts automatically inside the API (`WATCHER_WITH_SERVE=true`), so dropping new
question folders into `NHPC_SOURCE_ROOT` — or reference documents into
`organized/supporting_documents/<category>/` — auto-ingests them. No extra process.

Health check: `curl -s localhost:8099/health` → `{"ok": true, ...}`.

---

## 7. Put a proxy in front (required)

The API trusts identity from the proxy and **binds to loopback only** — never expose 8099
directly. Terminate TLS at nginx (or the org SSO proxy) and forward to `127.0.0.1:8099`. A
minimal nginx server block:

```nginx
server {
    listen 443 ssl;
    server_name nhpc-qa.internal;
    ssl_certificate     /etc/ssl/nhpc.crt;
    ssl_certificate_key /etc/ssl/nhpc.key;
    location / { proxy_pass http://127.0.0.1:8099; proxy_set_header Host $host; }
}
```

If officers reach it over https, set `AUTH_COOKIE_SECURE=true` in `.env` and restart.

---

## Day-2 operations

| Task | Command |
|---|---|
| Status / logs | `systemctl status nhpc-api` · `journalctl -u nhpc-api -f` |
| Re-embed after a model change | `.venv/bin/python -m nhpc_qa.pipeline.index.embedder --stale` then `... embed_answers --stale` |
| Rebuild entity dictionary | `.venv/bin/python -m nhpc_qa.entities.build --llm` |
| Purge soft-deleted records | enabled via `nhpc-purge.timer` (already installed) |
| Backup | `docker exec <pg> pg_dump -U nhpc nhpc > backup.sql` + copy `organized/` |
| Change any tuning knob | edit `.env`, `systemctl restart nhpc-api` |

---

## Appendix A — verifying it's fully offline

```bash
sudo ss -tupn | grep -vE '127.0.0.1|::1'   # the app should open NO external connections
grep -RniE 'https?://(?!localhost|127)' .env   # every URL in .env must be localhost
```

## Appendix B — no GPU (CPU-only server)

The NIM images require a GPU. Without one, serve the embedder and reranker from local
HuggingFace weights instead (documented in `nhpc_qa/core/providers/embeddings.py`,
`NvidiaSelfHostedEmbedder` path B): set `EMBED_SELFHOSTED_MODEL_PATH` to the copied weights,
implement the sentence-transformers `_embed`, and delete the two `nim` services from
`deploy/models/docker-compose.yml`. Ollama already runs CPU-only (just remove its `deploy:`
GPU block). Expect higher latency; the pipeline is otherwise unchanged.

## Appendix C — the config knobs you might touch

All in `.env`; restart the API after changing. Full list in `deploy/.env.production.example`.
Most-tuned: `SIMILARITY_THRESHOLD` (0.02), `RETRIEVE_DENSE_TOP_N` (50), `DRAFT_CONTEXT_K`
(10), `USE_ANSWER_EMBEDDINGS` (1), `LLM_VERIFY_ENABLED` (1). Turn off a whole subsystem with
`DRAFT_ENABLED=0`, `SUPPORTING_ENABLED=0`, `RERANK_ENABLED=0`, or `WATCHER_WITH_SERVE=false`.
