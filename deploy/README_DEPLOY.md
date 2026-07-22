# NHPC Parliamentary Q&A — On-Prem Deployment Guide

Deploy the whole system on a single air-gapped NHPC server. Nothing leaves the network:
Postgres, the embedding + reranker models, and the LLM all run locally. Everything is
config-controlled through one `.env` file, and one bootstrap script builds the index.

**You do three things:** (1) build an offline bundle on an internet machine, (2) copy it to
the server and install, (3) run `bootstrap` and start the service. The rest is automatic.

**No data is migrated.** You do NOT copy a database dump or the parsed `organized/` folder.
You copy only the **raw source PDFs**, and the bootstrap runs the full pipeline
(crawl → parse → index → embed) to REGENERATE `organized/` and every database row on the
server. The admin login is created fresh from `.env`. So the server rebuilds its own data
from the raw documents — the only thing you carry over is the source PDFs and the four
secrets.

---

## 0. What runs where

| Component | How it runs | Local endpoint | Config backend |
|---|---|---|---|
| Postgres + pgvector | docker (`deploy/postgres`) | `localhost:5433` | `PHASE3_DB_DSN` |
| Embeddings (`llama-nemotron-embed-1b-v2`) | docker NIM (`deploy/models`) | `localhost:8801` | `EMBED_BACKEND=nvidia_selfhosted` |
| Reranker (`llama-nemotron-rerank-1b-v2`) | docker NIM (`deploy/models`) | `localhost:8802` | `RERANK_BACKEND=nvidia_selfhosted` |
| **LLM (`Nemotron-Super-49B`)** | docker Ollama (`deploy/models`) | `localhost:11434` | `NHPC_LLM_BACKEND=ollama` |
| API + officer UI + watcher | systemd (`nhpc-api`) | `localhost:8099` | — |
| Reverse proxy (TLS + auth headers) | nginx / org SSO | :443 | — |
| *(optional)* CV NIMs — OCR / page / table | docker `--profile cv` | `:8010/8011/8012` | `NHPC_PARSER_BACKEND=nemotron` |

**Hardware.** The two embed/rerank NIMs need an NVIDIA GPU + the NVIDIA Container Toolkit.
The production LLM is **Llama-3.3-Nemotron-Super-49B** on the GPU. Rough sizing: 1× 24 GB
GPU comfortably holds both 1B NIMs; the 49B LLM wants its **own** GPU (~28–40 GB at 4-bit).
So a typical box is **2 GPUs** — one for embed+rerank, one for the LLM. No GPU at all? See
**Appendix B** for the CPU/HuggingFace-weights path. The optional CV parsing NIMs (below)
need GPU too, but Docling (the default parser) runs CPU-only and needs none.

---

## 1. On an internet-connected machine — build the offline bundle

Use a machine with the **same OS + Python 3.12 + CPU architecture** as the server.

```bash
git clone <repo> nhpc && cd nhpc
docker login nvcr.io                      # NGC API key — needed to pull the NIM images
./deploy/build-offline-bundle.sh          # downloads wheels + docker images + models

# The production LLM: Llama-3.3-Nemotron-Super-49B, pulled into Ollama, then bundled.
ollama pull nemotron-super-49b            # or the exact tag your Ollama registry uses
cp -r ~/.ollama/models nhpc-offline-bundle/models/ollama

# (optional) high-accuracy CV NIMs for parsing — only if you'll use them:
# docker pull nvcr.io/nim/nvidia/nemoretriever-ocr-v1:latest              (+ save)
# docker pull nvcr.io/nim/nvidia/nemoretriever-page-elements-v2:latest    (+ save)
# docker pull nvcr.io/nim/nvidia/nemoretriever-table-structure-v1:latest  (+ save)
```

> **Nemotron tag note.** `nemotron-super-49b` is a placeholder for whatever tag your Ollama
> library / internal registry exposes for Llama-3.3-Nemotron-Super-49B (GGUF). Set the same
> value in `NHPC_LLM_MODEL`. If your registry doesn't carry it, import from a GGUF file with
> an Ollama `Modelfile` (`FROM ./nemotron-super-49b.gguf`) — see "the LLM" in step 3.

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

# 2d. The RAW SOURCE PDFs (the ONLY data you copy — no DB dump, no organized/)
mkdir -p /opt/nhpc/data
cp -r "/path/to/Original Data" "/opt/nhpc/data/Original Data"
# NHPC_SOURCE_ROOT in .env must point here (default /opt/nhpc/data/Original Data).
# organized/ and every DB row are REGENERATED from this by the bootstrap — do not copy them.
```

---

## 3. Start Postgres + the models

```bash
# Postgres
cd /opt/nhpc/deploy/postgres
cp .env.postgres.example .env
sed -i 's/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=<strong-db-password>/' .env
docker compose up -d

# Models (embeddings + reranker + the Nemotron LLM)
cd /opt/nhpc/deploy/models
cp .env.models.example .env                # NGC_API_KEY stays EMPTY on the air-gapped box
docker compose up -d
docker compose ps                          # wait until embed/rerank are "healthy" (~3 min)

# --- the LLM (Nemotron Super 49B) into Ollama ---
# If you copied a populated ~/.ollama volume, the model is already present — skip this.
# Otherwise import the GGUF blob you bundled:
cp -r /path/to/nhpc-offline-bundle/models/ollama/. ./ollama-import/
printf 'FROM /models/nemotron-super-49b.gguf\n' > ollama-import/Modelfile
docker compose exec ollama ollama create nemotron-super-49b -f /models/Modelfile
docker compose exec ollama ollama list     # confirm the model is loaded

# --- (optional) high-accuracy CV parsing NIMs ---
# Only if you set NHPC_PARSER_BACKEND=nemotron in the project .env. Docling needs none.
# docker compose --profile cv up -d
```

### 3b. Langfuse observability (self-hosted)

```bash
cd /opt/nhpc/deploy/langfuse
cp .env.langfuse.example .env
# set every secret (openssl rand -base64 24 for passwords, -hex 32 for the encryption key)
docker compose up -d
# open http://localhost:3000, create an account + a project, copy the pk-lf-/sk-lf- keys,
# and paste them into the PROJECT-ROOT .env (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY).
```

To skip observability entirely, set `LANGFUSE_ENABLED=0` in the project `.env` and don't
start this stack — the SDK is then never imported.

---

## 4. Configure — the one file you edit

```bash
cd /opt/nhpc
cp deploy/.env.production.example .env
```

Edit `.env` and change the **«CHANGE ME» values**:

- `DB_PASSWORD` / the password inside `PHASE3_DB_DSN` — must match step 3's Postgres password
- `AUTH_SECRET_KEY` — `python -c "import secrets; print(secrets.token_urlsafe(48))"`
- `AUTH_ADMIN_EMAIL` — the first admin's email
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` — from the Langfuse UI (step 3b); or set
  `LANGFUSE_ENABLED=0` to skip observability

**Admin login.** Add `AUTH_ADMIN_PASSWORD=<your chosen password>` to `.env` and you log in
with exactly that — the bootstrap creates the admin with it, no forced change. Leave it out
and the bootstrap prints a one-time password to the logs instead (which you must change on
first login).

`NHPC_SOURCE_ROOT` already points at the raw PDFs you copied in step 2d
(`/opt/nhpc/data/Original Data`). Everything else is wired to localhost.

---

## 5. Build everything from the raw PDFs — one command

```bash
cd /opt/nhpc
./deploy/preflight.sh          # confirms DB + all model services answer
./deploy/bootstrap.sh          # migrate schema -> REBUILD from source -> entities -> admin
```

`bootstrap.sh`:
1. migrates the (empty) database **schema** only,
2. runs the full pipeline (`nhpc run`: crawl → parse → index → embed) to **regenerate
   `organized/` and every DB row + embedding from the raw PDFs** — this is the slow step
   (OCR + LLM extraction + embedding over the whole corpus; budget accordingly),
3. seeds + consolidates the entity dictionary and embeds the answer groups,
4. **creates the admin** from `AUTH_ADMIN_EMAIL` (+ `AUTH_ADMIN_PASSWORD` if set).

It is idempotent and resumable — re-run it and only what's missing is redone. It prints a
summary (documents, vectors, entities) at the end.

> **Faster iteration:** `BOOTSTRAP_RUN_ARGS="--stages index" ./deploy/bootstrap.sh` skips
> crawl+parse and only (re)loads + embeds — useful when the parse output already exists.

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
| Backup | copy the **raw source PDFs** (everything else regenerates via `bootstrap`); optionally `docker exec <pg> pg_dump -U nhpc nhpc > backup.sql` to skip a rebuild |
| Langfuse traces | http://localhost:3000 (self-hosted). `LANGFUSE_ENABLED=0` to disable |
| Change any tuning knob | edit `.env`, `systemctl restart nhpc-api` |

---

## Appendix A — why Docker, and how much disk it really needs

**Why Docker at all.** On an air-gapped server you cannot `apt install` or `pip install`
from the internet. Docker lets you carry each dependency — Postgres+pgvector, the GPU
model servers, Ollama, Langfuse — as a single self-contained image that runs identically to
how it was tested, with `docker compose up -d` and nothing else to configure. The
alternative (installing CUDA, a specific Postgres+pgvector build, Ollama, Node/ClickHouse
for Langfuse, all natively, offline, by hand) is far more work and far less reproducible.
Docker is what makes this a copy-and-run deploy instead of a multi-day sysadmin project. If
you prefer native, **Appendix B** shows the no-Docker path for the models.

**The images do NOT each bake in a copy of the weights.** A common worry is "one image per
model = many copies of the weights = huge disk". That is not how this is set up:

- The NIM/Ollama **images** contain the SERVER code, not the model weights.
- The **weights** are downloaded once into a docker **volume** (`nim_embed_cache`,
  `nim_rerank_cache`, `ollama_models`) and mounted — one copy on disk, not one per image.
- Docker also shares common base **layers** between images, so overlapping OS/runtime
  layers are stored once.

**Realistic disk budget** (order-of-magnitude; verify against your saved tarballs):

| Item | Approx size |
|---|---|
| pgvector/pgvector:pg16 | ~0.4 GB |
| embed NIM image + weights (1B) | ~6–10 GB |
| rerank NIM image + weights (1B) | ~6–10 GB |
| Ollama image | ~1 GB |
| Nemotron-Super-49B weights (4-bit GGUF) | ~28–30 GB |
| Langfuse stack (6 images: app, worker, clickhouse, minio, pg, redis) | ~4–6 GB |
| Python venv + wheelhouse | ~6–8 GB |
| regenerated `organized/` + Postgres data | grows with corpus (GBs) |
| **Total, typical** | **~60–80 GB** |

Most of that is the 49B LLM weights — unavoidable for a local 49B model, and independent of
Docker. If disk is tight: use `requirements-serve.txt` (drops the ~6 GB torch stack when the
server never parses), skip Langfuse, and skip the optional CV NIMs.

## Appendix A2 — verifying it's fully offline

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

## Appendix C — document parsing: Docling vs the NVIDIA CV NIMs

Parsing turns raw PDFs/scans into structured text + tables. Two backends:

- **Docling (default, `NHPC_PARSER_BACKEND=docling`)** — runs entirely on CPU, fully
  offline, no GPU, no extra services. It handled the whole corpus in testing. **This is all
  most deployments need.**
- **NVIDIA CV NIMs (`NHPC_PARSER_BACKEND=nemotron` + `NVIDIA_MODE=onprem`)** — three
  GPU microservices for sharper extraction on messy scans and complex tables:
  | NIM | Purpose | Port | `.env` URL |
  |---|---|---|---|
  | `nemoretriever-ocr-v1` | OCR of scanned pages | 8010 | `NVIDIA_OCR_URL` |
  | `nemoretriever-page-elements-v2` | page layout / regions | 8011 | `NVIDIA_PAGE_ELEMENTS_URL` |
  | `nemoretriever-table-structure-v1` | table cell structure | 8012 | `NVIDIA_TABLE_STRUCTURE_URL` |

  They run locally (nothing leaves the network). Start them and switch on:
  ```bash
  cd deploy/models && docker compose --profile cv up -d     # starts the 3 CV NIMs
  # in the project .env, uncomment the NHPC_PARSER_BACKEND=nemotron block
  systemctl restart nhpc-api
  ```
  They are pulled/saved by `build-offline-bundle.sh` (uncomment their lines) and imported
  with the other images. Only worthwhile if Docling's table/scan output is not sharp enough
  for your corpus — otherwise skip them entirely.

## Appendix D — the config knobs you might touch

All in `.env`; restart the API after changing. Full list in `deploy/.env.production.example`.
Most-tuned: `SIMILARITY_THRESHOLD` (0.02), `RETRIEVE_DENSE_TOP_N` (50), `DRAFT_CONTEXT_K`
(10), `USE_ANSWER_EMBEDDINGS` (1), `LLM_VERIFY_ENABLED` (1). Turn off a whole subsystem with
`DRAFT_ENABLED=0`, `SUPPORTING_ENABLED=0`, `RERANK_ENABLED=0`, or `WATCHER_WITH_SERVE=false`.
