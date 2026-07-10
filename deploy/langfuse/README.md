# Langfuse (self-hosted) for NHPC Phase-2

Optional trace-UI mirror over the pipeline's durable Postgres/JSONL trace layer.
**Off by default** — the pipeline runs identically whether or not any of this exists.
Bring it up only at deployment, on the on-prem server.

## The two halves (you need both)

| Piece | What it is | Where |
|-------|-----------|-------|
| **Langfuse server** | The web app + UI at `http://localhost:3000`. Runs in Docker (this folder). Generates the API keys. | `docker compose up` here |
| **`langfuse` SDK** | The Python client the pipeline uses to send traces to that server. | `pip install "langfuse>=2.53,<3"` |

They are not alternatives. The server is where traces are stored and viewed and where
the keys come from; the SDK is how the pipeline talks to it. Everything stays on the
on-prem machine — **do not use Langfuse cloud** (traces carry document content).

## One-time setup at deployment

1. **Server secrets** — fill in the compose env:
   ```bash
   cd deploy/langfuse
   cp .env.langfuse.example .env
   # replace every value (openssl rand ... — see comments in that file)
   ```

2. **Start the server** (Postgres, ClickHouse, Redis, MinIO come with it):
   ```bash
   docker compose up -d
   docker compose ps          # all healthy?
   ```
   Air-gapped server: pre-mirror the images (`docker save`/`load`, or an internal
   registry) — they come from Docker Hub.

3. **Create project + get keys** — open `http://localhost:3000`, sign up (first user
   is admin), create an Organization → Project, then **Settings → API Keys → Create**.
   Copy the `pk-lf-…` and `sk-lf-…`.
   *(Optional: set `AUTH_DISABLE_SIGNUP=true` in the compose env and restart, so no
   further accounts can be created.)*

4. **Install the SDK** in the pipeline's Python environment:
   ```bash
   pip install "langfuse>=2.53,<3"
   ```

5. **Turn it on** — in the PROJECT-ROOT `.env` (not this folder's):
   ```
   LANGFUSE_ENABLED=1
   LANGFUSE_HOST=http://localhost:3000
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   ```

6. **Verify** with a tiny run:
   ```bash
   python -X utf8 -m phase2.pipeline --only "<session>/<house>" --limit 2 \
       --llm-backend ollama --llm-grouping
   ```
   The banner should read `langfuse:http://localhost:3000` (not `langfuse=disabled`).
   In the UI you'll see one trace per `run_id` with nested spans (routing → parse →
   extract), sharing the same `run_id` as the Postgres `run_steps` rows.

## Turning it back off

Set `LANGFUSE_ENABLED=0` (or remove the line). The pipeline immediately reverts to the
durable-trace-only behavior; the SDK is not even imported. You can leave the Docker
server up or `docker compose down` it — the pipeline does not depend on it being
reachable, and a tracing failure never blocks a document.

## Version note

The pipeline uses the **v2 Langfuse Python SDK** (`langfuse>=2.53,<3`). The compose
pins a v3 **server**, which still ingests v2-SDK events. If you ever move to the v3
SDK, `phase2/trace/langfuse_client.py` must be rewritten to the OTEL-based API — not
required for this deployment.
