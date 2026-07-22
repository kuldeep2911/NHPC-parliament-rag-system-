#!/usr/bin/env bash
# NHPC Q&A — build the OFFLINE deployment bundle. Run this on a machine WITH internet
# (same OS + Python 3.12 + CPU arch as the NHPC server). It downloads every artifact the
# air-gapped server needs into ./nhpc-offline-bundle/, which you then copy to the server.
#
#   ./deploy/build-offline-bundle.sh
#
# Produces:
#   nhpc-offline-bundle/wheelhouse/         all pip wheels (from requirements.txt)
#   nhpc-offline-bundle/docker-images/      saved docker image tarballs
#   nhpc-offline-bundle/models/             Docling + Ollama model blobs
#   nhpc-offline-bundle/MANIFEST.txt        checksums
#
# Copy the whole folder to the server (USB/secure transfer) and follow README_DEPLOY.md.

set -euo pipefail
cd "$(dirname "$0")/.."
OUT="nhpc-offline-bundle"
mkdir -p "$OUT"/{wheelhouse,docker-images,models}

echo "== 1. Python wheels =="
python -m pip download -r requirements.txt -d "$OUT/wheelhouse"
python -m pip download pip setuptools wheel -d "$OUT/wheelhouse"
echo "   $(ls "$OUT/wheelhouse" | wc -l) wheels"

echo "== 2. Docker images (Postgres + NIMs + Ollama) =="
# NIM images require a prior `docker login nvcr.io` with your NGC key.
# Core images. The Langfuse stack images are added when INCLUDE_LANGFUSE=1 (see below).
IMAGES=(
  "pgvector/pgvector:pg16"
  "nvcr.io/nim/nvidia/llama-nemotron-embed-1b-v2:latest"
  "nvcr.io/nim/nvidia/llama-nemotron-rerank-1b-v2:latest"
  "ollama/ollama:0.5.7"
)
# Langfuse observability stack (self-hosted). Set INCLUDE_LANGFUSE=1 to bundle it.
if [[ "${INCLUDE_LANGFUSE:-1}" == "1" ]]; then
  # exact tags come from deploy/langfuse/docker-compose.yml — read them so they never drift
  while read -r img; do IMAGES+=("$img"); done < <(
    grep -Eo 'image:\s*\S+' deploy/langfuse/docker-compose.yml | awk '{print $2}' | sort -u)
fi
# Optional CV parsing NIMs (uncomment if you use NHPC_PARSER_BACKEND=nemotron):
# IMAGES+=( "nvcr.io/nim/nvidia/nemoretriever-ocr-v1:latest"
#           "nvcr.io/nim/nvidia/nemoretriever-page-elements-v2:latest"
#           "nvcr.io/nim/nvidia/nemoretriever-table-structure-v1:latest" )
for img in "${IMAGES[@]}"; do
  echo "   pulling $img"
  docker pull "$img"
  fname="$OUT/docker-images/$(echo "$img" | tr '/:' '__').tar"
  docker save "$img" -o "$fname"
  echo "   saved $(basename "$fname")"
done

echo "== 3. Docling models (parsing/OCR) =="
# Docling caches its layout/table/OCR models under HF_HOME on first run. Trigger a tiny
# parse so they download, then copy the cache.
python - <<'PYEOF' || echo "   (docling warmup skipped — copy your existing HF cache manually)"
try:
    from docling.document_converter import DocumentConverter
    DocumentConverter()   # constructing it fetches the default models
    print("   docling models cached")
except Exception as e:
    print("   docling warmup note:", e)
PYEOF
HF="${HF_HOME:-$HOME/.cache/huggingface}"
[[ -d "$HF" ]] && { mkdir -p "$OUT/models/huggingface"; cp -r "$HF/." "$OUT/models/huggingface/"; \
                    echo "   copied HF cache from $HF"; }

echo "== 4. Ollama LLM blob =="
echo "   On this internet machine run:  ollama pull ${NHPC_LLM_MODEL:-qwen3:14b}"
echo "   then copy ~/.ollama/models into $OUT/models/ollama/  (or use 'ollama save')."

echo "== 5. Manifest =="
( cd "$OUT" && find . -type f -exec sha256sum {} \; > MANIFEST.txt )
echo
echo "✅ Bundle ready: $OUT/  ($(du -sh "$OUT" | cut -f1))"
echo "   Copy the whole folder to the NHPC server, then follow deploy/README_DEPLOY.md."
