#!/usr/bin/env bash
set -euo pipefail

echo "=== RAG Chatbot Startup Script ==="

echo "Waiting for Ollama to be ready..."
OLLAMA_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
        echo "Ollama is ready!"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Waiting for Ollama... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Error: Ollama did not become ready at $OLLAMA_URL after $MAX_RETRIES retries." >&2
    exit 1
fi

CHAT_MODEL="${OLLAMA_MODEL_NAME:-hf.co/google/gemma-2b-it}"
EMBED_MODEL="${EMBEDDING_MODEL_NAME:-hf.co/Snowflake/snowflake-arctic-embed-m-v1.5:BF16}"

pull_model() {
    local model="$1"
    echo "Pulling Ollama model: $model"
    curl -fsS -X POST "$OLLAMA_URL/api/pull" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"$model\", \"stream\": false}"
    echo
}

pull_model "$CHAT_MODEL"
pull_model "$EMBED_MODEL"

INDEX_DIR="${INDEX_DIR:-/app/index/lucene_index}"
RAW_DATA_DIR="${RAW_DATA_DIR:-/app/data/raw}"
REBUILD_INDEX="${REBUILD_INDEX:-0}"

# Lucene writes a segments_* file once the index has been committed.
# If one exists, we treat the index as already built and skip the rebuild
# unless the caller explicitly opts in via REBUILD_INDEX=1.
if [ "$REBUILD_INDEX" = "1" ] || ! ls "$INDEX_DIR"/segments_* >/dev/null 2>&1; then
    echo "Building Lucene index in $INDEX_DIR..."
    rm -rf "$INDEX_DIR"
    mkdir -p "$INDEX_DIR"
    python -m scripts.build_index
else
    echo "Existing Lucene index found in $INDEX_DIR, skipping rebuild (set REBUILD_INDEX=1 to force)."
fi

echo "=== Starting Streamlit app ==="
exec streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
