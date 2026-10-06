#!/usr/bin/env bash
set -euo pipefail

echo "Starting RAG chatbot"

echo "Waiting for Ollama"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
        echo "Ollama is ready"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Waiting for Ollama ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Error: Ollama did not become ready after $MAX_RETRIES retries" >&2
    exit 1
fi

CHAT_MODEL="${OLLAMA_MODEL_NAME:-llama3.2:3b-instruct-q8_0}"
EMBED_MODEL="${EMBEDDING_MODEL_NAME:-nomic-embed-text}"

pull_model() {
    local model="$1"
    if curl -fsS "$OLLAMA_URL/api/show" -d "{\"model\": \"$model\"}" > /dev/null 2>&1; then
        echo "Ollama model already present: $model"
        return
    fi
    echo "Pulling Ollama model: $model"
    curl -fsS -X POST "$OLLAMA_URL/api/pull" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"$model\", \"stream\": false}"
    echo
}

pull_model "$CHAT_MODEL"
pull_model "$EMBED_MODEL"

echo "Building index"
python -m scripts.build_index

echo "Starting Streamlit app"
exec streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0