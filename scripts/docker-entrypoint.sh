set +e

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
    echo "Warning: Ollama did not become ready in time. Continuing anyway..."
fi

INDEX_DIR="${INDEX_DIR:-/app/index/lucene_index}"
RAW_DATA_DIR="${RAW_DATA_DIR:-/app/data/raw}"

echo "Rebuilding Lucene index..."

rm -rf "$INDEX_DIR"
mkdir -p "$INDEX_DIR"

python -m scripts.build_index

echo "=== Starting Streamlit app ==="
exec streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
