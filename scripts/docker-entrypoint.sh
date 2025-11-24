#!/bin/bash
# Don't exit on error - we handle errors explicitly
set +e

echo "=== RAG Chatbot Startup Script ==="

# Wait for Ollama to be ready
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

# Check if index exists
INDEX_DIR="${INDEX_DIR:-/app/index/lucene_index}"
RAW_DATA_DIR="${RAW_DATA_DIR:-/app/data/raw}"

# Ensure index directory exists
# mkdir -p "$INDEX_DIR"

# if [ -d "$INDEX_DIR" ] && [ "$(ls -A $INDEX_DIR 2>/dev/null)" ]; then
#     echo "Index directory exists and is not empty. Skipping index build."
#     echo "To rebuild the index, run: python -m scripts.build_index"
# else
#     # Check if we have documents to index
#     if [ -d "$RAW_DATA_DIR" ] && [ "$(ls -A $RAW_DATA_DIR/*.txt $RAW_DATA_DIR/*.md 2>/dev/null)" ]; then
#         echo "Building Lucene index..."
#         python -m scripts.build_index || {
#             echo "ERROR: Index building failed!"
#             echo "The app will start, but search functionality will not work."
#             echo "To build the index manually, run: python -m scripts.build_index"
#         }
#     else
#         echo "WARNING: No documents found in $RAW_DATA_DIR"
#         echo "Index directory created at $INDEX_DIR but is empty."
#         echo "Add .txt or .md files to data/raw/ and run: python -m scripts.build_index"
#     fi
# fi

echo "Rebuilding Lucene index..."

rm -rf "$INDEX_DIR"
mkdir -p "$INDEX_DIR"

python -m scripts.build_index

echo "=== Starting Streamlit app ==="
exec streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0

