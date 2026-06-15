FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONPATH=/app:${PYTHONPATH}

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir --upgrade pip uv && \
    uv export --frozen --no-dev --no-emit-project -o /tmp/requirements.txt && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

COPY . .
RUN chmod +x /app/scripts/docker-entrypoint.sh

RUN mkdir -p data/raw index

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["bash", "/app/scripts/docker-entrypoint.sh"]
