FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONPATH=/app:${PYTHONPATH}
COPY requirements.txt ./
# CPU-only torch. The default PyPI build pulls in GBs of CUDA libraries on Linux.
RUN pip install --no-cache-dir --upgrade pip uv && \
    uv pip install --system --no-cache torch==2.14.1 --extra-index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --system --no-cache -r requirements.txt
COPY . .

RUN mkdir -p data/raw index

EXPOSE 8501

ENTRYPOINT ["bash", "/app/scripts/docker-entrypoint.sh"]