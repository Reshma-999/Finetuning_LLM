# Multi-stage Dockerfile for the Financial LLM inference service
# Base: CUDA 12.1 + Python 3.11 + vLLM

# ── Build stage: install Python deps ─────────────────────────────────────────
FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf python3.11 /usr/bin/python3 && \
    ln -sf python3    /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.3.0+cu121 torchvision==0.18.0+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/opt/models/mistral-finance-v1-merged \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf python3.11 /usr/bin/python3 && ln -sf python3 /usr/bin/python

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin            /usr/local/bin

# Copy source
COPY inference/ ./inference/
COPY training/src/dataset.py ./training/src/dataset.py
COPY training/__init__.py    ./training/__init__.py

# Create model directory
RUN mkdir -p /opt/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["uvicorn", "inference.src.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools"]
