# ── stage 1: dependencies ────────────────────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /app

# System libs required by RDKit
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 libxext6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to avoid pulling in the CUDA build (~2 GB saved)
RUN pip install --no-cache-dir \
    torch==2.10.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install torch-geometric and the rest of the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch-geometric==2.7.0 \
    --find-links https://data.pyg.org/whl/torch-2.10.0+cpu.html \
    && pip install --no-cache-dir -r requirements.txt \
        --ignore-installed torch torch-geometric


# ── stage 2: application ─────────────────────────────────────────────────────
FROM deps AS app

WORKDIR /app

COPY app.py inference.py evaluate.py explain.py \
     train.py train_activity.py precompute_cache.py \
     config.yaml ./
COPY src/ ./src/
COPY web/ ./web/

# checkpoints/ and data/ are mounted at runtime via docker-compose volumes.
# Run precompute_cache.py once after mounting weights to warm the inference cache.

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
