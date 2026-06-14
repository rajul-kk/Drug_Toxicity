# ── stage 1: dependencies ────────────────────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /app

# System libs required by RDKit
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 libxext6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch and PyG first (avoids pulling the ~2 GB CUDA build).
# torch==2.10.0+cpu is a different dist-tag from torch==2.10.0, so we install
# it explicitly here and then skip those lines when reading requirements.txt.
RUN pip install --no-cache-dir --timeout 120 \
    torch==2.10.0+cpu \
    torch-geometric==2.7.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://data.pyg.org/whl/torch-2.10.0+cpu.html

# Install the rest of the dependencies, skipping the torch lines already handled
COPY requirements.txt .
RUN grep -vE "^torch" requirements.txt \
    | pip install --no-cache-dir --timeout 120 -r /dev/stdin


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
