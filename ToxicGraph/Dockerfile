# ── stage 1: dependencies ────────────────────────────────────────────────────
FROM python:3.13-slim AS deps

WORKDIR /app

# System libs required by RDKit and OpenGL (needed by some RDKit renderers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 libxext6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ── stage 2: application ─────────────────────────────────────────────────────
FROM deps AS app

WORKDIR /app

COPY app.py inference.py evaluate.py train.py precompute_cache.py config.yaml ./
COPY src/  ./src/
COPY web/  ./web/

# checkpoints/ and data/ are mounted at runtime via docker-compose volumes.
# Run precompute_cache.py once after mounting weights to skip slow startup.

EXPOSE 8000

CMD ["python", "app.py"]
