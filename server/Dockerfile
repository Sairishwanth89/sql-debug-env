# syntax: docker/dockerfile:1
FROM python:3.11-slim

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── App directory ─────────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps (cached layer) ────────────────────────────────────────────────
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source ───────────────────────────────────────────────────────────────
COPY . .

# ── HF Spaces requires port 7860 ─────────────────────────────────────────────
EXPOSE 7860

# ── Create output dir ─────────────────────────────────────────────────────────
RUN mkdir -p /app/outputs/logs /app/outputs/evals

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
