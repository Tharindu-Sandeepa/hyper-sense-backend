# ══════════════════════════════════════════════════════════════════════════════
#  hyper-sense-backend  ·  Dockerfile (Fixed for PermissionError)
#  Multi-stage build → final image ~200 MB with YOLO26n pre-downloaded
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────── Stage 1: Builder ─────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/pkg --no-cache-dir -r requirements.txt

# ─────────────────────────────── Stage 2: Runtime ─────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="hyper-sense-backend"
LABEL description="IoT people-counting API powered by YOLO26n + FastAPI"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /pkg /usr/local

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

COPY main.py .

# Pre-download YOLO model AS ROOT (before USER appuser) → fixes PermissionError
RUN python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"

# Now switch to non-root user
USER appuser

EXPOSE 8000

# CMD for Railway/Render (uses $PORT, workers=1 for free tier)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --no-access-log"]