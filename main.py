"""
hyper-sense-backend — main.py
=============================
FastAPI microservice that:
  • Accepts an image (JPEG / PNG) via  POST /detect
  • Runs YOLO inference to count people (class 0)
  • Returns JSON with count, bounding boxes, and metadata
  • Exposes GET /health for orchestration / load-balancer checks

Production notes
----------------
• The model is loaded ONCE at startup (lifespan context) and reused
  for every request, so GPU/CPU allocation stays warm.
• Images wider than MAX_INFER_SIZE px are down-scaled *before* inference;
  this gives ≈80-150 ms latency on a modern CPU without harming accuracy.
• All heavy numpy/PIL work runs inside run_in_executor so the event loop
  never blocks for more than a few microseconds.
"""

from __future__ import annotations

import io
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

# ─── Logging ──────────────────────────────────────────────────────────────────
# Keep logs minimal in production; change to DEBUG locally.
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
logger = logging.getLogger("hyper-sense")

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_NAME = "yolo26n.pt"          # nano model — smallest & fastest in YOLO26 family
PERSON_CLASS_ID = 0                # COCO class 0 = "person"
DEFAULT_CONF = 0.50                # minimum detection confidence
MAX_INFER_SIZE = 1280              # resize images larger than this before inference
VERSION = "1.0"

# ─── Global model handle (set in lifespan) ────────────────────────────────────
model: YOLO | None = None


# ─── Lifespan: load model once, release on shutdown ───────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context.
    Ultralytics auto-downloads yolo26n.pt on first run and caches it in
    ~/.cache/ultralytics/.  Subsequent starts load from disk in <1 s.
    """
    global model
    logger.info("⏳ Loading YOLO model: %s …", MODEL_NAME)
    loop = asyncio.get_running_loop()
    # Load on a thread so we don't block the event loop during startup
    model = await loop.run_in_executor(None, YOLO, MODEL_NAME)
    logger.info("✅ Model ready — %s", MODEL_NAME)
    yield
    # Cleanup (optional, but good practice)
    logger.info("🛑 Shutting down — releasing model.")
    del model


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="hyper-sense-backend",
    description="Lightweight YOLO-powered people-counting REST API for IoT door sensors.",
    version=VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins so ESP32-CAM / Pis on any IP can reach the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _preprocess(data: bytes) -> np.ndarray:
    """
    Decode raw bytes → RGB PIL image → numpy array.
    Down-scales the longer edge to MAX_INFER_SIZE while preserving aspect ratio.
    This keeps inference time predictable even for 12-MP ESP32-CAM images.
    """
    img = Image.open(io.BytesIO(data)).convert("RGB")

    # Resize if the image is very large (saves 2-3× on inference time)
    w, h = img.size
    if max(w, h) > MAX_INFER_SIZE:
        scale = MAX_INFER_SIZE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        logger.debug("Resized %dx%d → %dx%d", w, h, img.width, img.height)

    return np.array(img)


def _run_inference(image_array: np.ndarray, conf: float) -> list[dict[str, Any]]:
    """
    Run YOLO inference synchronously.
    Returns a list of detection dicts filtered to class 0 (person).
    Called via run_in_executor so the event loop stays non-blocking.
    """
    results = model.predict(
        source=image_array,
        conf=conf,
        classes=[PERSON_CLASS_ID],  # only detect people
        verbose=False,              # suppress per-inference console noise
    )

    detections: list[dict[str, Any]] = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "bbox": [round(x1), round(y1), round(x2), round(y2)],
                    "confidence": round(float(box.conf[0]), 4),
                }
            )
    return detections


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    summary="Health check",
    response_description="Service status and model name",
    tags=["Monitoring"],
)
async def health_check():
    """
    Lightweight health endpoint.
    Load-balancers and k8s probes should call this every 10–30 s.
    """
    return {"status": "ok", "model": MODEL_NAME.replace(".pt", ""), "version": VERSION}


@app.post(
    "/detect",
    summary="Detect and count people in an image",
    response_description="People count plus optional bounding boxes",
    tags=["Inference"],
)
async def detect_people(
    file: UploadFile = File(..., description="JPEG or PNG image to analyse"),
    conf: float = Query(
        default=DEFAULT_CONF,
        ge=0.01,
        le=1.0,
        description="Minimum confidence threshold (0.01 – 1.0). Default: 0.50",
    ),
):
    """
    POST /detect

    - **file**: image file (multipart/form-data, field name `file`)
    - **conf**: optional confidence override via query string, e.g. `?conf=0.6`

    Returns:
    ```json
    {
      "people_count": 3,
      "detections": [
        {"bbox": [x1, y1, x2, y2], "confidence": 0.87}
      ],
      "success": true,
      "timestamp": "2026-02-21T12:00:00Z"
    }
    ```
    """
    # ── Validate content-type ────────────────────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Must be image/jpeg or image/png.",
        )

    # ── Read bytes ───────────────────────────────────────────────────────────
    try:
        data = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {exc}")

    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # ── Preprocess on thread (CPU-bound) ─────────────────────────────────────
    loop = asyncio.get_running_loop()
    try:
        image_array = await loop.run_in_executor(None, _preprocess, data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}")

    # ── Run YOLO inference on thread (CPU/GPU-bound) ──────────────────────────
    t0 = time.perf_counter()
    try:
        detections = await loop.run_in_executor(None, _run_inference, image_array, conf)
    except Exception as exc:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info("Detected %d person(s) in %.1f ms [conf=%.2f]", len(detections), elapsed_ms, conf)

    return {
        "people_count": len(detections),
        "detections": detections,
        "success": True,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inference_ms": elapsed_ms,    # handy for debugging latency
    }


# ─── 400 / 422 JSON error responses ──────────────────────────────────────────
# FastAPI already returns JSON for HTTPException; this override keeps the
# shape consistent for 422 Unprocessable Entity (missing file field, etc.)
from fastapi.exceptions import RequestValidationError
from fastapi import Request

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"success": False, "detail": exc.errors()},
    )


# ─── Dev entry-point (not used inside Docker) ─────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
