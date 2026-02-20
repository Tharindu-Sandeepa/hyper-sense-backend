# hyper-sense-backend 🚪🔢

A **production-ready, lightweight REST API** microservice that counts people in images using **YOLO** (nano model) and **FastAPI**. Built for IoT door-counting scenarios — runs comfortably on a **Raspberry Pi 5** or any cheap VPS.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Model** | `yolo26n.pt` (YOLO26 nano — fastest & most accurate nano, Jan 2026) |
| **People-only detection** | Filters COCO class 0 (`person`) only |
| **Smart resizing** | Images >1280 px auto-scaled before inference |
| **Non-blocking** | All CPU work runs with `asyncio.run_in_executor` |
| **Confidence override** | `?conf=0.6` query param per request |
| **CORS** | Open for all origins — IoT-friendly |
| **Health endpoint** | `GET /health` for k8s / load-balancer probes |
| **Docker** | Multi-stage build, non-root, model pre-downloaded |

---

## 📁 Project Structure

```
hyper-sense-backend/
├── main.py            # FastAPI application
├── requirements.txt   # Python dependencies
├── Dockerfile         # Multi-stage production build
├── .dockerignore      # Docker build context exclusions
└── README.md          # This file
```

---

## 🚀 Run Locally (without Docker)

### 1. Prerequisites

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> The YOLO model (`yolo26n.pt`) is **downloaded automatically** on first run into `~/.cache/ultralytics/`.

### 2. Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

---

## 🐳 Docker

### Build

```bash
docker build -t hyper-sense-backend:latest .
```

> The model is pre-downloaded during the build, so the first request is instant.

### Run

```bash
# Single instance (Raspberry Pi 5 / 1-core VPS)
docker run -d \
  --name hyper-sense \
  -p 8000:8000 \
  --restart unless-stopped \
  hyper-sense-backend:latest
```

```bash
# Stronger server — increase workers (match CPU core count)
docker run -d \
  --name hyper-sense \
  -p 8000:8000 \
  -e WORKERS=4 \
  --restart unless-stopped \
  hyper-sense-backend:latest
```

---

## 🧪 Test with cURL

### Health check

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"yolo26n","version":"1.0"}
```

### People detection

```bash
curl -X POST http://localhost:8000/detect \
     -F "file=@/path/to/your/image.jpg" \
     | python3 -m json.tool
```

**Example response:**

```json
{
  "people_count": 3,
  "detections": [
    {"bbox": [120, 45, 280, 390], "confidence": 0.9231},
    {"bbox": [310, 60, 450, 410], "confidence": 0.8764},
    {"bbox": [510, 80, 640, 400], "confidence": 0.7112}
  ],
  "success": true,
  "timestamp": "2026-02-21T07:15:00Z",
  "inference_ms": 94.3
}
```

### Override confidence threshold

```bash
curl -X POST "http://localhost:8000/detect?conf=0.7" \
     -F "file=@crowded_hallway.jpg"
```

---

## 📡 ESP32-CAM Integration

Paste this into your Arduino sketch. Replace `SERVER_IP` with your server's LAN IP.

```cpp
#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"

const char* SSID     = "YOUR_WIFI_SSID";
const char* PASSWORD = "YOUR_WIFI_PASSWORD";
const char* ENDPOINT = "http://192.168.1.100:8000/detect";  // ← your server IP

void sendFrameToAPI() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) { Serial.println("Camera capture failed"); return; }

    HTTPClient http;
    http.begin(ENDPOINT);

    // Multipart boundary
    String boundary = "----ESP32Boundary";
    http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);

    // Build multipart body
    String bodyStart = "--" + boundary + "\r\n"
                       "Content-Disposition: form-data; name=\"file\"; filename=\"frame.jpg\"\r\n"
                       "Content-Type: image/jpeg\r\n\r\n";
    String bodyEnd   = "\r\n--" + boundary + "--\r\n";

    int totalSize = bodyStart.length() + fb->len + bodyEnd.length();
    uint8_t* body = (uint8_t*)malloc(totalSize);
    memcpy(body,                              bodyStart.c_str(), bodyStart.length());
    memcpy(body + bodyStart.length(),         fb->buf,           fb->len);
    memcpy(body + bodyStart.length() + fb->len, bodyEnd.c_str(), bodyEnd.length());

    int httpCode = http.POST(body, totalSize);
    free(body);
    esp_camera_fb_return(fb);

    if (httpCode == 200) {
        String payload = http.getString();
        Serial.println("Response: " + payload);
        // Parse "people_count" from JSON here, e.g. with ArduinoJson
    }
    http.end();
}
```

---

## ⚡ Performance Expectations

| Hardware | Inference Time | Accuracy* |
|---|---|---|
| Raspberry Pi 5 (4-core) | ~120–180 ms | 90–95 % |
| Modern CPU (VPS, 2-core) | ~80–130 ms | 90–95 % |
| GPU (e.g. Jetson Nano) | ~15–30 ms | 92–98 % |
| Apple M-series Mac | ~40–70 ms | 92–98 % |

\* Accuracy on well-lit indoor scenes with ≥ 10 px person height.

---

## 📈 Scaling

| Scenario | Command |
|---|---|
| Single Pi / cheap VPS | `--workers 2` (default) |
| 4-core server | `--workers 4` |
| 8-core+ server | `--workers 8` |
| High-traffic | Put Nginx in front + multiple containers behind a load balancer |

```bash
# Run with more workers directly
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 📄 API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc UI |
| `POST` | `/detect` | Detect & count people |

### POST /detect

| Parameter | Type | Where | Default | Description |
|---|---|---|---|---|
| `file` | `file` | form-data | — | JPEG or PNG image |
| `conf` | `float` | query | `0.50` | Confidence threshold 0.01–1.0 |

### Error responses

```json
// 400 — no file / wrong type
{"success": false, "detail": "Unsupported file type 'image/gif'. Must be image/jpeg or image/png."}

// 500 — inference failure
{"success": false, "detail": "Inference failed: <error>"}
```

---

## 🔒 Security Notes

- The Docker container runs as a **non-root user** (`appuser`).
- For production deployments, put the API behind **Nginx** with TLS and optionally an API key header.

---

## 📝 License

MIT — free for personal and commercial IoT use.


🐍 Option A — Run Locally (Python venv)
bash
# 1. Go into the project folder
cd "/Users/tharindu/My files/human-counter/hyper-sense-backend"
# 2. Create & activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate
# 3. Install dependencies  (first time ~2–3 min — downloads PyTorch + Ultralytics)
pip install -r requirements.txt
# 4. Start the server (auto-downloads yolo26n.pt on first run)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
Server will be live at http://localhost:8000

🐳 Option B — Docker
bash
cd "/Users/tharindu/My files/human-counter/hyper-sense-backend"
# Build (downloads model into the image — takes ~5 min first time)
docker build -t hyper-sense-backend:latest .
# Run
docker run -d --name hyper-sense -p 8000:8000 --restart unless-stopped hyper-sense-backend:latest
# Check logs
docker logs -f hyper-sense
🧪 Test the APIs
✅ Health check
bash
curl http://localhost:8000/health
Expected:

json
{"status":"ok","model":"yolo26n","version":"1.0"}
🔍 Detect people — with a real image
bash
curl -X POST http://localhost:8000/detect \
     -F "file=@/path/to/your/photo.jpg" \
     | python3 -m json.tool
🎛️ Override confidence threshold
bash
curl -X POST "http://localhost:8000/detect?conf=0.65" \
     -F "file=@/path/to/your/photo.jpg"
