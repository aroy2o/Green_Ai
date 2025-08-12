# Green AI - Smart Product Analysis Pipeline

An end-to-end pipeline for product label analysis using OCR, object detection, and MongoDB storage. Built with FastAPI, supports Tesseract and EasyOCR, YOLOv8 object detection, image quality enhancement, and a modern frontend.

---


## What's New: Accuracy, Robustness & Ops

- Pydantic response models added (strict schemas) for /ocr, /metrics, /health.
- Faster, more robust image validation: supports JPEG/PNG/WebP/GIF/TIFF/HEIC, OpenCV and ImageIO fallbacks; fewer false "corrupted image" errors.
- Speedups and stability for YOLO: thread-safe lazy loader, half-precision on CUDA, fused layers, vectorized box extraction, and adaptive input size.
- Slimmer, more secure Docker images: headless OpenCV via pip, non-root user, optional GPU profile, healthcheck.
- Production guidance: use multiple Uvicorn workers (2–4) for CPU-bound parallelism; concurrency guard for in-process tasks.
- Sustainability analysis module and endpoint `/sustainability` leveraging GPT-3.5-turbo for evidence-based lifecycle assessments using ONLY OCR text (backend-only; API key in env).

## Environment

- OPENAI_API_KEY must be present in environment (.env or secret store). Never expose to frontend.
- Optional overrides:
  - `SUSTAINABILITY_MODEL` (default: gpt-3.5-turbo)
  - `SUSTAINABILITY_TEMPERATURE` (default: 0.25)
  - `SUSTAINABILITY_MAX_TOKENS` (default: 1536)

---

## Features

- **Image Upload & Camera Support**: Upload product images or capture via camera (frontend ready, see `index.html`).
- **OCR**: Extracts text using Tesseract and EasyOCR (auto-selects best result).
- **Object Detection**: YOLOv8-based detection of packaging types and material guessing.
- **Image Quality Metrics**: Computes blur, brightness, and contrast; enhances marginal images.
- **MongoDB Storage**: Stores OCR results, image info, and analytics in MongoDB (with GridFS support).
- **Metrics Dashboard**: Minimal in-memory metrics endpoint for monitoring.
- **Docker Support**: CPU and GPU Dockerfiles, healthcheck, non-root user.
- **Testing**: Pytest-based API tests for upload, validation, and error handling.

---

## Folder Structure

```
.
├── Dockerfile                 # Slim, non-root, headless OpenCV, healthcheck
├── Dockerfile.gpu             # CUDA profile (use --gpus all)
├── index.html
├── README.md
├── requirements.txt
├── yolov8n.pt
├── app/
│   ├── db.py                # MongoDB/AsyncIOMotor integration
│   ├── logger.py            # Loguru logger config
│   ├── main.py              # FastAPI app, endpoints, metrics (typed responses)
│   ├── object_detection.py  # YOLOv8 detection & material guessing (optimized)
│   ├── ocr.py               # OCR logic (Tesseract, EasyOCR)
│   ├── schemas.py           # Pydantic response models
│   ├── text_processing.py   # Text cleaning and structuring
│   ├── utils.py             # Image validation, enhancement, metrics (robust)
│   └── __pycache__/         # Python bytecode cache
├── tests/
│   ├── sample.txt           # Invalid file for negative test
│   ├── sample1.jpg          # Sample image for tests
│   └── test_api.py          # Pytest API tests
```

---

## Requirements

- Python 3.10+
- See `requirements.txt` for package list. System deps: Tesseract (libs are installed in Dockerfiles).

---

## Installation (local)

```bash
pip install -r requirements.txt
```

---

## MongoDB Setup

- Ensure MongoDB is running and accessible (default: `mongodb://localhost:27017`).
- Override via `MONGO_URI` and `MONGO_DB` environment variables.

---

## Running the API

### Local (development)

```bash
uvicorn app.main:app --reload
```

### With Docker (CPU)

The image installs `opencv-python-headless` from pip (no system OpenCV). Base image is slim and runs as non-root.

Build:

```bash
docker build -t ocr-pipeline .
```

Run (2–4 workers recommended on CPU):

```bash
docker run -p 8000:8000 \
  -e UVICORN_WORKERS=4 \
  -e MONGO_URI="mongodb://host.docker.internal:27017" \
  -e MONGO_DB=ocr_pipeline \
  -v $(pwd)/yolov8n.pt:/app/yolov8n.pt \
  ocr-pipeline
```

### With Docker (GPU)

Use `Dockerfile.gpu` with the official PyTorch CUDA runtime and run with `--gpus all`:

```bash
docker build -f Dockerfile.gpu -t ocr-pipeline-gpu .
# Docker 19.03+
docker run --gpus all -p 8000:8000 \
  -e UVICORN_WORKERS=2 \
  -v $(pwd)/yolov8n.pt:/app/yolov8n.pt \
  ocr-pipeline-gpu
```

YOLO/EasyOCR will use CUDA automatically when available.

---
## API Endpoints

- `POST /ocr` – Typed response: OCR text, detection list, quality metrics, latency.
- `POST /sustainability` – Application/json: structured sustainability JSON with positives/negatives, explanations with sources, recommendations, and limited_analysis flag.
- `GET /metrics` – Rolling averages for dashboard.
- `GET /health` – Lightweight health info.

See `/docs` for OpenAPI.

---
## Testing

```bash
pytest
```

---
## Production Notes

- **Workers**: set `UVICORN_WORKERS=2..4` depending on CPU cores and memory.
- **Concurrency**: set `MAX_INFER_CONCURRENCY` to limit in-process tasks.
- **Timeouts**: tweak `DETECTION_TIMEOUT_SEC`, `OCR_TIMEOUT_SEC`, `ROI_OCR_TIMEOUT_SEC`.
- **Image limits**: `MAX_IMAGE_SIDE` for downscaling oversized uploads; `ALLOWED_IMAGE_FORMATS` to restrict formats (e.g., `jpeg,png,webp`).

---
## Troubleshooting

- Getting "Corrupted image file" for a valid image?
  - Ensure the client sends multipart/form-data with correct `Content-Type` and actual file bytes.
  - We now support JPEG, PNG, WebP, GIF (first frame), TIFF, HEIC/HEIF. HEIC needs `pillow-heif` (already in requirements).
  - Some camera apps embed unusual metadata; try re-saving the image or removing EXIF, or test with another image to isolate the issue.
  - Check logs for detailed parser/OpenCV/ImageIO fallback errors (debug level).

- YOLO model not loading:
  - Ensure `yolov8n.pt` is present in project root or set `YOLO_WEIGHTS` env to the correct path.

---
## Security & Hardening

- Runs as non-root in Docker; only required apt packages installed.
- Avoids system OpenCV; uses headless build from pip to reduce attack surface and image size.
- CORS is wide-open by default for dev; restrict `allow_origins` in production.
- Validates image size (<=5MB by default) and decodes safely with Pillow/OpenCV/ImageIO fallbacks.
