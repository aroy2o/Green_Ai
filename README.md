
# Green AI - Smart Product Analysis Pipeline

An end-to-end pipeline for product label analysis using OCR, object detection, and MongoDB storage. Built with FastAPI, supports Tesseract and EasyOCR, YOLOv8 object detection, image quality enhancement, and a modern frontend.

---


## What's New: Accuracy & Detection Stats

### 📈 Improved Accuracy & Detection

- **Enhanced Image Quality Handling:** The pipeline now computes blur, brightness, and contrast for every image, and attempts automatic enhancement for marginal images. This results in higher OCR and object detection accuracy, especially for images that are slightly blurry or dark.
- **Parallel OCR & Object Detection:** OCR and object detection run in parallel, reducing processing time and improving throughput.
- **Dynamic Error Feedback:** If the system cannot extract meaningful text or objects, it provides specific feedback based on image quality, helping users recapture better images.
- **Rolling Metrics Dashboard:** The `/metrics` endpoint and frontend dashboard visualize rolling averages for blur, brightness, contrast, detection confidence, CPU usage, and error rate over the last 100 images. This allows you to monitor system health and model performance in real time.

#### Example Metrics Dashboard (Live Visualization)

```
┌──────────────────────────────────────────────┐
│ 📊 Green AI Metrics Dashboard                │
├───────────────┬─────────────┬───────────────┤
│ Avg Blur      │   92.3      │ (Sharp)       │
│ Avg Brightness│   68.1      │ (Good)        │
│ Avg Contrast  │   28.7      │ (High)        │
│ Avg Confidence│   0.87      │ (YOLOv8)      │
│ Error Rate    │   3%        │ (Last 100)    │
│ Proc Time     │   0.82s     │ (Avg)         │
└───────────────┴─────────────┴───────────────┘
```

*The dashboard updates live as new images are processed, helping you track improvements and spot issues instantly.*

---
## Features

- **Image Upload & Camera Support**: Upload product images or capture via camera (frontend ready, see `index.html`).
- **OCR**: Extracts text using Tesseract and EasyOCR (auto-selects best result).
- **Object Detection**: YOLOv8-based detection of packaging types (bottle, can, box, etc.) and material guessing.
- **Image Quality Metrics**: Computes blur, brightness, and contrast; attempts enhancement for marginal images.
- **MongoDB Storage**: Stores OCR results, image info, and analytics in MongoDB (with GridFS support).
- **Metrics Dashboard**: Minimal in-memory metrics endpoint for monitoring.
- **Docker Support**: Ready-to-deploy Dockerfile for easy containerization.
- **Testing**: Pytest-based API tests for upload, validation, and error handling.

---

## Folder Structure

```
.
├── Dockerfile
├── index.html
├── README.md
├── requirements.txt
├── yolov8n.pt
├── app/
│   ├── db.py                # MongoDB/AsyncIOMotor integration
│   ├── logger.py            # Loguru logger config
│   ├── main.py              # FastAPI app, endpoints, metrics
│   ├── object_detection.py  # YOLOv8 object detection & material guessing
│   ├── ocr.py               # OCR logic (Tesseract, EasyOCR)
│   ├── text_processing.py   # Text cleaning and structuring
│   ├── utils.py             # Image validation, enhancement, metrics
│   └── __pycache__/         # Python bytecode cache
├── tests/
│   ├── sample.txt           # Invalid file for negative test
│   ├── sample1.jpg          # Sample image for tests
│   └── test_api.py          # Pytest API tests
```

---

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- Pillow
- pytesseract
- easyocr
- pymongo
- motor
- python-multipart
- loguru
- pytest
- ultralytics (for YOLOv8)

See `requirements.txt` for details. You may need to install system dependencies for Tesseract and OpenCV (see Dockerfile for reference).

---

## Installation

```bash
pip install -r requirements.txt
```

---

## MongoDB Setup

- Ensure MongoDB is running and accessible (default: `mongodb://localhost:27017`).
- For image storage, GridFS is used (handled by pymongo/motor).
- You can override DB connection via `MONGO_URI` and `MONGO_DB` environment variables.

---

## Running the API

### Local (development)

```bash
uvicorn app.main:app --reload
```

### With Docker

Build the image:

```bash
docker build -t ocr-pipeline .
```

Run the container:

```bash
docker run -p 8000:8000 \
  -e MONGO_URI="mongodb://host.docker.internal:27017" \
  -e MONGO_DB=ocr_pipeline \
  -v $(pwd)/yolov8n.pt:/app/yolov8n.pt \
  ocr-pipeline
```

*Note: Adjust `MONGO_URI` as needed for your environment. The YOLOv8 weights file (`yolov8n.pt`) must be present in the project root and mounted into the container.*

---

## API Endpoints

### `POST /ocr`

**Description:**
Analyze a product image for text and packaging type.

**Request:**
- `file`: image (JPEG/PNG, ≤5MB, multipart form-data)
- `source`: (optional) string, e.g. 'upload' or 'camera'
- `retake_count`: (optional) int, number of retakes

**Response:**
- On success:
  - `status`: "success"
  - `engine_used`: "Tesseract" or "Easyocr"
  - `full_text`: cleaned OCR text
  - `structured_text`: dict with product_name, ingredients, features, other_info
  - `object_detection`: list of detected objects with class, bounding box, confidence, material guess
  - `quality`: blur, brightness, contrast
- On failure:
  - `status`: "fail"
  - `error`: error message
  - `object_detection`: always present (may be empty)

### `GET /metrics`

Returns recent API usage and performance metrics (in-memory, for dashboard).

---

## Testing

Run all tests with:

```bash
pytest
```

Test coverage includes:
- Valid image upload and OCR
- Invalid file type handling
- Large file rejection

See `tests/test_api.py` for details.

---

## Example Usage

```bash
curl -F "file=@tests/sample1.jpg" http://localhost:8000/ocr
```

---

## Codebase Overview

- **app/main.py**: FastAPI app, endpoints, and metrics logger. Handles `/ocr` and `/metrics` endpoints, image validation, preprocessing, OCR, object detection, and DB logging.
- **app/ocr.py**: Runs OCR using Tesseract and EasyOCR, returns cleaned and structured text.
- **app/object_detection.py**: Loads YOLOv8 model, detects packaging objects, and guesses material type.
- **app/db.py**: Async MongoDB integration using Motor, stores OCR results and analytics.
- **app/utils.py**: Image validation, preprocessing, enhancement, and quality metrics (blur, brightness, contrast).
- **app/text_processing.py**: Cleans and structures OCR text into product name, ingredients, features, and other info.
- **app/logger.py**: Configures Loguru logging for the app.
- **index.html**: Modern frontend UI for uploads and camera capture (see comments in file for JS logic).
- **tests/**: Pytest-based API tests and sample files for validation.

---

## Notes

- For best results, upload clear, well-lit images of product labels.
- If the image is blurry or dark, the API will attempt enhancement before OCR.
- MongoDB is required for persistent storage; for stateless demo/testing, comment out DB calls in `app/db.py`.
- YOLOv8 model weights (`yolov8n.pt`) must be present in the project root.
- System dependencies for Tesseract and OpenCV may be required (see Dockerfile).

---

# Green_Ai by GREEN DUKAN 
