# OCR Product Analysis Pipeline

## Requirements

- Python 3.8+
- FastAPI
- uvicorn
- Pillow
- pytesseract
- easyocr
- pymongo
- motor
- python-multipart
- loguru
- pytest

## Installation

```bash
pip install -r requirements.txt
```

## MongoDB
- Ensure MongoDB is running and accessible.
- For image storage, GridFS is used (handled by pymongo/motor).

## Running the API

```bash
uvicorn app.main:app --reload
```

## Testing

```bash
pytest
```

## Docker (optional)

Build and run:
```bash
docker build -t ocr-pipeline .
docker run -p 8000:8000 ocr-pipeline
```

## API Usage

POST `/ocr` with multipart form-data:
- `file`: image (JPEG/PNG, ≤5MB)

## Response
- On success: JSON with status, engine used, full and structured text
- On failure: JSON with status, error message, and user guidance
# Green_Ai
