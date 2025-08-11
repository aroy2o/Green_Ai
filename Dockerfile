# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Environment for predictable Python behavior and performance
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    UVICORN_WORKERS=2

# Install system dependencies for Tesseract and Pillow (no system OpenCV)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libtesseract-dev \
        libleptonica-dev \
        pkg-config \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user and switch
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Copy app code
COPY --chown=appuser:appuser app ./app

# Expose port
EXPOSE 8000

# Healthcheck (simple check on /health)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 CMD ["python", "-c", "import json,urllib.request,sys; url='http://127.0.0.1:8000/health';\n\ntry:\n    r=urllib.request.urlopen(url, timeout=2)\n    data=json.loads(r.read().decode('utf-8'))\n    sys.exit(0 if data.get('status')=='ok' else 1)\nexcept Exception:\n    sys.exit(1)"]

# Run the API (use multiple workers for CPU-bound parallelism)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-2}"]
