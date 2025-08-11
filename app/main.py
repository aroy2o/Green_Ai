from fastapi import FastAPI, File, UploadFile, HTTPException, status, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import io
import numpy as np
from PIL import Image, ImageOps
from app.ocr import process_image_ocr
from app.db import store_ocr_result
from app.utils import validate_image_upload, preprocess_image, compute_blur_brightness_contrast, enhance_blurry_image, crop_to_roi, ThresholdTuner
from app.object_detection import detect_objects_and_materials
from loguru import logger
import time
import psutil
import threading
import os
import collections
# Optional dotenv
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False

# --- New: Pydantic response models for strict schemas ---
from app.schemas import OCRResponse, MetricsResponse, HealthResponse

app = FastAPI(title="OCR Product Analysis Pipeline")

# Global concurrency guard for heavy inference work
_INFER_SEM = asyncio.Semaphore(int(os.getenv("MAX_INFER_CONCURRENCY", "4")))
_DET_TIMEOUT = float(os.getenv("DETECTION_TIMEOUT_SEC", "6"))
_OCR_TIMEOUT = float(os.getenv("OCR_TIMEOUT_SEC", "10"))
_ROI_OCR_TIMEOUT = float(os.getenv("ROI_OCR_TIMEOUT_SEC", "4"))
_MAX_IMG_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "1920"))

# CORS (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve only a safe static directory if present (avoid exposing project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
public_dir = os.path.join(project_root, "public")
if os.path.isdir(public_dir):
    app.mount("/static", StaticFiles(directory=public_dir), name="static")


@app.on_event("startup")
async def startup_event():
    # Load environment from .env if present and pre-warm optional heavy models
    try:
        load_dotenv()
    except Exception:
        pass
    # Warm up models in background to reduce first-request latency
    async def _warm():
        try:
            from app.object_detection import get_yolo_model
            await asyncio.to_thread(get_yolo_model)
        except Exception:
            pass
        try:
            # Touch EasyOCR reader
            from app.ocr import _get_easyocr_reader
            await asyncio.to_thread(_get_easyocr_reader)
        except Exception:
            pass
    asyncio.create_task(_warm())


@app.get("/")
async def root():
    # Serve index.html from project root if present; otherwise provide a simple message
    idx = os.path.join(project_root, "index.html")
    if os.path.isfile(idx):
        return FileResponse(idx)
    return JSONResponse({"message": "Green AI API is running. See /docs for API specs."})


@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        proc = psutil.Process()
        mem = proc.memory_info().rss
        cpu = proc.cpu_percent(interval=None)
    except Exception:
        mem = 0
        cpu = 0
    return HealthResponse(status="ok", cpu_percent=cpu, rss_bytes=mem)


@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(
    request: Request,
    file: UploadFile = File(...),
    source: str = Form(None),
    retake_count: int = Form(0)
):
    start_time = time.time()
    # Process-level CPU metric
    try:
        proc = psutil.Process()
        _ = proc.cpu_percent(interval=None)  # prime
    except Exception:
        proc = None

    try:
        # Validate image (type, size, integrity)
        image_bytes, image_info = await validate_image_upload(file)
    except ValueError as ve:
        logger.warning(f"Image validation failed: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error during image validation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during image validation.")

    # Add source and retake_count to image_info for DB logging
    image_info = dict(image_info) if image_info else {}
    image_info["source"] = source or "upload"
    image_info["retake_count"] = retake_count

    # Preprocess image and compute quality metrics
    pil_img = Image.open(io.BytesIO(image_bytes))
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    # Optional downscale for very large images to speed up processing
    w, h = pil_img.size
    m = max(w, h)
    if m > _MAX_IMG_SIDE:
        scale = _MAX_IMG_SIDE / float(m)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    blur, brightness, contrast = compute_blur_brightness_contrast(pil_img)
    image_info["blur"] = blur
    image_info["brightness"] = brightness
    image_info["contrast"] = contrast
    segment_times = {}
    segment_times['validate'] = time.time() - start_time

    # If marginally blurry/dark, try enhancement before OCR
    marginal_blur = 40 < blur < 100 or 30 < brightness < 60 or 15 < contrast < 25
    restoration_attempted = False
    if marginal_blur:
        logger.info(f"Sharpening image for better detection... please wait")
        pil_img = enhance_blurry_image(pil_img)
        restoration_attempted = True
        blur, brightness, contrast = compute_blur_brightness_contrast(pil_img)
        image_info["blur"] = blur
        image_info["brightness"] = brightness
        image_info["contrast"] = contrast
    segment_times['preprocess'] = time.time() - start_time - segment_times['validate']

    # Serialize processed PIL image back to bytes, preserving original format when possible
    buf = io.BytesIO()
    orig_fmt = (image_info.get("format") or "jpeg").lower()
    if orig_fmt == "png":
        pil_img.save(buf, format="PNG", optimize=True)
    else:
        pil_img.save(buf, format="JPEG", quality=95, subsampling=0)
    processed_bytes = buf.getvalue()

    # Offload detection and OCR to threads with concurrency guard and timeouts
    conf_threshold = ThresholdTuner.get_thresholds().get("conf_threshold", 0.6)
    async with _INFER_SEM:
        det_future = asyncio.to_thread(detect_objects_and_materials, processed_bytes, True, conf_threshold)
        ocr_future = asyncio.to_thread(process_image_ocr, processed_bytes)
        try:
            objects_task = asyncio.wait_for(det_future, timeout=_DET_TIMEOUT)
        except Exception:
            objects_task = asyncio.to_thread(lambda: [])
        try:
            ocr_task = asyncio.wait_for(ocr_future, timeout=_OCR_TIMEOUT)
        except Exception:
            ocr_task = asyncio.to_thread(lambda: {"status": "fail", "error": "OCR timeout"})
        try:
            objects_detected, ocr_result = await asyncio.gather(objects_task, ocr_task, return_exceptions=False)
        except asyncio.TimeoutError:
            # Fallback if both timed out
            objects_detected, ocr_result = [], {"status": "fail", "error": "Processing timeout"}
    segment_times['inference'] = time.time() - start_time - segment_times['validate'] - segment_times['preprocess']

    # Crop to ROI and retry OCR if initial result is poor
    if ocr_result.get("status") != "success" and objects_detected:
        texts = []
        for obj in objects_detected[:4]:  # limit number of ROIs
            roi_img = crop_to_roi(pil_img, obj["bounding_box"])
            buf_roi = io.BytesIO()
            # Save ROI as high-quality JPEG to keep small text edges
            roi_img.save(buf_roi, format="JPEG", quality=95, subsampling=0)
            roi_bytes = buf_roi.getvalue()
            try:
                roi_ocr_future = asyncio.to_thread(process_image_ocr, roi_bytes)
                roi_ocr = await asyncio.wait_for(roi_ocr_future, timeout=_ROI_OCR_TIMEOUT)
            except asyncio.TimeoutError:
                continue
            if roi_ocr.get("status") == "success":
                texts.append(roi_ocr.get("full_text", ""))
        if texts:
            ocr_result["full_text"] = "\n".join([t for t in texts if t])
            if ocr_result["full_text"]:
                ocr_result["status"] = "success"
    segment_times['postprocess'] = time.time() - start_time - segment_times['validate'] - segment_times['preprocess'] - segment_times['inference']

    # Refined post-processing for text
    if ocr_result.get("full_text"):
        text = ocr_result["full_text"]
        text = text.replace("-\n", "")
        text = " ".join(text.splitlines())
        text = " ".join(text.split())
        ocr_result["full_text"] = text

    avg_conf = 0
    if objects_detected:
        avg_conf = sum([o["confidence"] for o in objects_detected]) / len(objects_detected)

    # Adaptive tuning of detection threshold (small, bounded adjustments)
    try:
        current = ThresholdTuner.get_thresholds().get("conf_threshold", 0.6)
        target = 0.65 if avg_conf >= 0.75 else (0.55 if avg_conf <= 0.5 else current)
        new_conf = current + max(-0.05, min(0.05, target - current))
        ThresholdTuner.update(conf=round(new_conf, 2))
    except Exception:
        pass

    response_json = {
        "status": ocr_result.get("status", "fail"),
        "engine_used": ocr_result.get("engine_used"),
        "ocr": {
            "full_text": ocr_result.get("full_text", ""),
            "structured_text": ocr_result.get("structured_text", {})
        },
        "object_detection": objects_detected,
        "quality": {"blur": blur, "brightness": brightness, "contrast": contrast},
        "latency": segment_times,
        "restoration_attempted": restoration_attempted
    }

    if response_json["status"] == "fail" and not objects_detected:
        response_json["error"] = "Image quality is poor (blur/brightness/contrast). Please retake or upload a clearer image."

    try:
        cpu_usage = proc.cpu_percent(interval=0.05) if proc else 0.0
        mem_usage = proc.memory_info().rss if proc else 0
    except Exception:
        cpu_usage = 0.0
        mem_usage = 0

    MetricsLogger.log(blur, brightness, contrast, avg_conf, cpu_usage, time.time()-start_time, error=(response_json["status"]=="fail"))
    try:
        await store_ocr_result(
            image_info,
            response_json,
            engine_used=ocr_result.get("engine_used"),
            processing_time=time.time()-start_time
        )
    except Exception:
        # Storage is best-effort; do not fail request
        pass

    # Return typed response for validation & docs
    return OCRResponse(**response_json)

# --- Minimal in-memory metrics logger for /metrics endpoint ---
class MetricsLogger:
    _lock = threading.Lock()
    _maxlen = 100
    _data = collections.deque(maxlen=_maxlen)
    @classmethod
    def log(cls, blur, brightness, contrast, conf, cpu, proc_time, error):
        with cls._lock:
            cls._data.append({
                "blur": blur, "brightness": brightness, "contrast": contrast,
                "conf": conf, "cpu": cpu, "proc_time": proc_time, "error": error,
                "timestamp": time.time()
            })
    @classmethod
    def get_metrics(cls):
        with cls._lock:
            data = list(cls._data)
        if not data:
            return {
                "avg_blur": 0.0,
                "avg_brightness": 0.0,
                "avg_contrast": 0.0,
                "avg_confidence": 0.0,
                "avg_cpu": 0.0,
                "error_rate": 0.0,
                "avg_latency": 0.0,
                "latency_trend": [],
                "accuracy_trend": [],
                "cpu_trend": [],
                "count": 0,
            }
        avg = lambda k: round(sum(d[k] for d in data)/len(data), 3)
        error_rate = round(100*sum(1 for d in data if d["error"])/len(data), 2)
        latency = [d["proc_time"] for d in data]
        conf_trend = [d["conf"] for d in data]
        cpu_trend = [d["cpu"] for d in data]
        return {
            "avg_blur": avg("blur"),
            "avg_brightness": avg("brightness"),
            "avg_contrast": avg("contrast"),
            "avg_confidence": avg("conf"),
            "avg_cpu": avg("cpu"),
            "error_rate": error_rate,
            "avg_latency": avg("proc_time"),
            "latency_trend": latency[-10:],
            "accuracy_trend": conf_trend[-10:],
            "cpu_trend": cpu_trend[-10:],
            "count": len(data)
        }

@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """Return rolling averages for dashboard charting."""
    return MetricsLogger.get_metrics()
