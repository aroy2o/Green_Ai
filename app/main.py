from fastapi import FastAPI, File, UploadFile, HTTPException, status, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import io
import numpy as np
from PIL import Image
from app.ocr import process_image_ocr
from app.db import store_ocr_result
from app.utils import validate_image_upload, preprocess_image, compute_blur_brightness_contrast
from app.object_detection import detect_objects_and_materials
from loguru import logger
import time
import psutil
import threading
import os

app = FastAPI(title="OCR Product Analysis Pipeline")

# Serve static files (index.html and others)
static_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    # Serve index.html from project root
    return FileResponse(os.path.join(static_dir, "index.html"))



@app.post("/ocr")
async def ocr_endpoint(
    request: Request,
    file: UploadFile = File(...),
    source: str = Form(None),
    retake_count: int = Form(0)
):
    start_time = time.time()
    cpu_start = psutil.cpu_percent(interval=None)
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
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    blur, brightness, contrast = compute_blur_brightness_contrast(pil_img)
    image_info["blur"] = blur
    image_info["brightness"] = brightness
    image_info["contrast"] = contrast

    # If marginally blurry/dark, try enhancement before OCR
    marginal_blur = 40 < blur < 100 or 30 < brightness < 60 or 15 < contrast < 25
    restoration_attempted = False
    if marginal_blur:
        from app.utils import enhance_blurry_image
        logger.info(f"Marginal image quality detected (blur={blur:.1f}, brightness={brightness:.1f}, contrast={contrast:.1f}). Attempting enhancement.")
        pil_img = enhance_blurry_image(pil_img)
        restoration_attempted = True
        # Recompute metrics after enhancement
        blur, brightness, contrast = compute_blur_brightness_contrast(pil_img)
        image_info["blur_post_enhance"] = blur
        image_info["brightness_post_enhance"] = brightness
        image_info["contrast_post_enhance"] = contrast

    pil_img = preprocess_image(pil_img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    processed_bytes = buf.getvalue()

    # Run OCR and object detection in parallel (tune detection threshold for live capture)
    import asyncio
    try:
        ocr_task = asyncio.create_task(process_image_ocr(processed_bytes))
        loop = asyncio.get_event_loop()
        detect_task = loop.run_in_executor(None, detect_objects_and_materials, processed_bytes, True, 0.5)  # lower conf for live
        ocr_result, objects_detected = await asyncio.gather(ocr_task, detect_task)
        # If OCR failed and restoration was attempted, return specific error
        if ocr_result.get("status") == "fail" and restoration_attempted:
            logger.info("Restoration attempted but OCR still failed.")
            ocr_result["error"] = "We tried to enhance and clarify the image, but text was still unreadable. Please recapture with better focus and lighting."
    except Exception as e:
        logger.error(f"Pipeline parallel execution failed: {e}")
        ocr_result = {"status": "fail", "error": "OCR and detection failed."}
        objects_detected = []

    # Unified response structure
    response_json = {
        "status": ocr_result.get("status", "fail"),
        "engine_used": ocr_result.get("engine_used"),
        "ocr": {
            "full_text": ocr_result.get("full_text", ""),
            "structured_text": ocr_result.get("structured_text", {})
        },
        "object_detection": objects_detected,
        "quality": {"blur": blur, "brightness": brightness, "contrast": contrast}
    }

    # If error or all stages fail, check quality and return specific error if poor
    poor_quality = blur < 80 or brightness < 60 or contrast < 25
    if response_json["status"] == "fail" and not objects_detected:
        if poor_quality:
            if restoration_attempted:
                response_json["error"] = ocr_result.get("error", "We tried to enhance and clarify the image, but text was still unreadable. Please recapture with better focus and lighting.")
            else:
                response_json["error"] = "No readable text detected; try again with a clearer, brighter image."
        else:
            response_json["error"] = ocr_result.get("error", "Meaningful text or object could not be detected. Please try again or specify details.")
        # Log restoration attempts
        image_info["restoration_attempted"] = restoration_attempted
        await store_ocr_result(image_info, response_json, engine_used=None, processing_time=time.time()-start_time)
        MetricsLogger.log(blur, brightness, contrast, 0, psutil.cpu_percent(interval=None)-cpu_start, time.time()-start_time, error=True)
        return JSONResponse(status_code=200, content=response_json)

    # Store successful OCR + detection result in DB
    avg_conf = 0
    if objects_detected:
        avg_conf = float(np.mean([o.get("confidence", 0) for o in objects_detected]))
    MetricsLogger.log(blur, brightness, contrast, avg_conf, psutil.cpu_percent(interval=None)-cpu_start, time.time()-start_time, error=False)
    await store_ocr_result(
        image_info,
        response_json,
        engine_used=ocr_result.get("engine_used"),
        processing_time=time.time()-start_time
    )
    return JSONResponse(status_code=200, content=response_json)

# --- Minimal in-memory metrics logger for /metrics endpoint ---
import collections
class MetricsLogger:
    _lock = threading.Lock()
    _maxlen = 100
    _data = collections.deque(maxlen=_maxlen)
    @classmethod
    def log(cls, blur, brightness, contrast, conf, cpu, proc_time, error):
        with cls._lock:
            cls._data.append({
                "ts": time.time(),
                "blur": blur,
                "brightness": brightness,
                "contrast": contrast,
                "confidence": conf,
                "cpu": cpu,
                "proc_time": proc_time,
                "error": error
            })
    @classmethod
    def get_metrics(cls):
        with cls._lock:
            items = list(cls._data)
        if not items:
            return {}
        avg = lambda k: float(np.mean([x[k] for x in items]))
        err_rate = sum(1 for x in items if x["error"])/len(items)
        # Return both rolling averages and the full recent history (for charting)
        return {
            "avg_blur": avg("blur"),
            "avg_brightness": avg("brightness"),
            "avg_contrast": avg("contrast"),
            "avg_confidence": avg("confidence"),
            "avg_cpu": avg("cpu"),
            "avg_proc_time": avg("proc_time"),
            "error_rate": err_rate,
            "count": len(items),
            "history": [
                {
                    "ts": x["ts"],
                    "cpu": x["cpu"],
                    "confidence": x["confidence"],
                    "error": 1.0 if x["error"] else 0.0
                } for x in items
            ]
        }

@app.get("/metrics")
async def metrics():
    """Return rolling averages for dashboard charting."""
    return MetricsLogger.get_metrics()
