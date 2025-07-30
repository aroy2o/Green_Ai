import pytesseract
import easyocr
from PIL import Image
import io
import re
from app.text_processing import clean_and_structure_text
from loguru import logger

import torch

async def process_image_ocr(image_bytes: bytes):
    engines = ["tesseract", "easyocr"]
    gpu_available = False
    try:
        gpu_available = torch.cuda.is_available()
    except Exception:
        pass
    for engine in engines:
        try:
            if engine == "tesseract":
                text = pytesseract.image_to_string(Image.open(io.BytesIO(image_bytes)))
            elif engine == "easyocr":
                reader = easyocr.Reader(["en"], gpu=gpu_available)
                result = reader.readtext(image_bytes, detail=0)
                text = "\n".join(result)
            else:
                continue
            cleaned, structured = clean_and_structure_text(text)
            if cleaned.strip() and len(cleaned.strip()) > 5:
                return {
                    "status": "success",
                    "engine_used": engine.capitalize(),
                    "full_text": cleaned,
                    "structured_text": structured
                }
        except Exception as e:
            logger.warning(f"OCR engine {engine} failed: {e}")
            continue
    # If all engines fail or no meaningful text
    return {
        "status": "fail",
        "error": "No readable text detected. Please upload a clear image or enter key details manually."
    }
