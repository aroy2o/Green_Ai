from fastapi import UploadFile
import io
from PIL import Image, ImageFilter, ImageEnhance
from loguru import logger
import numpy as np

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_TYPES = {"jpeg", "png", "jpg"}

def compute_blur_brightness_contrast(pil_img):
    arr = np.array(pil_img.convert("L"))
    # Blur: Laplacian variance
    lap = np.abs(np.diff(arr, 2)).var() if arr.shape[0] > 2 and arr.shape[1] > 2 else 0
    # Brightness: mean
    brightness = arr.mean()
    # Contrast: std
    contrast = arr.std()
    return float(lap), float(brightness), float(contrast)

def preprocess_image(pil_img):
    # Denoise (median), enhance contrast, sharpen
    img = pil_img.convert("RGB")
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    return img

# --- Deblurring and adaptive enhancement for marginally blurry images ---
def enhance_blurry_image(pil_img):
    import cv2
    arr = np.array(pil_img)
    # Deblur using simple Wiener filter (OpenCV, fallback if Real-ESRGAN not available)
    try:
        # Convert to grayscale for deblurring
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        # Use OpenCV's Laplacian for edge enhancement
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharp = cv2.convertScaleAbs(gray + 0.7 * laplacian)
        # Adaptive histogram equalization for contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(sharp)
        # Merge back to 3 channels
        enhanced_rgb = cv2.merge([enhanced, enhanced, enhanced])
        pil_enhanced = Image.fromarray(enhanced_rgb)
        return pil_enhanced
    except Exception as e:
        logger.warning(f"Deblurring/enhancement failed: {e}")
        return pil_img

async def validate_image_upload(file: UploadFile):
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise ValueError("File too large. Please upload an image ≤5MB.")
    # Try Pillow for robust type detection
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
        img_format = img.format.lower() if img.format else None
    except Exception:
        raise ValueError("Corrupted image file. Please upload a valid JPG or PNG.")
    # Accept both 'jpeg' and 'jpg' as valid
    if img_format in ("jpeg", "jpg", "png"):
        return contents, {"filename": file.filename, "content_type": file.content_type, "size": len(contents), "format": img_format}
    # If Pillow can open it, but format is not standard, allow but warn
    logger.warning(f"Image format '{img_format}' is not standard, but Pillow can open it. Accepting upload.")
    return contents, {"filename": file.filename, "content_type": file.content_type, "size": len(contents), "format": img_format}
