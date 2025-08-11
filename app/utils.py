from fastapi import UploadFile
import io
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageFile, UnidentifiedImageError
from loguru import logger
import numpy as np
import re
import os

# Allow loading of truncated images instead of failing hard
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Try optional OpenCV
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

# Try optional HEIC/HEIF support
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover
    pass

# Optional ImageIO as an additional decoder fallback (installed via scikit-image)
try:  # pragma: no cover - exercised indirectly
    import imageio.v3 as iio  # type: ignore
except Exception:  # pragma: no cover
    iio = None

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
# Optional allow-list via env (comma-separated, e.g., "jpeg,png,webp"). If not set, accept any Pillow-supported image.
_ALLOWED_ENV = os.getenv("ALLOWED_IMAGE_FORMATS")
ALLOWED_TYPES = (
    {fmt.strip().lower() for fmt in _ALLOWED_ENV.split(",") if fmt.strip()} if _ALLOWED_ENV else None
)


def compute_blur_brightness_contrast(pil_img):
    arr = np.array(pil_img.convert("L"))
    lap = 0.0
    if cv2 is not None and arr.shape[0] > 2 and arr.shape[1] > 2:
        try:
            lap = float(cv2.Laplacian(arr, cv2.CV_64F).var())
        except Exception:
            lap = 0.0
    brightness = float(arr.mean())
    contrast = float(arr.std())
    return lap, brightness, contrast


def preprocess_image(pil_img, skip_denoise=False, contrast_factor=1.5, sharpness_factor=1.5):
    img = pil_img.convert("RGB")
    if not skip_denoise:
        img = img.filter(ImageFilter.MedianFilter(size=3))
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    img = ImageEnhance.Sharpness(img).enhance(sharpness_factor)
    return img

# --- Deblurring and adaptive enhancement for marginally blurry images ---

def enhance_blurry_image(pil_img):
    """Light enhancement for marginally blurry/low-contrast images using OpenCV.
    Applies CLAHE for local contrast and a gentle unsharp mask via Laplacian.
    Falls back to original image on any error.
    """
    if cv2 is None:
        return pil_img
    try:
        arr = np.array(pil_img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        # Local contrast boost
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(gray)
        # Unsharp-like enhancement via Laplacian
        lap = cv2.Laplacian(g, cv2.CV_16S, ksize=3)
        lap_abs = cv2.convertScaleAbs(lap)
        sharp = cv2.addWeighted(g, 1.5, lap_abs, -0.5, 0)
        # Convert back to RGB
        enhanced = cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(enhanced)
    except Exception as e:  # pragma: no cover
        logger.warning(f"Deblurring/enhancement failed: {e}")
        return pil_img


def crop_to_roi(pil_img, bbox):
    arr = np.array(pil_img)
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(arr.shape[1], int(x2)), min(arr.shape[0], int(y2))
    if x2 <= x1 or y2 <= y1:
        return pil_img
    roi = arr[y1:y2, x1:x2]
    return Image.fromarray(roi)

async def validate_image_upload(file: UploadFile):
    contents = await file.read()
    if not contents:
        # Some clients send 0 bytes but correct filename/content-type. Keep message for tests.
        raise ValueError("Corrupted image file. Please upload a valid image.")
    if len(contents) > MAX_FILE_SIZE:
        raise ValueError("File too large. Please upload an image ≤5MB.")

    img = None
    img_format = ""

    # Helper to finalize return tuple from a PIL image
    def _finalize_from_pil(pil: Image.Image, prefer_png=False):
        nonlocal img_format
        buf = io.BytesIO()
        # Preserve original format if known, otherwise choose PNG for lossless
        fmt = (getattr(pil, "format", None) or img_format or ("PNG" if prefer_png else "JPEG")).upper()
        if fmt == "PNG":
            pil.save(buf, format="PNG", optimize=True)
            img_format = img_format or "png"
        else:
            pil.save(buf, format="JPEG", quality=95, subsampling=0)
            img_format = img_format or "jpeg"
        return buf.getvalue()

    # 1) Try Pillow first (with HEIF registered if available)
    try:
        img = Image.open(io.BytesIO(contents))
        # If animated (GIF/WebP/TIFF), take the first frame
        if getattr(img, "is_animated", False):
            try:
                img.seek(0)
            except Exception:
                pass
        # Force full decode to catch truncated/corrupt files
        img.load()
        # Preserve original format before conversion (convert clears .format)
        original_format = (getattr(img, "format", None) or "").lower()
        img = img.convert("RGB")
        img_format = original_format or img_format
    except UnidentifiedImageError as err_pil:
        logger.debug(f"Pillow could not identify image: {err_pil}")
        img = None
    except Exception as err_pil:
        logger.debug(f"Pillow failed to open image: {err_pil}")
        img = None

    # 1b) Pillow incremental parser as a softer fallback for slightly corrupted files
    if img is None:
        try:
            parser = ImageFile.Parser()
            parser.feed(contents)
            parsed = parser.close()
            # Parsed images often lack format metadata; infer from filename if possible
            img = parsed.convert("RGB")
            if not img_format:
                _, ext = os.path.splitext(file.filename or "")
                img_format = ext.lstrip(".").lower() or ""
        except Exception as err_par:
            logger.debug(f"Pillow parser failed: {err_par}")
            img = None

    # 2) Fallback: try OpenCV decoding if Pillow failed
    if img is None:
        try:
            if cv2 is not None:
                arr = np.frombuffer(contents, dtype=np.uint8)
                mat = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                if mat is not None:
                    # Normalize to RGB
                    try:
                        if len(mat.shape) == 2:
                            rgb = cv2.cvtColor(mat, cv2.COLOR_GRAY2RGB)
                        else:
                            rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
                    except Exception:
                        rgb = mat if len(mat.shape) == 3 else np.stack([mat]*3, axis=-1)
                    pil = Image.fromarray(rgb)
                    contents = _finalize_from_pil(pil, prefer_png=True)
                    img = pil
                    img_format = img_format or "png"
                else:
                    raise ValueError("OpenCV failed to decode image bytes")
            else:
                raise ValueError("Pillow failed and OpenCV unavailable")
        except Exception as err_cv:
            logger.debug(f"OpenCV fallback failed: {err_cv}")
            img = None

    # 3) Final fallback: ImageIO (handles a wide range including some TIFF/WEBP cases)
    if img is None and iio is not None:
        try:
            arr = iio.imread(contents, index=0)
            if arr is not None:
                if arr.ndim == 2:
                    rgb = np.stack([arr]*3, axis=-1)
                elif arr.shape[2] == 4:
                    # Drop alpha channel for consistency
                    rgb = arr[..., :3]
                else:
                    rgb = arr
                pil = Image.fromarray(rgb.astype(np.uint8))
                contents = _finalize_from_pil(pil, prefer_png=True)
                img = pil
                img_format = img_format or "png"
        except Exception as err_iio:
            logger.debug(f"ImageIO fallback failed: {err_iio}")
            img = None

    if img is None:
        # Keep message compatible with tests: must contain 'Corrupted image file'
        raise ValueError("Corrupted image file. Please upload a valid image.")

    # If an env-specified allow-list exists, enforce it; otherwise accept any Pillow-recognized image
    if ALLOWED_TYPES is not None and (img_format or "unknown") not in ALLOWED_TYPES:
        raise ValueError(
            f"Unsupported image format '{img_format}'. Allowed formats: {', '.join(sorted(ALLOWED_TYPES))}."
        )

    # If format is missing, try infer from filename extension for downstream logging
    if not img_format:
        _, ext = os.path.splitext(file.filename or "")
        img_format = ext.lstrip(".").lower() or "unknown"

    return contents, {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(contents),
        "format": img_format,
    }

# --- Dynamic threshold tuning based on live metrics ---
class ThresholdTuner:
    """Singleton for live adjustment of blur/contrast/brightness and detection thresholds."""
    _blur = 50
    _brightness = 40
    _contrast = 20
    _conf_threshold = 0.6

    @classmethod
    def update(cls, blur=None, brightness=None, contrast=None, conf=None):
        # Called by metrics logger after each run
        if blur is not None:
            cls._blur = blur
        if brightness is not None:
            cls._brightness = brightness
        if contrast is not None:
            cls._contrast = contrast
        if conf is not None:
            cls._conf_threshold = conf

    @classmethod
    def get_thresholds(cls):
        return dict(blur=cls._blur, brightness=cls._brightness, contrast=cls._contrast, conf_threshold=cls._conf_threshold)

# --- Fast post-processing for text cleaning ---

def fast_text_postprocess(text):
    # Remove split words, join lines, filter detection noise
    text = re.sub(r"-\s*\n", "", text)  # Remove hyphenated line breaks
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ")
    return text.strip()
