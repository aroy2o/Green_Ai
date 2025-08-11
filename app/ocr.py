import os
import pytesseract
from PIL import Image, ImageOps
import io
import re
import numpy as np
from app.text_processing import clean_and_structure_text
from loguru import logger

# Optional torch and GPU detection
try:
    import torch
    _gpu_available = torch.cuda.is_available()
except Exception:
    torch = None
    _gpu_available = False

# Try optional OpenCV for preprocessing
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

# Add Sauvola threshold from scikit-image for tough low-contrast text
try:
    from skimage.filters import threshold_sauvola  # type: ignore
except Exception:  # pragma: no cover
    threshold_sauvola = None

_EASY_OCR_READER = None

# Common words to boost scoring on product labels
_COMMON_LABEL_WORDS = {
    "permanent", "marker", "pen", "bullet", "tip", "waterproof",
    "luxor", "fine", "medium", "broad", "ink", "black", "blue",
    "red", "green", "made", "india", "manufactured", "brand"
}


def _get_easyocr_reader():
    """Get or create the EasyOCR reader instance."""
    global _EASY_OCR_READER
    if _EASY_OCR_READER is None:
        try:
            import easyocr  # type: ignore
        except Exception as e:  # pragma: no cover
            logger.warning(f"EasyOCR unavailable: {e}")
            return None
        _EASY_OCR_READER = easyocr.Reader(["en"], gpu=_gpu_available)
    return _EASY_OCR_READER


def _bytes_to_numpy_rgb(image_bytes: bytes):
    """Convert image bytes to RGB numpy array."""
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(pil)


def _auto_rotate(pil: Image.Image) -> Image.Image:
    """Rotate using Tesseract OSD if needed."""
    try:
        osd = pytesseract.image_to_osd(pil)
        m = re.search(r"Rotate:\s+(\d+)", osd)
        if m:
            angle = int(m.group(1)) % 360
            if angle in (90, 180, 270):
                pil = pil.rotate(360 - angle, expand=True)
    except Exception:
        pass
    return pil


def _deskew(pil: Image.Image, max_angle: float = 15.0) -> Image.Image:
    """Deskew using minAreaRect of edges. Limits rotation to +/- max_angle degrees.
    Falls back to original on any error.
    """
    if cv2 is None:
        return pil
    try:
        arr = np.array(pil.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        # Light blur then edge detect
        g = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(g, 60, 180)
        ys, xs = np.where(edges > 0)
        if len(xs) < 50:
            return pil
        pts = np.column_stack((xs, ys)).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        angle = rect[-1]
        # OpenCV returns angle in [-90, 0)
        if angle < -45:
            angle = 90 + angle
        if abs(angle) < 0.5 or abs(angle) > max_angle:
            return pil
        # Rotate around center
        (h, w) = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(rotated)
    except Exception:
        return pil


def _estimate_blur(pil: Image.Image) -> float:
    """Estimate blur via Laplacian variance (grayscale)."""
    if cv2 is None:
        return 0.0
    try:
        gray = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def _scale_min_side(pil: Image.Image, min_side: int = 1200) -> Image.Image:
    w, h = pil.size
    s = max(1.0, float(min_side) / float(min(w, h)))
    if s > 1.01:
        pil = pil.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return pil


def _dynamic_scale(pil: Image.Image) -> Image.Image:
    """Dynamically upscale based on blur. Heavier upscaling for blurrier images."""
    default_side = int(os.getenv("OCR_MIN_SIDE", 1400))
    b = _estimate_blur(pil)
    target = default_side
    if b and b < 15:
        target = max(target, 2400)
    elif b and b < 35:
        target = max(target, 2000)
    elif b and b < 60:
        target = max(target, 1600)
    return _scale_min_side(pil, target)


def _apply_gamma(pil: Image.Image, gamma: float = 1.25) -> Image.Image:
    try:
        arr = np.asarray(pil).astype(np.float32) / 255.0
        arr = np.power(arr, 1.0 / max(1e-3, float(gamma)))
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    except Exception:
        return pil


def _prep_variants(pil: Image.Image):
    """Generate a small set of preprocessed variants for robust OCR."""
    variants = [pil, pil.convert("L"), _apply_gamma(pil, 1.3)]
    if cv2 is None:
        return [v for v in variants if v is not None]

    arr = np.array(pil)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # CLAHE to boost local contrast
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g1 = clahe.apply(gray)
    except Exception:
        g1 = gray
    # LAB L-channel CLAHE
    try:
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)
        Lc = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(L)
        lab2 = cv2.merge([Lc, A, B])
        lab_rgb = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
        labL = cv2.cvtColor(lab_rgb, cv2.COLOR_RGB2GRAY)
    except Exception:
        labL = g1
    # Bilateral filter to denoise while preserving edges
    try:
        g1b = cv2.bilateralFilter(g1, 7, 50, 50)
    except Exception:
        g1b = g1
    # Unsharp-like sharpen
    blur = cv2.GaussianBlur(g1b, (0, 0), 1.0)
    sharp = cv2.addWeighted(g1b, 1.6, blur, -0.6, 0)

    # Illumination correction: top-hat / black-hat
    kernel = np.ones((3, 3), np.uint8)
    tophat = cv2.morphologyEx(sharp, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(sharp, cv2.MORPH_BLACKHAT, kernel)

    # Adaptive thresholds
    thr = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    thr_inv = cv2.bitwise_not(thr)
    # OTSU threshold
    try:
        _, thr_otsu = cv2.threshold(g1b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception:
        thr_otsu = thr

    # Sauvola threshold for severe uneven illumination
    if threshold_sauvola is not None:
        try:
            window = 25
            k = 0.2
            sau_t = threshold_sauvola(g1b, window_size=window, k=k)
            sau = (g1b > sau_t).astype(np.uint8) * 255
            sau = sau.astype(np.uint8)
        except Exception:
            sau = None
    else:
        sau = None

    # Morph close to connect thin strokes
    thr_close = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Morph gradient (emphasize edges)
    grad = cv2.morphologyEx(g1b, cv2.MORPH_GRADIENT, kernel)

    variants.extend([
        Image.fromarray(g1),
        Image.fromarray(labL),
        Image.fromarray(g1b),
        Image.fromarray(sharp),
        Image.fromarray(tophat),
        Image.fromarray(blackhat),
        Image.fromarray(thr),
        Image.fromarray(thr_inv),
        Image.fromarray(thr_otsu),
        Image.fromarray(thr_close),
        Image.fromarray(grad),
        Image.fromarray(sau) if sau is not None else None,
    ])
    return [v for v in variants if v is not None]


def _score_text(t: str) -> int:
    t = (t or "").strip()
    if not t:
        return 0
    alnum = re.sub(r"[^A-Za-z0-9]+", "", t)
    base = len(alnum)
    # Boost when expected label words appear
    words = {w.lower() for w in re.findall(r"[A-Za-z]+", t)}
    bonus = sum(6 for w in words if w in _COMMON_LABEL_WORDS and len(w) >= 3)
    # Penalize if only punctuation
    if base == 0 and len(t) > 0:
        return 0
    return base + bonus


def _mser_rois(pil: Image.Image):
    """Detect candidate text ROIs using MSER. Returns list of (x1,y1,x2,y2)."""
    if cv2 is None:
        return []
    arr = np.array(pil)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    try:
        mser = cv2.MSER_create(_min_area=60)
    except Exception:
        mser = cv2.MSER_create()
    regions, bboxes = mser.detectRegions(gray)
    rois = []
    h, w = gray.shape[:2]
    for (x, y, bw, bh) in bboxes:
        if bw < 12 or bh < 10:
            continue
        aspect = bw / float(bh)
        if aspect < 0.8:  # allow more vertical text too
            continue
        area = bw * bh
        if area < 80 or area > 0.6 * w * h:
            continue
        pad = max(4, int(0.06 * max(bw, bh)))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
        rois.append((x1, y1, x2, y2))

    # Non-maximum suppression
    if not rois:
        return rois
    boxes = np.array(rois, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = areas.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 1)
        h1 = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w1 * h1
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= 0.3)[0]
        order = order[inds + 1]
    return boxes[keep].astype(np.int32).tolist()


def _warp_quad(arr: np.ndarray, quad, pad: int = 6) -> Image.Image:
    """Perspective rectify a quadrilateral ROI to a fronto-parallel crop."""
    if cv2 is None:
        # Fallback to axis-aligned crop
        xs = [int(p[0]) for p in quad]; ys = [int(p[1]) for p in quad]
        x1, y1, x2, y2 = max(0, min(xs)-pad), max(0, min(ys)-pad), min(arr.shape[1], max(xs)+pad), min(arr.shape[0], max(ys)+pad)
        return Image.fromarray(arr[y1:y2, x1:x2])
    pts = np.array(quad, dtype=np.float32)
    # Order points roughly TL, TR, BR, BL
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    # Compute target size
    def dist(a, b):
        return float(np.linalg.norm(a - b))
    width = int(max(dist(br, bl), dist(tr, tl))) + 2 * pad
    height = int(max(dist(tr, br), dist(tl, bl))) + 2 * pad
    width = max(16, min(width, 4096))
    height = max(16, min(height, 4096))
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[pad, pad], [width-pad, pad], [width-pad, height-pad], [pad, height-pad]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(arr, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped)


def _tess_configs():
    """Yield a curated set of Tesseract configs for tough images."""
    allow = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:+-()/%.,' \t"
    base = [
        "--oem 3 --psm 6 -l eng",
        "--oem 3 --psm 7 -l eng",
        "--oem 1 --psm 7 -l eng",
        "--oem 3 --psm 11 -l eng",
        "--oem 3 --psm 12 -l eng",
        "--oem 1 --psm 13 -l eng",
        "--oem 3 --psm 3 -l eng",
        "--oem 3 --psm 8 -l eng",
    ]
    for b in base:
        for invert in (0, 1):
            yield f"{b} -c tessedit_char_whitelist={allow} -c preserve_interword_spaces=1 -c tessedit_do_invert={invert} -c user_defined_dpi=300"


def process_image_ocr(image_bytes: bytes):
    """Process the image for OCR and return structured text data."""
    engines_env = os.getenv("OCR_ENGINES")
    engines = [e.strip().lower() for e in engines_env.split(",") if e.strip()] if engines_env else ["tesseract", "easyocr"]

    # Load, normalize, rotate, deskew, dynamically scale
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil = ImageOps.exif_transpose(pil)
    pil = _auto_rotate(pil)
    pil = _deskew(pil)
    pil = _dynamic_scale(pil)

    best_text = ""
    best_engine = None
    best_score = 0

    # Prepare variants once
    variants = _prep_variants(pil)

    for engine in engines:
        try:
            if engine == "tesseract":
                for img in variants:
                    for cfg in _tess_configs():
                        text = pytesseract.image_to_string(img, config=cfg)
                        score = _score_text(text)
                        if score > best_score:
                            best_score, best_text, best_engine = score, text, "Tesseract"
                        if best_score >= 16:
                            break
                    if best_score >= 16:
                        break

                # MSER-based ROI re-OCR if full-frame weak
                if best_score < 16:
                    rois = _mser_rois(pil)
                    arr = np.array(pil)
                    for (x1, y1, x2, y2) in rois[:12]:
                        roi = Image.fromarray(arr[y1:y2, x1:x2])
                        roi = _scale_min_side(roi, 600)
                        for cfg in (
                            "--oem 3 --psm 7 -l eng -c preserve_interword_spaces=1 -c user_defined_dpi=300",
                            "--oem 1 --psm 7 -l eng -c preserve_interword_spaces=1 -c user_defined_dpi=300",
                            "--oem 3 --psm 6 -l eng -c preserve_interword_spaces=1 -c user_defined_dpi=300",
                            "--oem 3 --psm 12 -l eng -c preserve_interword_spaces=1 -c user_defined_dpi=300",
                            "--oem 1 --psm 13 -l eng -c preserve_interword_spaces=1 -c user_defined_dpi=300",
                        ):
                            t = pytesseract.image_to_string(roi, config=cfg)
                            s = _score_text(t)
                            if s > best_score:
                                best_score, best_text, best_engine = s, t, "Tesseract+MSER-ROI"
                            if best_score >= 18:
                                break
                        if best_score >= 18:
                            break

            elif engine == "easyocr":
                reader = _get_easyocr_reader()
                if reader is None:
                    continue
                arr = np.array(pil)
                results = reader.readtext(arr, detail=1, paragraph=True)
                texts = [r[1] for r in results if len(r) >= 2]
                joined = "\n".join(texts)
                score = _score_text(joined)
                if score > best_score:
                    best_score, best_text, best_engine = score, joined, "EasyOCR"

                # ROI refine with Tesseract on top boxes (perspective-rectified and axis-aligned)
                if results:
                    def box_area(b):
                        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = b
                        xs = [x1, x2, x3, x4]; ys = [y1, y2, y3, y4]
                        return abs((max(xs) - min(xs)) * (max(ys) - min(ys)))
                    results = sorted(results, key=lambda r: box_area(r[0]), reverse=True)[:10]

                    for b, _, _ in results:
                        # Perspective-rectified crop
                        roi_warp = _warp_quad(arr, b, pad=8)
                        roi_warp = _scale_min_side(roi_warp, 600)
                        # Axis-aligned crop as fallback
                        xs = [int(p[0]) for p in b]; ys = [int(p[1]) for p in b]
                        x1, y1, x2, y2 = max(0, min(xs)), max(0, min(ys)), min(arr.shape[1], max(xs)), min(arr.shape[0], max(ys))
                        pad = 8
                        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                        x2, y2 = min(arr.shape[1], x2 + pad), min(arr.shape[0], y2 + pad)
                        roi_axis = Image.fromarray(arr[y1:y2, x1:x2])
                        roi_axis = _scale_min_side(roi_axis, 600)

                        for roi in (roi_warp, roi_axis):
                            for cfg in (
                                "--oem 3 --psm 7 -l eng -c preserve_interword_spaces=1 -c user_defined_dpi=300",
                                "--oem 1 --psm 7 -l eng -c preserve_interword_spaces=1 -c user_defined_dpi=300",
                                "--oem 3 --psm 6 -l eng -c preserve_interword_spaces=1 -c user_defined_dpi=300",
                                "--oem 3 --psm 12 -l eng -c preserve_interword_spaces=1 -c user_defined_dpi=300",
                                "--oem 1 --psm 13 -l eng -c preserve_interword_spaces=1 -c user_defined_dpi=300",
                            ):
                                t = pytesseract.image_to_string(roi, config=cfg)
                                s = _score_text(t)
                                if s > best_score:
                                    best_score, best_text, best_engine = s, t, "EasyOCR+Tesseract-ROI"
                                if best_score >= 18:
                                    break
                            if best_score >= 18:
                                break
        except Exception as e:
            logger.warning(f"OCR engine {engine} failed: {e}")
            continue

        if best_score >= 18:
            break

    if best_text and _score_text(best_text) > 4:
        cleaned, structured = clean_and_structure_text(best_text)
        return {
            "status": "success",
            "engine_used": best_engine,
            "full_text": cleaned,
            "structured_text": structured,
        }

    return {
        "status": "fail",
        "error": "No readable text detected. Please retake in better light, hold steady, and fill the frame with the label.",
    }
