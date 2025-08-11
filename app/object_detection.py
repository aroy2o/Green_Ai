import os
import io
import numpy as np
from PIL import Image
from loguru import logger
import threading

# Material guessing map and confidence
MATERIAL_MAP = {
    "bottle": ("plastic", 0.9),
    "can": ("metal", 0.95),
    "box": ("paper/cardboard", 0.85),
    "carton": ("paper/cardboard", 0.85),
    "wrapper": ("plastic or foil", 0.7),
    "sachet": ("plastic or foil", 0.7),
    "barcode": ("label/paper/plastic", 0.6),
    "cup": ("plastic or paper", 0.7),
    "wine glass": ("glass", 0.95),
}

try:
    import torch
    _cuda = torch.cuda.is_available()
except Exception:
    torch = None
    _cuda = False

# Optional OpenCV
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

_DISABLE_DETECTION = os.getenv("DISABLE_DETECTION", "0") == "1"

_yolo_model = None
_model_lock = threading.Lock()
_model_device = None  # cache where the model resides


def get_yolo_model():
    """Thread-safe singleton loader for YOLOv8 model. Import lazily to avoid hard dependency at import time."""
    global _yolo_model, _model_device
    if _yolo_model is not None:
        return _yolo_model
    with _model_lock:
        if _yolo_model is not None:
            return _yolo_model
        weights_path = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")
        if not os.path.exists(weights_path):
            logger.warning(f"YOLO weights not found at {weights_path}. Object detection disabled.")
            return None
        try:
            from ultralytics import YOLO  # type: ignore
            model = YOLO(weights_path)
            # Move to CUDA if available to avoid repeated transfers
            if torch is not None and _cuda:
                try:
                    model.to("cuda")
                    # Optional half precision for faster inference on GPU
                    try:
                        model.model.half()
                    except Exception:
                        pass
                    _model_device = "cuda"
                except Exception:
                    _model_device = "cpu"
            else:
                _model_device = "cpu"
            # Fuse conv+bn layers if available for speed
            try:
                model.fuse()
            except Exception:
                pass
            _yolo_model = model
        except Exception as e:
            logger.warning(f"Failed to load YOLO model: {e}")
            return None
    return _yolo_model


def guess_material(label):
    """Return material guess and confidence for a given label."""
    return MATERIAL_MAP.get(label, ("unknown", 0.4))


def _label_from_names(names, class_id: int) -> str:
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))
    if isinstance(names, list):
        return names[class_id] if 0 <= class_id < len(names) else str(class_id)
    return str(class_id)


def _compute_blur(arr: np.ndarray) -> float:
    if cv2 is None:
        return 0.0
    try:
        g = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())
    except Exception:
        return 0.0


def _enhance_for_detection(arr: np.ndarray, blur_val: float) -> np.ndarray:
    """Light enhancement for detection on blurry images using CLAHE + unsharp on L channel."""
    if cv2 is None:
        return arr
    try:
        if blur_val >= 60:
            return arr  # already sharp enough
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        Lc = clahe.apply(L)
        blur = cv2.GaussianBlur(Lc, (0, 0), 1.0)
        Lsharp = cv2.addWeighted(Lc, 1.5, blur, -0.5, 0)
        lab2 = cv2.merge([Lsharp, A, B])
        rgb = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
        return rgb
    except Exception:
        return arr


def detect_objects_and_materials(image_bytes: bytes, use_gpu=True, conf_threshold=0.6):
    """
    Detect objects and guess material from image bytes.
    Returns list of dicts with class, bounding_box, confidence, material_guess, material_confidence, low_confidence.
    """
    if _DISABLE_DETECTION:
        return []

    model = get_yolo_model()
    if model is None:
        return []

    # Select device without forcing CUDA context if unavailable
    device = 0 if (use_gpu and _cuda) else "cpu"

    # Decode image
    try:
        img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    except Exception:
        return []

    # Dynamic imgsz based on blur and size for small objects on blurry frames
    blur_val = _compute_blur(img)
    min_side = min(img.shape[0], img.shape[1])
    imgsz = 640
    if blur_val < 12:
        imgsz = 1280
    elif blur_val < 30 or min_side > 1000:
        imgsz = 960

    # Light enhancement for detection if blurry
    img_in = _enhance_for_detection(img, blur_val)
    img_in = np.ascontiguousarray(img_in)

    try:
        # Use Ultralytics predict API so we can pass conf/iou directly
        # Use inference_mode/no_grad for speed
        if torch is not None:
            ctx = torch.inference_mode()
        else:
            class _Noop:
                def __enter__(self):
                    return None
                def __exit__(self, exc_type, exc, tb):
                    return False
            ctx = _Noop()
        with ctx:
            results = model.predict(
                source=img_in,
                device=device,
                imgsz=imgsz,
                conf=conf_threshold,
                iou=0.5,
                verbose=False,
                max_det=100
            )
    except Exception as e:
        logger.warning(f"YOLO inference failed: {e}")
        return []

    names = getattr(model, "names", None)
    h, w = img.shape[:2]
    objects = []
    for r in results:
        boxes = getattr(r, 'boxes', None)
        if boxes is None:
            continue
        try:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            clss = boxes.cls.detach().cpu().numpy().astype(int)
        except Exception:
            # Fallback per-box iteration
            for box in boxes or []:
                try:
                    class_id = int(box.cls[0])
                    label = _label_from_names(names, class_id)
                    conf = float(box.conf[0])
                    if conf < conf_threshold:
                        continue
                    xyxy_box = box.xyxy[0].detach().cpu().numpy().tolist()
                    x1, y1, x2, y2 = [int(round(x)) for x in xyxy_box]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    material, mat_conf = guess_material(label)
                    low_conf = conf < 0.7 or mat_conf < 0.7
                    objects.append({
                        "class": label,
                        "bounding_box": [x1, y1, x2, y2],
                        "confidence": round(conf, 3),
                        "material_guess": material,
                        "material_confidence": round(mat_conf, 2),
                        "low_confidence": low_conf,
                    })
                except Exception:
                    continue
            continue
        # Vectorized path
        for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, clss):
            if conf < conf_threshold:
                continue
            label = _label_from_names(names, int(class_id))
            xi1, yi1, xi2, yi2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            xi1, yi1 = max(0, xi1), max(0, yi1)
            xi2, yi2 = min(w, xi2), min(h, yi2)
            if xi2 <= xi1 or yi2 <= yi1:
                continue
            material, mat_conf = guess_material(label)
            low_conf = float(conf) < 0.7 or mat_conf < 0.7
            objects.append({
                "class": label,
                "bounding_box": [xi1, yi1, xi2, yi2],
                "confidence": round(float(conf), 3),
                "material_guess": material,
                "material_confidence": round(mat_conf, 2),
                "low_confidence": low_conf,
            })
    return objects
