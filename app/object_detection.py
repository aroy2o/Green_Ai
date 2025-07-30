import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

# COCO class labels (id: name)
COCO_LABELS = {
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    81: "can", 82: "box", 83: "carton", 84: "wrapper", 85: "sachet", 80: "barcode",
    # ...existing code for other classes...
}

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
    "wine glass": ("glass", 0.95)
}

_yolo_model = None

def get_yolo_model():
    """Singleton loader for YOLOv8 model."""
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model

def guess_material(label):
    """Return material guess and confidence for a given label."""
    return MATERIAL_MAP.get(label, ("unknown", 0.4))

def detect_objects_and_materials(image_bytes: bytes, use_gpu=True, conf_threshold=0.6):
    """
    Detect objects and guess material from image bytes.
    Returns list of dicts with class, bounding_box, confidence, material_guess, material_confidence, low_confidence.
    """
    img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    model = get_yolo_model()
    results = model(img, device=0 if use_gpu else 'cpu')
    objects = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            label = COCO_LABELS.get(class_id, str(class_id))
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            material, mat_conf = guess_material(label)
            low_conf = conf < 0.7 or mat_conf < 0.7
            objects.append({
                "class": label,
                "bounding_box": [int(x) for x in xyxy],
                "confidence": round(conf, 3),
                "material_guess": material,
                "material_confidence": round(mat_conf, 2),
                "low_confidence": low_conf
            })
    return objects
