from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from datetime import datetime
import os
from loguru import logger

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "ocr_pipeline")
COLLECTION = "ocr_results"

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
gfs = AsyncIOMotorGridFSBucket(db)

async def store_ocr_result(image_info, ocr_result, engine_used, processing_time):
    doc = {
        "timestamp": datetime.utcnow(),
        "image_info": image_info,
        "ocr_result": ocr_result,
        "engine_used": engine_used,
        "processing_time": processing_time,
        "status": ocr_result.get("status"),
        "error": ocr_result.get("error"),
        # For analytics/debugging, also log at top level for easy querying
        "source": (image_info or {}).get("source", "upload"),
        "retake_count": (image_info or {}).get("retake_count", 0),
    }
    try:
        await db[COLLECTION].insert_one(doc)
    except Exception as e:
        logger.error(f"Failed to store OCR result: {e}")
