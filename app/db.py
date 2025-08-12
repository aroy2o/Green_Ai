from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from datetime import datetime
from loguru import logger
import os
from typing import Optional

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "ocr_pipeline")
COLLECTION = "ocr_results"

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
gfs = AsyncIOMotorGridFSBucket(db)

async def ping_db(timeout: float = 1.5) -> bool:
    """Quick health check for Mongo connectivity."""
    try:
        await db.command("ping")
        return True
    except Exception as e:  # pragma: no cover
        logger.debug(f"Mongo ping failed: {e}")
        return False

async def close_db():
    """Close Mongo client (useful for shutdown hooks)."""
    try:
        client.close()
    except Exception:
        pass

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
        logger.debug(f"Failed to store OCR result: {e}")

async def store_sustainability_result(ocr_text: str, analysis: dict):
    """Best-effort persistence of sustainability analyses (no secrets)."""
    try:
        await db[os.getenv("SUSTAINABILITY_COLLECTION", "sustainability_analyses")].insert_one({
            "timestamp": datetime.utcnow(),
            "ocr_text": (ocr_text or "")[:20000],
            "analysis": analysis,
        })
    except Exception as e:
        logger.debug(f"Failed to store sustainability result: {e}")
