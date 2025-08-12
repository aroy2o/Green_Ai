"""Central configuration constants and helpers.

This module centralizes environment-derived configuration values so that other
modules avoid duplicating constant definitions (e.g., ANALYSIS_DATE) and can
provide consistent fallback logic. Keep this lightweight (no heavy imports).
"""
from __future__ import annotations
import os
from datetime import datetime
from functools import lru_cache

# Fixed analysis date requirement. Allow override via env but default to the
# mandated date to keep responses aligned with spec/tests.
ANALYSIS_DATE: str = os.getenv("ANALYSIS_FIXED_DATE", "August 11, 2025")

# OpenAI + sustainability model parameters
SUSTAINABILITY_MODEL: str = os.getenv("SUSTAINABILITY_MODEL", "gpt-3.5-turbo")
SUSTAINABILITY_TEMPERATURE: float = float(os.getenv("SUSTAINABILITY_TEMPERATURE", "0.25"))
SUSTAINABILITY_MAX_TOKENS: int = int(os.getenv("SUSTAINABILITY_MAX_TOKENS", "1536"))

# Enrichment time / limits
ENRICH_MAX_PAGES: int = int(os.getenv("ENRICH_MAX_PAGES", "6"))
ENRICH_SEARCH_TIMEOUT: float = float(os.getenv("ENRICH_SEARCH_TIMEOUT", "4.0"))
ENRICH_FETCH_TIMEOUT: float = float(os.getenv("ENRICH_FETCH_TIMEOUT", "4.0"))
ENRICH_TOTAL_TIMEOUT: float = float(os.getenv("ENRICH_TOTAL_TIMEOUT", "8.0"))
ENRICH_ENABLE: bool = os.getenv("ENABLE_ENRICH", "1") not in ("0", "false", "False")
ENRICH_CACHE_TTL: float = float(os.getenv("ENRICH_CACHE_TTL", "3600"))  # 1h default
ENRICH_CACHE_MAX: int = int(os.getenv("ENRICH_CACHE_MAX", "64"))

# Operational (inference) limits
INFER_MAX_CONCURRENCY: int = int(os.getenv("MAX_INFER_CONCURRENCY", "4"))
DETECTION_TIMEOUT_SEC: float = float(os.getenv("DETECTION_TIMEOUT_SEC", "6"))
OCR_TIMEOUT_SEC: float = float(os.getenv("OCR_TIMEOUT_SEC", "10"))
ROI_OCR_TIMEOUT_SEC: float = float(os.getenv("ROI_OCR_TIMEOUT_SEC", "4"))
MAX_IMAGE_SIDE: int = int(os.getenv("MAX_IMAGE_SIDE", "1920"))


@lru_cache(maxsize=1)
def get_runtime_config_snapshot() -> dict:
    """Return a cached snapshot of key runtime config values (for diagnostics)."""
    return {
        "analysis_date": ANALYSIS_DATE,
        "model": SUSTAINABILITY_MODEL,
        "temperature": SUSTAINABILITY_TEMPERATURE,
        "max_tokens": SUSTAINABILITY_MAX_TOKENS,
        "enrich_enabled": ENRICH_ENABLE,
        "enrich_total_timeout": ENRICH_TOTAL_TIMEOUT,
        "enrich_search_timeout": ENRICH_SEARCH_TIMEOUT,
        "enrich_fetch_timeout": ENRICH_FETCH_TIMEOUT,
        "enrich_max_pages": ENRICH_MAX_PAGES,
        "enrich_cache_ttl": ENRICH_CACHE_TTL,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

__all__ = [
    "ANALYSIS_DATE",
    "SUSTAINABILITY_MODEL",
    "SUSTAINABILITY_TEMPERATURE",
    "SUSTAINABILITY_MAX_TOKENS",
    "ENRICH_MAX_PAGES",
    "ENRICH_SEARCH_TIMEOUT",
    "ENRICH_FETCH_TIMEOUT",
    "ENRICH_TOTAL_TIMEOUT",
    "ENRICH_ENABLE",
    "ENRICH_CACHE_TTL",
    "ENRICH_CACHE_MAX",
    "INFER_MAX_CONCURRENCY",
    "DETECTION_TIMEOUT_SEC",
    "OCR_TIMEOUT_SEC",
    "ROI_OCR_TIMEOUT_SEC",
    "MAX_IMAGE_SIDE",
    "get_runtime_config_snapshot",
]
