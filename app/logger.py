from loguru import logger
import sys, os, json, contextlib

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "app.log")
ENABLE_JSON_LOGS = os.getenv("JSON_LOGS", "0") in ("1","true","True")

logger.remove()
# STDERR sink for container logs (optionally JSON)
if ENABLE_JSON_LOGS:
    def _json_sink(message):  # pragma: no cover (formatting logic)
        r = message.record
        payload = {
            "time": r["time"].isoformat(),
            "level": r["level"].name,
            "msg": r["message"],
            "module": r["module"],
            "function": r["function"],
            "line": r["line"],
        }
        if "extra" in r and r["extra"]:
            payload.update({k:v for k,v in r["extra"].items() if not k.startswith('_')})
        print(json.dumps(payload, ensure_ascii=False))
    logger.add(_json_sink, level=LOG_LEVEL, enqueue=True, backtrace=False, diagnose=False)
else:
    logger.add(sys.stderr, level=LOG_LEVEL, enqueue=True, backtrace=False, diagnose=False)

# Rotating file sink (always text to ease local debugging)
logger.add(
    LOG_PATH,
    level=LOG_LEVEL,
    rotation=os.getenv("LOG_ROTATION", "10 MB"),
    retention=os.getenv("LOG_RETENTION", "7 days"),
    compression=os.getenv("LOG_COMPRESSION", "zip"),
    enqueue=True,
    backtrace=False,
    diagnose=False,
)

@contextlib.contextmanager
def log_context(**kv):
    """Context manager to temporarily bind structured context to logs.
    Usage:
        with log_context(request_id="abc123"):
            logger.info("processing")
    """
    token = logger.bind(**kv)
    try:
        yield token
    finally:  # pragma: no cover
        # loguru binding returns a new logger; nothing to unbind explicitly
        pass

__all__ = ["logger", "log_context"]
