# config.py
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# ----------------------
# Load environment (resolve from project directory first)
# ----------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logger = logging.getLogger(__name__)

# ----------------------
# API KEYS / TOKENS
# ----------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not set. Running in preview/offline mode.")

# ----------------------
# Models & Runtime
# ----------------------
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "llama-3.3-70b-versatile").strip()

# Used for Groq audio transcription
AUDIO_MODEL = os.getenv("AUDIO_MODEL", "whisper-large-v3-turbo").strip()

# ----------------------
# Memory / Persistence
# ----------------------
def _safe_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


CONTEXT_WINDOW = _safe_int(os.getenv("CONTEXT_WINDOW", "5"), 5)

# Redis / Mongo endpoints (optional)
REDIS_URL = os.getenv("REDIS_URL", "").strip()
MONGODB_URI = os.getenv("MONGODB_URI", "").strip()

if not MONGODB_URI:
    logger.info("MongoDB not configured. Memory persistence may be limited.")

if not REDIS_URL:
    logger.info("Redis not configured. Speed layer disabled.")

# ----------------------
# Debug
# ----------------------
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

if DEBUG_MODE:
    logging.basicConfig(level=logging.DEBUG)
    logger.debug("Debug mode enabled.")
