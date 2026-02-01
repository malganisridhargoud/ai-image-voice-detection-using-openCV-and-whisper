# config.py
import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

# ----------------------
# API KEYS / TOKENS
# ----------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY missing in .env")

# ----------------------
# Models & Runtime
# ----------------------
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "llama-3.3-70b-versatile")
# AUDIO_MODEL required by app.py
AUDIO_MODEL = os.getenv("AUDIO_MODEL", "whisper-large-v3-turbo")

# ----------------------
# Memory / Persistence
# ----------------------
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "5"))

# Redis / Mongo endpoints
REDIS_URL = os.getenv("REDIS_URL", "").strip()
MONGODB_URI = os.getenv("MONGODB_URI", "").strip()

# ----------------------
# Debug
# ----------------------
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
