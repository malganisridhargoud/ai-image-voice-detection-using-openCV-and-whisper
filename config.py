import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MONGODB_URI = os.getenv("MONGODB_ATLAS_URI", "").strip()

if not GROQ_API_KEY:
    raise EnvironmentError("❌ Missing GROQ_API_KEY in .env file!")

# --- Model Configurations ---

# Text Model (Brain)
# "llama-3.3-70b-versatile" is the current flagship for complex tasks
PRIMARY_MODEL = os.getenv("GROQ_PRIMARY_MODEL", "llama-3.3-70b-versatile")

# Audio Model (Ears) - UPDATED
# 'distil-whisper-large-v3-en' is DEPRECATED.
# We now use 'whisper-large-v3-turbo' (Fastest) or 'whisper-large-v3' (Most Accurate)
AUDIO_MODEL = os.getenv("GROQ_AUDIO_MODEL", "whisper-large-v3-turbo")

# --- App Settings ---
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "5"))
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() in ("1", "true", "yes")