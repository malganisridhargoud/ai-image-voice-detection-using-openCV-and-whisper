# auth.py
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

import bcrypt
from pymongo import MongoClient
from pymongo.collection import Collection

from config import MONGODB_URI

logger = logging.getLogger(__name__)
_local_users: Dict[str, Dict[str, object]] = {}


def get_users_collection() -> Optional[Collection]:
    """Return the users collection or None if Mongo is unavailable."""
    if not MONGODB_URI:
        return None

    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client["groq_chat_db"]["users"]
    except Exception as exc:
        logger.error("Users collection unavailable: %s", exc)
        return None


def is_auth_available() -> bool:
    """Auth is available via Mongo when reachable, else local in-memory fallback."""
    return True


def hash_password(password: str) -> bytes:
    """Hash a plaintext password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())


def verify_password(password: str, password_hash) -> bool:
    """
    Verify a plaintext password against a stored hash.
    Accepts bytes, bson.Binary, or str for password_hash.
    """
    if isinstance(password_hash, str):
        password_hash = password_hash.encode("utf-8")
    # bson.binary.Binary behaves like bytes, so bcrypt will accept it
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash)
    except Exception as exc:
        logger.warning("Password verification error: %s", exc)
        return False


def create_user(username: str, password: str) -> bool:
    """Create a new user. Uses Mongo when available, else local in-memory fallback."""
    users = get_users_collection()
    if users is None:
        if username in _local_users:
            return False
        _local_users[username] = {
            "username": username,
            "password_hash": hash_password(password),
            "created_at": datetime.now(timezone.utc),
        }
        logger.warning("Mongo unavailable; created local temporary user: %s", username)
        return True

    # Check existence
    if users.find_one({"username": username}):
        return False

    try:
        users.insert_one(
            {
                "username": username,
                "password_hash": hash_password(password),
                "created_at": datetime.now(timezone.utc),
            }
        )
        return True
    except Exception as exc:
        logger.error("Failed to create user: %s", exc)
        return False


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """
    Authenticate and return the user document (with password_hash removed).
    Returns None on failure or if auth backend is unavailable.
    """
    users = get_users_collection()
    if users is None:
        user = _local_users.get(username)
        if not user:
            return None
        if verify_password(password, user.get("password_hash")):
            return {"username": username}
        return None

    user = users.find_one({"username": username})
    if not user:
        return None

    if verify_password(password, user.get("password_hash")):
        # Return a sanitized user object (avoid returning password hash)
        user.pop("password_hash", None)
        return user
    return None
