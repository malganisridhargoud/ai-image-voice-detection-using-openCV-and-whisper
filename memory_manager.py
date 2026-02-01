# memory_manager.py
import redis
import json
import logging
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict
from pymongo import MongoClient
from config import REDIS_URL, MONGODB_URI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# Global clients (cached)
# ======================
_redis_client: Optional[redis.Redis] = None
_mongo_client: Optional[MongoClient] = None
_mongo_collection = None


# ======================
# Redis (Speed Layer)
# ======================
def get_redis_client() -> Optional[redis.Redis]:
    global _redis_client

    if _redis_client is not None:
        return _redis_client

    if not REDIS_URL:
        return None

    try:
        _redis_client = redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5
        )
        _redis_client.ping()
        logger.info("✅ Redis connected")
        return _redis_client

    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        _redis_client = None
        return None


# ======================
# MongoDB (Source of Truth)
# ======================
def get_mongo_collection():
    """Return MongoDB collection or None (PyMongo-safe)."""
    global _mongo_client, _mongo_collection

    # 🔴 IMPORTANT: explicit None check (no truthiness!)
    if _mongo_collection is not None:
        return _mongo_collection

    if not MONGODB_URI:
        return None

    try:
        _mongo_client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000
        )
        _mongo_client.admin.command("ping")

        # Database + Collection
        _mongo_collection = _mongo_client["groq_chat_db"]["conversations"]
        logger.info("✅ MongoDB connected")
        return _mongo_collection

    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        _mongo_collection = None
        return None


# ======================
# Save Conversation
# ======================
def save_conversation(user_id: str, user_msg: str, ai_msg: str) -> bool:
    timestamp = datetime.now(timezone.utc)

    # --- Redis write (best-effort) ---
    r = get_redis_client()
    if r is not None:
        try:
            r.lpush(
                f"conversation:{user_id}",
                json.dumps({
                    "user_message": user_msg,
                    "ai_response": ai_msg,
                    "timestamp": timestamp.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Redis save failed: {e}")

    # --- Mongo write (source of truth) ---
    col = get_mongo_collection()
    if col is not None:
        try:
            col.insert_one({
                "user_id": user_id,
                "user_message": user_msg,
                "ai_response": ai_msg,
                "timestamp": timestamp
            })
        except Exception as e:
            logger.error(f"Mongo save failed: {e}")

    return True


# ======================
# Context History (Redis → Mongo fallback)
# ======================
def get_context_history(user_id: str, limit: int = 5) -> List[Dict[str, str]]:
    r = get_redis_client()

    if r is not None:
        try:
            items = r.lrange(f"conversation:{user_id}", 0, limit - 1)
            history: List[Dict[str, str]] = []

            for item in reversed(items):
                d = json.loads(item)
                history.append({"role": "user", "content": d["user_message"]})
                history.append({"role": "assistant", "content": d["ai_response"]})

            return history

        except Exception as e:
            logger.error(f"Redis context read failed: {e}")

    return _get_context_from_mongo(user_id, limit)


def _get_context_from_mongo(user_id: str, limit: int) -> List[Dict[str, str]]:
    col = get_mongo_collection()
    if col is None:
        return []

    try:
        records = (
            col.find({"user_id": user_id})
            .sort("timestamp", -1)
            .limit(limit)
        )

        history: List[Dict[str, str]] = []
        for r in reversed(list(records)):
            history.append({"role": "user", "content": r.get("user_message", "")})
            history.append({"role": "assistant", "content": r.get("ai_response", "")})

        return history

    except Exception as e:
        logger.error(f"Mongo context read failed: {e}")
        return []


# ======================
# Sidebar History
# ======================
def load_recent_conversations(
    user_id: str,
    limit: int = 10
) -> List[Tuple[str, str, str]]:

    r = get_redis_client()
    if r is None:
        return []

    try:
        items = r.lrange(f"conversation:{user_id}", 0, limit - 1)
        results = []

        for item in items:
            d = json.loads(item)
            ts = datetime.fromisoformat(d["timestamp"])
            results.append((
                d["user_message"],
                d["ai_response"],
                ts.strftime("%b %d, %H:%M")
            ))

        return results

    except Exception as e:
        logger.error(f"Redis sidebar read failed: {e}")
        return []


# ======================
# Delete / Clear
# ======================
def delete_last_conversation(user_id: str) -> bool:
    r = get_redis_client()
    if r is not None:
        try:
            r.lpop(f"conversation:{user_id}")
        except Exception as e:
            logger.error(f"Redis delete failed: {e}")

    col = get_mongo_collection()
    if col is not None:
        try:
            latest = col.find_one(
                {"user_id": user_id},
                sort=[("timestamp", -1)]
            )
            if latest:
                col.delete_one({"_id": latest["_id"]})
        except Exception as e:
            logger.error(f"Mongo delete failed: {e}")

    return True


def clear_all_history(user_id: str) -> None:
    r = get_redis_client()
    if r is not None:
        try:
            r.delete(f"conversation:{user_id}")
        except Exception as e:
            logger.error(f"Redis clear failed: {e}")

    col = get_mongo_collection()
    if col is not None:
        try:
            col.delete_many({"user_id": user_id})
        except Exception as e:
            logger.error(f"Mongo clear failed: {e}")
