import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import redis
from pymongo import MongoClient
from pymongo.collection import Collection

from config import REDIS_URL, MONGODB_URI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Cached clients / collection
_redis_client: Optional[redis.Redis] = None
_mongo_client: Optional[MongoClient] = None
_mongo_collection: Optional[Collection] = None

# Local in-process fallback storage
# Structure: { user_id: [ { "user_message": ..., "ai_response": ..., "timestamp": iso } ] }
_local_conversations: Dict[str, List[Dict[str, str]]] = {}


# --------------------
# Redis client (best-effort)
# --------------------
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
            socket_connect_timeout=5,
        )
        _redis_client.ping()
        logger.info("✅ Redis connected")
        return _redis_client
    except Exception as exc:
        logger.warning("❌ Redis connection failed: %s", exc)
        _redis_client = None
        return None


# --------------------
# Mongo collection (source of truth)
# --------------------
def get_mongo_collection() -> Optional[Collection]:
    """
    Return the MongoDB collection used for conversations, or None if unavailable.
    This function caches the collection object in module scope.
    """
    global _mongo_client, _mongo_collection

    if _mongo_collection is not None:
        return _mongo_collection

    if not MONGODB_URI:
        return None

    try:
        _mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        _mongo_client.admin.command("ping")
        _mongo_collection = _mongo_client["groq_chat_db"]["conversations"]
        logger.info("✅ MongoDB connected")
        return _mongo_collection
    except Exception as exc:
        logger.warning("❌ MongoDB connection failed: %s", exc)
        _mongo_collection = None
        return None


# --------------------
# Local helpers
# --------------------
def _local_append(user_id: str, user_msg: str, ai_msg: str, timestamp: datetime) -> None:
    _local_conversations.setdefault(user_id, []).append(
        {
            "user_message": user_msg,
            "ai_response": ai_msg,
            "timestamp": timestamp.isoformat(),
        }
    )


def _local_get_recent(user_id: str, limit: int) -> List[Dict[str, str]]:
    items = _local_conversations.get(user_id, [])
    if limit <= 0:
        return []
    return items[-limit:]


# --------------------
# Save conversation
# --------------------
def save_conversation(user_id: str, user_msg: str, ai_msg: str) -> bool:
    """
    Save a conversation entry to Redis and MongoDB (best-effort).
    Returns True if saved to any store (or local fallback).
    """
    timestamp = datetime.now(timezone.utc)
    payload = {
        "user_message": user_msg,
        "ai_response": ai_msg,
        "timestamp": timestamp.isoformat(),
    }

    saved_any = False

    # Redis (best-effort)
    r = get_redis_client()
    if r is not None:
        try:
            # store newest at index 0 (LPUSH)
            r.lpush(f"conversation:{user_id}", json.dumps(payload))
            saved_any = True
        except Exception as exc:
            logger.warning("Redis save failed: %s", exc)

    # Mongo (source-of-truth)
    col = get_mongo_collection()
    if col is not None:
        try:
            col.insert_one(
                {
                    "user_id": user_id,
                    "user_message": user_msg,
                    "ai_response": ai_msg,
                    "timestamp": timestamp,
                }
            )
            saved_any = True
        except Exception as exc:
            logger.warning("Mongo save failed: %s", exc)

    # Local fallback
    if not saved_any:
        _local_append(user_id, user_msg, ai_msg, timestamp)
        saved_any = True

    return saved_any


# --------------------
# Context history (Redis -> Mongo -> Local)
# --------------------
def get_context_history(user_id: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    Return a list of messages (role/content) ordered oldest -> newest.
    `limit` refers to number of conversation *pairs* (user+assistant will become two messages each).
    """
    if limit <= 0:
        return []

    # Redis
    r = get_redis_client()
    if r is not None:
        try:
            # lrange with 0..limit-1 returns newest->older; reverse for chronological
            raw = r.lrange(f"conversation:{user_id}", 0, max(0, limit - 1))
            history: List[Dict[str, str]] = []
            for item in reversed(raw):
                d = json.loads(item)
                history.append({"role": "user", "content": d.get("user_message", "")})
                history.append({"role": "assistant", "content": d.get("ai_response", "")})
            return history
        except Exception as exc:
            logger.warning("Redis context read failed: %s", exc)

    # Mongo
    mongo_hist = _get_context_from_mongo(user_id, limit)
    if mongo_hist:
        return mongo_hist

    # Local fallback
    history: List[Dict[str, str]] = []
    for item in _local_get_recent(user_id, limit):
        history.append({"role": "user", "content": item.get("user_message", "")})
        history.append({"role": "assistant", "content": item.get("ai_response", "")})
    return history


def _get_context_from_mongo(user_id: str, limit: int) -> List[Dict[str, str]]:
    col = get_mongo_collection()
    if col is None:
        return []

    try:
        cursor = col.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)
        rows = list(cursor)  # newest->oldest
        history: List[Dict[str, str]] = []
        for r in reversed(rows):  # oldest->newest
            history.append({"role": "user", "content": r.get("user_message", "")})
            history.append({"role": "assistant", "content": r.get("ai_response", "")})
        return history
    except Exception as exc:
        logger.warning("Mongo context read failed: %s", exc)
        return []


# --------------------
# Load recent conversations (sidebar) (Redis -> Mongo -> Local)
# --------------------
def load_recent_conversations(user_id: str, limit: int = 10) -> List[Tuple[str, str, str]]:
    """
    Return list of tuples (user_message, ai_response, timestamp_str).
    Ordered newest -> oldest (sidebar-friendly).
    """
    if limit <= 0:
        return []

    # Redis first
    r = get_redis_client()
    if r is not None:
        try:
            items = r.lrange(f"conversation:{user_id}", 0, max(0, limit - 1))  # newest->oldest
            results: List[Tuple[str, str, str]] = []
            for item in items:
                d = json.loads(item)
                try:
                    ts = datetime.fromisoformat(d.get("timestamp"))
                except Exception:
                    ts = datetime.now(timezone.utc)
                results.append(
                    (
                        d.get("user_message", ""),
                        d.get("ai_response", ""),
                        ts.strftime("%b %d, %H:%M"),
                    )
                )
            return results
        except Exception as exc:
            logger.warning("Redis sidebar read failed: %s", exc)

    # Mongo fallback
    col = get_mongo_collection()
    if col is not None:
        try:
            rows = list(col.find({"user_id": user_id}).sort("timestamp", -1).limit(limit))
            results: List[Tuple[str, str, str]] = []
            for row in rows:
                ts = row.get("timestamp") or datetime.now(timezone.utc)
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts)
                    except Exception:
                        ts = datetime.now(timezone.utc)
                results.append(
                    (row.get("user_message", ""), row.get("ai_response", ""), ts.strftime("%b %d, %H:%M"))
                )
            return results
        except Exception as exc:
            logger.warning("Mongo sidebar read failed: %s", exc)

    # Local fallback (return newest->oldest)
    local_items = _local_get_recent(user_id, limit)
    results: List[Tuple[str, str, str]] = []
    for d in reversed(local_items):  # local stored oldest->newest; reversed => newest->oldest
        try:
            ts = datetime.fromisoformat(d.get("timestamp"))
        except Exception:
            ts = datetime.now(timezone.utc)
        results.append((d.get("user_message", ""), d.get("ai_response", ""), ts.strftime("%b %d, %H:%M")))
    return results


# --------------------
# Delete / Clear
# --------------------
def delete_last_conversation(user_id: str) -> bool:
    """
    Delete the most recent conversation entry. Returns True if any deletion occurred.
    """
    deleted_any = False

    # Redis: lpop removes newest
    r = get_redis_client()
    if r is not None:
        try:
            res = r.lpop(f"conversation:{user_id}")
            if res is not None:
                deleted_any = True
        except Exception as exc:
            logger.warning("Redis delete failed: %s", exc)

    # Mongo: remove latest document by timestamp
    col = get_mongo_collection()
    if col is not None:
        try:
            latest = col.find_one({"user_id": user_id}, sort=[("timestamp", -1)])
            if latest:
                col.delete_one({"_id": latest["_id"]})
                deleted_any = True
        except Exception as exc:
            logger.warning("Mongo delete failed: %s", exc)

    # Local fallback
    if not deleted_any and _local_conversations.get(user_id):
        try:
            _local_conversations[user_id].pop()
            deleted_any = True
        except Exception:
            pass

    return deleted_any


def clear_all_history(user_id: str) -> None:
    # Redis
    r = get_redis_client()
    if r is not None:
        try:
            r.delete(f"conversation:{user_id}")
        except Exception as exc:
            logger.warning("Redis clear failed: %s", exc)

    # Mongo
    col = get_mongo_collection()
    if col is not None:
        try:
            col.delete_many({"user_id": user_id})
        except Exception as exc:
            logger.warning("Mongo clear failed: %s", exc)

    # Local
    _local_conversations.pop(user_id, None)
