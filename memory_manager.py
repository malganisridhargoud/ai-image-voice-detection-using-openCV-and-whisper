import logging
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any
from pymongo import MongoClient, errors as mongo_errors
from pymongo.collection import Collection
from config import MONGODB_URI

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Singleton globals for DB connection
_mongo_client: Optional[MongoClient] = None
_collection: Optional[Collection] = None

def get_mongo_collection() -> Optional[Collection]:
    """
    Initialize MongoDB connection once and reuse it (Singleton Pattern).
    Returns None if MONGODB_URI is missing or connection fails.
    """
    global _mongo_client, _collection

    if _collection is not None:
        return _collection

    if not MONGODB_URI:
        # Log only once to avoid spamming console
        if not getattr(get_mongo_collection, "has_warned", False):
            logger.warning("⚠️ MongoDB URI not configured. Persistent memory is OFF.")
            get_mongo_collection.has_warned = True
        return None

    try:
        # Connect to Atlas or Local Mongo
        if "mongodb+srv" in MONGODB_URI:
            _mongo_client = MongoClient(
                MONGODB_URI,
                tls=True,
                serverSelectionTimeoutMS=5000,
                retryWrites=True
            )
        else:
            _mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)

        # Force a connection check
        _mongo_client.admin.command("ping")
        
        db = _mongo_client["groq_chat_db"]
        _collection = db["conversations"]
        
        # Optimize lookups by User ID and Time
        _collection.create_index([("user_id", 1), ("timestamp", -1)])
        
        logger.info("✅ MongoDB connected successfully")
        return _collection

    except mongo_errors.ServerSelectionTimeoutError:
        logger.error("❌ MongoDB Connection Timeout. Check your internet or whitelist settings.")
        return None
    except mongo_errors.OperationFailure as e:
        logger.error(f"❌ MongoDB Authentication Failed: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected MongoDB Error: {e}")
        return None

def save_conversation(user_id: str, user_msg: str, ai_msg: str) -> bool:
    """Save a single conversation turn to MongoDB."""
    collection = get_mongo_collection()
    if collection is None:
        return False

    try:
        document = {
            "user_id": user_id,
            "user_message": user_msg,
            "ai_response": ai_msg,
            # Use timezone-aware UTC for modern Python compatibility
            "timestamp": datetime.now(timezone.utc)
        }
        collection.insert_one(document)
        # logger.info("💾 Saved to DB") # Uncomment if you want verbose logs
        return True
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")
        return False

def load_recent_conversations(user_id: str, limit: int = 10) -> List[Tuple[str, str, str]]:
    """
    Load recent conversations for the 'History' sidebar UI.
    Returns: List of (User Message, AI Message, Formatted Time string)
    """
    collection = get_mongo_collection()
    if collection is None:
        return []

    try:
        records = collection.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)
        results = []
        for r in records:
            ts = r.get("timestamp")
            # Format time nicely (e.g., "Oct 25, 14:30")
            fmt_time = ts.strftime("%b %d, %H:%M") if ts else "Unknown"
            results.append((
                r.get("user_message", ""),
                r.get("ai_response", ""),
                fmt_time
            ))
        return results
    except Exception as e:
        logger.error(f"Failed to load history: {e}")
        return []

def get_context_history(user_id: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    Get raw conversation history to inject into the LLM context window.
    Returns: List of dicts [{"role": "user", "content": ...}, ...]
    """
    collection = get_mongo_collection()
    if collection is None:
        return []

    try:
        # Fetch recent records (reverse logic: we grab the newest N, then flip them to chronological order)
        records = collection.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)
        
        history_sequence = []
        # 'reversed' ensures we feed the LLM the oldest-of-the-recent first
        for r in reversed(list(records)):
            history_sequence.append({"role": "user", "content": r.get("user_message", "")})
            history_sequence.append({"role": "assistant", "content": r.get("ai_response", "")})
            
        return history_sequence
    except Exception as e:
        logger.error(f"Failed to retrieve context context: {e}")
        return []
    

def delete_last_conversation(user_id: str) -> bool:
    """Delete the most recent conversation entry for a user."""
    collection = get_mongo_collection()
    if collection is None:
        return False

    try:
        # Find the latest document
        latest = collection.find_one(
            {"user_id": user_id},
            sort=[("timestamp", -1)]
        )
        
        if latest:
            collection.delete_one({"_id": latest["_id"]})
            logger.info("🗑️ Deleted last conversation turn")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to delete last conversation: {e}")
        return False