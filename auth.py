# auth.py
import bcrypt
from datetime import datetime, timezone
from pymongo import MongoClient
from config import MONGODB_URI

def get_users_collection():
    client = MongoClient(MONGODB_URI)
    return client["groq_chat_db"]["users"]

def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def verify_password(password: str, password_hash: bytes) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash)

def create_user(username: str, password: str) -> bool:
    users = get_users_collection()
    if users.find_one({"username": username}):
        return False

    users.insert_one({
        "username": username,
        "password_hash": hash_password(password),
        "created_at": datetime.now(timezone.utc)
    })
    return True

def authenticate_user(username: str, password: str):
    users = get_users_collection()
    user = users.find_one({"username": username})
    if not user:
        return None

    if verify_password(password, user["password_hash"]):
        return user
    return None
