# app.py
import streamlit as st
import logging
import time
from typing import List, Dict

# --- LangChain / Groq imports ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from groq import Groq

# --- App Config & Backend ---
from config import GROQ_API_KEY, PRIMARY_MODEL, CONTEXT_WINDOW, AUDIO_MODEL
from memory_manager import (
    load_recent_conversations,
    save_conversation,
    get_context_history,
    get_mongo_collection,
    delete_last_conversation,
    clear_all_history
)
from auth import authenticate_user, create_user

# ======================
# Setup & Logging
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Custom AI With Memory",
    page_icon="🎙️",
    layout="centered"
)

# ======================
# UI Styling (Original Clean Style)
# ======================
st.markdown("""
<style>
.block-container { padding-top: 3rem; max-width: 800px; }
header, footer { visibility: hidden; }
.stTextInput > div > div > input { border-radius: 8px; }
.stAudioInput { margin-top: 10px; }
button[kind="secondary"] { border-color: #ff4b4b; color: #ff4b4b; }
</style>
""", unsafe_allow_html=True)

# ======================
# Session State
# ======================
if "user" not in st.session_state:
    st.session_state.user = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "audio_uploader_key" not in st.session_state:
    st.session_state.audio_uploader_key = 0

def reset_audio_widget():
    st.session_state.audio_uploader_key += 1

# ======================
# 🔐 AUTHENTICATION (FIRST)
# ======================
if st.session_state.user is None:
    st.title("🔐 Login")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = authenticate_user(username, password)
            if user:
                st.session_state.user = user
                st.success("Logged in successfully")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            if create_user(new_user, new_pass):
                st.success("Account created. Please login.")
            else:
                st.error("Username already exists")

    st.stop()  # ⛔ Stop app until authenticated

# ======================
# ✅ SAFE AFTER LOGIN
# ======================
USER_ID = st.session_state.user["username"]

# ======================
# Load Long-Term Memory into Session (CRITICAL FIX)
# ======================
if not st.session_state.messages:
    past = load_recent_conversations(USER_ID, limit=20)
    for user_msg, ai_msg, _ in reversed(past):
        st.session_state.messages.append(
            {"role": "user", "content": user_msg}
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": ai_msg}
        )

# ======================
# Initialize Clients
# ======================
llm = ChatGroq(
    temperature=0.7,
    model_name=PRIMARY_MODEL,
    groq_api_key=GROQ_API_KEY
)

client = Groq(api_key=GROQ_API_KEY)

# ======================
# Sidebar
# ======================
with st.sidebar:
    st.caption(f"👤 {USER_ID}")

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.user = None
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption(f"🧠 Model: {PRIMARY_MODEL}")
    st.caption(
        "🟢 Memory: Active"
        if get_mongo_collection() is not None
        else "🔴 Memory: Offline"
    )

    if st.button("↩️ Undo Last Message", use_container_width=True):
        if len(st.session_state.messages) >= 2:
            st.session_state.messages.pop()
            st.session_state.messages.pop()
        delete_last_conversation(USER_ID)
        reset_audio_widget()
        st.rerun()

    if st.button("🗑️ Clear All History", use_container_width=True):
        st.session_state.messages = []
        clear_all_history(USER_ID)
        reset_audio_widget()
        st.rerun()

    st.divider()
    st.subheader("🧠 Memory Bank")

    past_memories = load_recent_conversations(USER_ID, limit=10)
    if past_memories:
        for u, a, t in past_memories:
            with st.expander(f"🕒 {t}"):
                st.markdown(f"**You:** {u}")
                st.markdown(f"**Assistant:** {a[:300]}...")
    else:
        st.caption("No stored memory yet.")

# ======================
# Main Chat UI
# ======================
st.title("🎙️ AI Voice With Long MEMORY")
st.caption("Powered by LangChain + Groq + Whisper")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

voice_input = st.audio_input(
    "Voice Input",
    key=f"audio_{st.session_state.audio_uploader_key}"
)
text_input = st.chat_input("Type a message...")

prompt_text = None

if voice_input:
    audio_bytes = voice_input.read()
    if audio_bytes:
        with st.spinner("Transcribing..."):
            prompt_text = client.audio.transcriptions.create(
                file=("input.wav", audio_bytes, "audio/wav"),
                model=AUDIO_MODEL,
                response_format="text"
            )
elif text_input:
    prompt_text = text_input

# ======================
# Response Generation
# ======================
if prompt_text:
    with st.chat_message("user"):
        st.markdown(prompt_text)

    st.session_state.messages.append(
        {"role": "user", "content": prompt_text}
    )

    # Build context from Redis/Mongo
    history_context = ""
    for entry in get_context_history(USER_ID, limit=CONTEXT_WINDOW):
        history_context += f"{entry['role'].capitalize()}: {entry['content']}\n"

    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Answer concisely."),
            ("system", "Conversation history:\n{context}"),
            ("user", "{input}")
        ])
        | llm
        | StrOutputParser()
    )

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            for chunk in chain.stream({
                "context": history_context,
                "input": prompt_text
            }):
                full_response += str(chunk)
                placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)
            save_conversation(USER_ID, prompt_text, full_response)

        except Exception as e:
            logger.error(e)
            full_response = "I encountered an error generating the response."
            placeholder.error(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

    reset_audio_widget()
    st.rerun()
