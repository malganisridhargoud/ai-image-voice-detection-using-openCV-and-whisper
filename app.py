# app.py
import logging
from typing import Optional

import streamlit as st
from groq import Groq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from ai_features import detect_objects_with_opencv, detect_sentiment
from auth import authenticate_user, create_user, is_auth_available
from config import AUDIO_MODEL, CONTEXT_WINDOW, GROQ_API_KEY, PRIMARY_MODEL
from memory_manager import (
    clear_all_history,
    delete_last_conversation,
    get_context_history,
    get_mongo_collection,
    load_recent_conversations,
    save_conversation,
)

# ======================
# Setup & Logging
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Assistant",
    page_icon="AI",
    layout="centered",
)

# ======================
# UI Styling
# ======================
st.markdown(
    """
<style>
:root {
    --bg: #000000;
    --card: #000000;
    --text: #ffffff;
    --muted: #ffffff;
    --border: #ffffff;
}

.stApp {
    background: var(--bg);
}

.block-container {
    padding-top: 2rem;
    max-width: 780px;
}

[data-testid="stSidebar"] {
    background: #000000;
}

header, footer { visibility: hidden; }

h1, h2, h3, p, label, span, div {
    color: var(--text);
}

.stTextInput > div > div > input,
.stChatInput > div,
.stTextArea textarea,
[data-testid="stFileUploaderDropzone"] {
    border-radius: 10px;
    border: 1px solid var(--border);
    background: var(--card);
    color: var(--text);
}

.stAudioInput { margin-top: 8px; }

.stButton > button {
    border: 1px solid #ffffff;
    background: #000000;
    color: #ffffff;
}

button[kind="secondary"] {
    border-color: #ffffff;
    color: #ffffff;
    background: #000000;
}

.stCaption { color: var(--muted); }
</style>
""",
    unsafe_allow_html=True,
)

# ======================
# Session State
# ======================
if "user" not in st.session_state:
    st.session_state.user = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "audio_uploader_key" not in st.session_state:
    st.session_state.audio_uploader_key = 0

if "sentiment_log" not in st.session_state:
    st.session_state.sentiment_log = []


def reset_audio_widget() -> None:
    st.session_state.audio_uploader_key += 1


# ======================
# Authentication
# ======================
if st.session_state.user is None:
    st.title("Sign in")
    st.caption("Use your account, or continue in preview mode.")

    auth_available = is_auth_available()
    if not auth_available:
        st.warning("Database auth is unavailable. Use guest preview or configure MONGODB_URI.")

    if st.button("Continue as Guest (Preview)", use_container_width=True):
        st.session_state.user = {"username": "guest_preview"}
        st.session_state.messages = []
        st.success("Preview mode enabled")
        st.rerun()

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", key="login_btn", disabled=not auth_available):
            user = authenticate_user(username, password)
            if user:
                st.session_state.user = user
                st.success("Logged in successfully")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        new_user = st.text_input("New Username", key="reg_username")
        new_pass = st.text_input("New Password", type="password", key="reg_password")

        if st.button("Create Account", key="create_account_btn", disabled=not auth_available):
            if create_user(new_user, new_pass):
                st.success("Account created. Please login.")
            else:
                st.error("Username already exists")

    st.stop()


# ======================
# Safe After Login
# ======================
USER_ID = st.session_state.user["username"]

# ======================
# Load Long-Term Memory into Session
# ======================
if not st.session_state.messages:
    past = load_recent_conversations(USER_ID, limit=20) or []
    for user_msg, ai_msg, _ in reversed(past):
        st.session_state.messages.append({"role": "user", "content": user_msg})
        st.session_state.messages.append({"role": "assistant", "content": ai_msg})

# ======================
# Initialize Clients
# ======================
llm: Optional[ChatGroq] = None
client: Optional[Groq] = None
if GROQ_API_KEY:
    try:
        llm = ChatGroq(temperature=0.7, model_name=PRIMARY_MODEL, groq_api_key=GROQ_API_KEY)
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as exc:
        logger.error("Failed to initialize Groq clients: %s", exc)
        llm = None
        client = None

# ======================
# Sidebar
# ======================
with st.sidebar:
    st.caption(f"User: {USER_ID}")

    if st.button("Logout", use_container_width=True):
        st.session_state.user = None
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption(f"Model: {PRIMARY_MODEL}")
    st.caption("Memory: Active" if get_mongo_collection() is not None else "Memory: Offline")

    if st.session_state.sentiment_log:
        latest = st.session_state.sentiment_log[-1]
        st.caption(f"Latest sentiment: {latest['label']} ({latest['score']})")

    if st.button("Undo Last Message", use_container_width=True):
        if len(st.session_state.messages) >= 2:
            st.session_state.messages.pop()
            st.session_state.messages.pop()
        delete_last_conversation(USER_ID)
        reset_audio_widget()
        st.rerun()

    if st.button("Clear All History", use_container_width=True):
        st.session_state.messages = []
        clear_all_history(USER_ID)
        reset_audio_widget()
        st.rerun()

    st.divider()
    st.subheader("Recent Memory")

    past_memories = load_recent_conversations(USER_ID, limit=10) or []
    if past_memories:
        for u, a, t in past_memories:
            with st.expander(f"{t}"):
                st.markdown(f"**You:** {u}")
                st.markdown(f"**Assistant:** {a[:300]}{'...' if len(a) > 300 else ''}")
    else:
        st.caption("No stored memory yet.")

# ======================
# Main Chat UI
# ======================
st.title("AI Assistant")
st.caption("Chat with memory, voice input, and image detection.")

if not GROQ_API_KEY:
    st.warning(
        "Preview mode: GROQ_API_KEY is not configured. Chat and voice are disabled, but UI and OpenCV preview are available."
    )

st.subheader("Image Detection")
st.caption("Use your camera or upload an image.")

camera_image = st.camera_input("Capture from camera", key="opencv_camera")
uploaded_image = st.file_uploader(
    "Or upload an image for object detection",
    type=["png", "jpg", "jpeg"],
    key="opencv_uploader",
)

image_source = camera_image if camera_image is not None else uploaded_image
if image_source is not None:
    try:
        with st.spinner("Detecting objects..."):
            annotated_image, objects = detect_objects_with_opencv(image_source.getvalue())
        st.image(annotated_image, caption="Detected objects", use_container_width=True)
        if objects:
            st.success("Detected: " + ", ".join(objects[:10]))
        else:
            st.info("No supported objects detected. Try a clearer face/body image with good lighting.")
    except Exception as exc:
        logger.exception("OpenCV object detection failed")
        st.warning(f"Object detection unavailable: {exc}")

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Voice input (browser microphone widget) - disabled when GROQ API not configured
voice_input = st.audio_input(
    "Voice input (microphone)",
    key=f"audio_{st.session_state.audio_uploader_key}",
    disabled=not bool(GROQ_API_KEY),
)

# Text input
text_input = st.chat_input("Type a message...", disabled=not bool(GROQ_API_KEY))

prompt_text: Optional[str] = None

# Handle voice transcription (only if we have a Groq client)
if voice_input and client is not None:
    try:
        audio_bytes = voice_input.read()
        if audio_bytes:
            with st.spinner("Transcribing..."):
                resp = client.audio.transcriptions.create(
                    file=("input.wav", audio_bytes, "audio/wav"),
                    model=AUDIO_MODEL,
                    response_format="text",
                )
                prompt_text = resp if isinstance(resp, str) else getattr(resp, "text", None) or str(resp)
    except Exception as e:
        logger.error("Transcription error: %s", e)
        st.warning("Transcription failed. Please try a different file or use text input.")

# Fallback to typed text
if not prompt_text and text_input:
    prompt_text = text_input


# ======================
# Response Generation
# ======================
if prompt_text and llm is not None:
    sentiment = detect_sentiment(prompt_text)
    st.session_state.sentiment_log.append(sentiment)

    with st.chat_message("user"):
        st.markdown(prompt_text)
        st.caption(f"Sentiment: {sentiment['label']} ({sentiment['score']})")

    st.session_state.messages.append({"role": "user", "content": prompt_text})

    history_context = ""
    for entry in get_context_history(USER_ID, limit=CONTEXT_WINDOW):
        history_context += f"{entry['role'].capitalize()}: {entry['content']}\n"

    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Answer concisely."),
            (
                "system",
                "Latest user sentiment: {sentiment_label} (confidence {sentiment_score}). Adapt tone with empathy when sentiment is negative.",
            ),
            ("system", "Conversation history:\n{context}"),
            ("user", "{input}"),
        ])
        | llm
        | StrOutputParser()
    )

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            for chunk in chain.stream(
                {
                    "context": history_context,
                    "input": prompt_text,
                    "sentiment_label": sentiment["label"],
                    "sentiment_score": sentiment["score"],
                }
            ):
                full_response += str(chunk)
                placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)
            save_conversation(USER_ID, prompt_text, full_response)

        except Exception as e:
            logger.error(e)
            full_response = "I encountered an error generating the response."
            placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    reset_audio_widget()
    st.rerun()
