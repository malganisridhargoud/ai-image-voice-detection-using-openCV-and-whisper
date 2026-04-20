# app.py
"""
AI Interview Coach — Dual-mode Streamlit application.
Modes:
  💬 Chat        — Original AI assistant with memory, voice, and image detection.
  🎯 Interview   — Personalized mock interviews with feedback loop.
"""
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
from feedback_engine import analyze_answer
from interview_engine import detect_topic, generate_question
from memory_manager import (
    clear_all_history,
    clear_interview_history,
    delete_last_conversation,
    get_context_history,
    get_interview_history,
    get_mongo_collection,
    get_weak_topics,
    load_recent_conversations,
    save_conversation,
    save_interview,
)

# ======================
# Setup & Logging
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🎯",
    layout="centered",
)

# ======================
# UI Styling — Premium Dark Theme
# ======================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0a0a0f;
    --card: #12121a;
    --card-border: #1e1e2e;
    --text: #e4e4e7;
    --muted: #71717a;
    --accent: #6366f1;
    --accent-glow: rgba(99, 102, 241, 0.15);
    --green: #22c55e;
    --amber: #f59e0b;
    --red: #ef4444;
}

*, *::before, *::after { font-family: 'Inter', sans-serif; }

.stApp {
    background: var(--bg);
}

.block-container {
    padding-top: 2rem;
    max-width: 800px;
}

[data-testid="stSidebar"] {
    background: #08080d;
    border-right: 1px solid var(--card-border);
}

header, footer { visibility: hidden; }

h1, h2, h3, p, label, span, div {
    color: var(--text);
}

h1 {
    background: linear-gradient(135deg, #6366f1 0%, #a78bfa 50%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    letter-spacing: -0.02em;
}

.stTextInput > div > div > input,
.stChatInput > div,
.stTextArea textarea,
[data-testid="stFileUploaderDropzone"] {
    border-radius: 12px;
    border: 1px solid var(--card-border);
    background: var(--card);
    color: var(--text);
    transition: border-color 0.2s ease;
}

.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-glow);
}

.stAudioInput { margin-top: 8px; }

.stButton > button {
    border: 1px solid var(--card-border);
    background: var(--card);
    color: var(--text);
    border-radius: 10px;
    font-weight: 500;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: var(--accent);
    background: var(--accent-glow);
    color: #fff;
    transform: translateY(-1px);
}

button[kind="secondary"] {
    border-color: var(--card-border);
    color: var(--text);
    background: var(--card);
}

.stCaption { color: var(--muted); }
.stDivider { border-color: var(--card-border); }

/* Feedback card styling */
.feedback-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    transition: border-color 0.3s ease;
}

.feedback-card:hover {
    border-color: var(--accent);
}

.score-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
}

.score-high {
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.score-mid {
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.score-low {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.question-box {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(167, 139, 250, 0.05));
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.topic-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 500;
    background: rgba(99, 102, 241, 0.12);
    color: #a78bfa;
    border: 1px solid rgba(99, 102, 241, 0.2);
    margin: 2px 4px 2px 0;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--card);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid var(--card-border);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: var(--muted);
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: var(--accent-glow);
    color: var(--accent);
}

/* Metrics */
.stMetric label { color: var(--muted) !important; font-size: 0.8rem !important; }
.stMetric [data-testid="stMetricValue"] { color: var(--text) !important; }
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

if "app_mode" not in st.session_state:
    st.session_state.app_mode = "🎯 Interview Coach"

if "interview_question" not in st.session_state:
    st.session_state.interview_question = None

if "interview_feedback" not in st.session_state:
    st.session_state.interview_feedback = None

if "interview_count" not in st.session_state:
    st.session_state.interview_count = 0


def reset_audio_widget() -> None:
    st.session_state.audio_uploader_key += 1


# ======================
# Authentication
# ======================
if st.session_state.user is None:
    st.title("AI Interview Coach")
    st.caption("Sign in to start your personalized mock interview experience.")

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
# Load Long-Term Memory into Session (for chat mode)
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
    st.caption(f"👤 {USER_ID}")

    if st.button("Logout", use_container_width=True):
        st.session_state.user = None
        st.session_state.messages = []
        st.session_state.interview_question = None
        st.session_state.interview_feedback = None
        st.rerun()

    st.divider()

    # Mode selector
    st.session_state.app_mode = st.radio(
        "Mode",
        ["🎯 Interview Coach", "💬 Chat"],
        index=0 if st.session_state.app_mode == "🎯 Interview Coach" else 1,
        key="mode_selector",
    )

    st.divider()
    st.caption(f"Model: {PRIMARY_MODEL}")
    st.caption("Memory: Active" if get_mongo_collection() is not None else "Memory: Offline")

    # -- Mode-specific sidebar content --
    if st.session_state.app_mode == "🎯 Interview Coach":
        # Weak topics display
        weak_topics = get_weak_topics(USER_ID)
        if weak_topics:
            st.subheader("📉 Weak Topics")
            for topic in weak_topics:
                st.markdown(f'<span class="topic-tag">{topic}</span>', unsafe_allow_html=True)
        else:
            st.caption("No weak topics yet — start interviewing!")

        st.divider()
        st.subheader("📋 Interview History")

        past_interviews = get_interview_history(USER_ID, limit=8)
        if past_interviews:
            for rec in past_interviews:
                q = rec.get("question", "")[:60]
                score = rec.get("feedback", {}).get("score", "—")
                topic = rec.get("topic", "")
                ts = rec.get("timestamp", "")
                with st.expander(f"📝 {q}{'...' if len(rec.get('question', '')) > 60 else ''}", expanded=False):
                    st.caption(f"🏷️ {topic}  |  ⏰ {ts}  |  Score: {score}/10")
                    st.markdown(f"**A:** {rec.get('answer', '')[:200]}{'...' if len(rec.get('answer', '')) > 200 else ''}")
        else:
            st.caption("No interview history yet.")

        if st.button("🗑️ Clear Interview History", use_container_width=True):
            clear_interview_history(USER_ID)
            st.session_state.interview_question = None
            st.session_state.interview_feedback = None
            st.session_state.interview_count = 0
            st.rerun()

    else:
        # Chat mode sidebar
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


# ================================================================
# 🎯 INTERVIEW COACH MODE
# ================================================================
if st.session_state.app_mode == "🎯 Interview Coach":
    st.title("AI Interview Coach")
    st.caption("Personalized mock interviews with AI-powered feedback. Your questions adapt to your weak areas.")

    if not GROQ_API_KEY:
        st.warning("Preview mode: GROQ_API_KEY is not configured. Interview features are disabled.")
        st.stop()

    # -- Stats row --
    col1, col2, col3 = st.columns(3)
    weak_topics = get_weak_topics(USER_ID)
    with col1:
        st.metric("Questions Answered", st.session_state.interview_count)
    with col2:
        st.metric("Weak Topics", len(weak_topics))
    with col3:
        history = get_interview_history(USER_ID, limit=100)
        if history:
            scores = [h.get("feedback", {}).get("score", 0) for h in history if isinstance(h.get("feedback", {}).get("score"), (int, float))]
            avg = round(sum(scores) / len(scores), 1) if scores else "—"
            st.metric("Avg Score", f"{avg}/10")
        else:
            st.metric("Avg Score", "—")

    st.divider()

    # -- Generate question if none exists --
    if st.session_state.interview_question is None:
        with st.spinner("Generating a personalized question..."):
            st.session_state.interview_question = generate_question(weak_topics)
            st.session_state.interview_feedback = None

    # -- Display the question --
    st.markdown(
        f"""<div class="question-box">
        <p style="color: var(--muted); font-size: 0.8rem; margin-bottom: 0.5rem; font-weight: 500;">INTERVIEW QUESTION</p>
        <p style="font-size: 1.1rem; line-height: 1.6; color: var(--text);">{st.session_state.interview_question}</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # -- Answer input --
    answer_text = st.text_area(
        "Your Answer",
        height=180,
        placeholder="Type your answer here... Be detailed and explain your reasoning.",
        key="interview_answer_input",
    )

    # -- Voice input --
    st.caption("Or answer with your voice:")
    voice_input = st.audio_input(
        "🎙️ Record your answer",
        key=f"interview_audio_{st.session_state.audio_uploader_key}",
        disabled=not bool(client),
    )

    transcribed_text = None
    if voice_input and client is not None:
        try:
            audio_bytes = voice_input.read()
            if audio_bytes:
                with st.spinner("Transcribing your answer..."):
                    resp = client.audio.transcriptions.create(
                        file=("input.wav", audio_bytes, "audio/wav"),
                        model=AUDIO_MODEL,
                        response_format="text",
                    )
                    transcribed_text = resp if isinstance(resp, str) else getattr(resp, "text", None) or str(resp)
                    st.info(f"🎙️ **Transcribed:** {transcribed_text}")
        except Exception as e:
            logger.error("Transcription error: %s", e)
            st.warning("Transcription failed. Please use text input.")

    # -- Submit --
    final_answer = transcribed_text if transcribed_text else answer_text

    col_submit, col_skip = st.columns([3, 1])

    with col_submit:
        submit_clicked = st.button("📤 Submit Answer", use_container_width=True, disabled=not final_answer)

    with col_skip:
        skip_clicked = st.button("⏭️ Skip", use_container_width=True)

    if skip_clicked:
        st.session_state.interview_question = None
        st.session_state.interview_feedback = None
        reset_audio_widget()
        st.rerun()

    if submit_clicked and final_answer:
        with st.spinner("Analyzing your answer..."):
            # Get feedback
            result = analyze_answer(st.session_state.interview_question, final_answer)
            # Detect topic
            topic = detect_topic(st.session_state.interview_question, final_answer)

            # Save to memory
            save_interview(USER_ID, {
                "question": st.session_state.interview_question,
                "answer": final_answer,
                "feedback": result,
                "topic": topic,
            })

            st.session_state.interview_feedback = result
            st.session_state.interview_feedback["topic"] = topic
            st.session_state.interview_count += 1

    # -- Display feedback --
    if st.session_state.interview_feedback:
        fb = st.session_state.interview_feedback
        score = fb.get("score", 0)
        sentiment = fb.get("sentiment", {})
        evaluation = fb.get("evaluation", "")
        topic = fb.get("topic", "General")

        # Score badge class
        if score >= 7:
            badge_class = "score-high"
        elif score >= 4:
            badge_class = "score-mid"
        else:
            badge_class = "score-low"

        st.markdown("---")

        st.markdown(
            f"""<div class="feedback-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <span style="font-weight: 600; font-size: 1.1rem; color: var(--text);">📊 Feedback</span>
                <span class="score-badge {badge_class}">{score}/10</span>
            </div>
            <div style="margin-bottom: 0.8rem;">
                <span class="topic-tag">🏷️ {topic}</span>
                <span class="topic-tag">💬 {sentiment.get('label', 'N/A')} ({sentiment.get('score', '—')})</span>
            </div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown(evaluation)

        st.markdown("---")

        if st.button("➡️ Next Question", use_container_width=True):
            st.session_state.interview_question = None
            st.session_state.interview_feedback = None
            reset_audio_widget()
            st.rerun()

    # -- Optional: OpenCV Attention Detection --
    with st.expander("📷 Posture / Attention Check (Optional)", expanded=False):
        st.caption("Use your camera to check posture during the interview.")
        camera_img = st.camera_input("Capture", key="interview_camera")
        if camera_img is not None:
            try:
                with st.spinner("Analyzing..."):
                    annotated, objects = detect_objects_with_opencv(camera_img.getvalue())
                st.image(annotated, caption="Detection result", use_container_width=True)
                face_detected = any("Face" in obj for obj in objects)
                if face_detected:
                    st.success("✅ Good posture — face detected, stay focused!")
                else:
                    st.warning("⚠️ Face not detected — adjust your position and lighting.")
            except Exception as exc:
                logger.exception("OpenCV detection failed")
                st.warning(f"Detection unavailable: {exc}")


# ================================================================
# 💬 CHAT MODE (Original functionality preserved)
# ================================================================
elif st.session_state.app_mode == "💬 Chat":
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
