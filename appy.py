import streamlit as st
import logging
import time
import os
from typing import List, Dict

# --- LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- App Config & Backend ---
from config import GROQ_API_KEY, PRIMARY_MODEL, CONTEXT_WINDOW, AUDIO_MODEL
from memory_manager import (
    load_recent_conversations,
    save_conversation,
    get_context_history,
    get_mongo_collection,
    delete_last_conversation
)
from groq import Groq

# ======================
# Setup & Logging
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI With Memory", page_icon="🎙️", layout="centered")

# Minimal styling
st.markdown("""
    <style>
        .block-container { padding-top: 3rem; max-width: 800px; }
        header, footer { visibility: hidden; }
        .stTextInput > div > div > input { border-radius: 8px; }
        .stAudioInput { margin-top: 10px; }
        /* Style for the Undo button to make it distinct */
        button[kind="secondary"] { border-color: #ff4b4b; color: #ff4b4b; }
    </style>
""", unsafe_allow_html=True)

# ======================
# Initialize Clients
# ======================
# 1. LangChain Chat Model (The Brain)
llm = ChatGroq(
    temperature=0.7,
    model_name=PRIMARY_MODEL,
    groq_api_key=GROQ_API_KEY
)

# 2. Raw Groq Client (The Ears - for Whisper)
client = Groq(api_key=GROQ_API_KEY)

# ======================
# Session State Management
# ======================
USER_ID = "demo_user"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Dynamic key for audio widget to allow forced resets
if "audio_uploader_key" not in st.session_state:
    st.session_state.audio_uploader_key = 0

def reset_audio_widget():
    """Increments the key to force Streamlit to redraw (reset) the audio widget."""
    st.session_state.audio_uploader_key += 1

# ======================
# Helper: Audio Processing
# ======================
def process_audio(audio_bytes):
    """Transcribe audio using Groq's Whisper API"""
    try:
        return client.audio.transcriptions.create(
            file=("input.wav", audio_bytes, "audio/wav"),
            model=AUDIO_MODEL,  # Uses 'whisper-large-v3-turbo' from config
            response_format="text"
        )
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return None

# ======================
# Sidebar: Settings & Actions
# ======================
with st.sidebar:
    st.title("Agent Settings")
    st.caption(f"🧠 Model: {PRIMARY_MODEL}")
    
    if get_mongo_collection() is not None:
        st.caption("🟢 Memory: Active")
    else:
        st.caption("🔴 Memory: Offline")

    st.divider()

    # --- UNDO / DELETE ACTIONS ---
    if st.button("↩️ Undo Last Message", use_container_width=True):
        # 1. Remove from Screen (Session State)
        if len(st.session_state.messages) >= 2:
            st.session_state.messages.pop() # Remove AI response
            st.session_state.messages.pop() # Remove User prompt
        
        # 2. Remove from Database
        delete_last_conversation(USER_ID)
        
        # 3. Reset Audio Widget (Clears the old recording)
        reset_audio_widget()
        
        st.toast("Last interaction deleted.")
        st.rerun()

    if st.button("🗑️ Clear All History", use_container_width=True):
        st.session_state.messages = []
        reset_audio_widget()
        st.rerun()

    st.divider()
    
    # Recent Conversations
    st.subheader("Memory Bank")
    past_memories = load_recent_conversations(USER_ID, limit=5)
    if past_memories:
        for u, a, t in past_memories:
            with st.expander(t, expanded=False):
                st.caption(f"User: {u[:50]}...")

# ======================
# Main Chat Interface
# ======================
st.title("🎙️ AI Voice Agent With Long Memory")
st.caption("Powered by LangChain + Groq + Whisper")

# 1. Render History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Capture Input
# The 'key' argument is what allows us to reset this widget programmatically
voice_input = st.audio_input("Voice Input", key=f"audio_{st.session_state.audio_uploader_key}")
text_input = st.chat_input("Type a message...")

prompt_text = None

# Logic: Process Voice if available, otherwise Text
if voice_input:
    with st.spinner("Processing voice..."):
        audio_bytes = voice_input.read()
        transcribed_text = process_audio(audio_bytes)
        if transcribed_text:
            prompt_text = transcribed_text
elif text_input:
    prompt_text = text_input

# ======================
# Response Generation
# ======================
if prompt_text:
    
    # Display User Message
    with st.chat_message("user"):
        st.markdown(prompt_text)
    st.session_state.messages.append({"role": "user", "content": prompt_text})

    # Build Context from MongoDB (Single Source of Truth)
    history_context = ""
    mongo_hist = get_context_history(USER_ID, limit=CONTEXT_WINDOW)
    for entry in mongo_hist:
        history_context += f"{entry['role'].capitalize()}: {entry['content']}\n"

    # Define LangChain Pipeline
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer concisely."),
        ("system", "Here is the relevant conversation history:\n{context}"),
        ("user", "{input}")
    ])
    
    chain = prompt_template | llm | StrOutputParser()

    # Generate & Stream Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            chunks = chain.stream({
                "context": history_context,
                "input": prompt_text
            })
            
            for chunk in chunks:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Save to Database
            save_conversation(USER_ID, prompt_text, full_response)
            
        except Exception as e:
            st.error(f"Generation Error: {e}")
            full_response = "I encountered an error generating the response."

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Optional: If you want the audio to auto-disappear after sending, uncomment below:
    reset_audio_widget()
    st.rerun()