# AI Interview Coach with Memory + Feedback Loop
An AI-powered mock interview system that conducts **personalized technical interviews** based on your weak areas. Built with Groq LLMs, LangChain, and a memory-driven feedback loop.

> Built an AI Interview Coach using Groq + LangChain that conducts personalized mock interviews based on user weak areas. Implemented a feedback engine for answer evaluation and used Redis (session memory) and MongoDB (long-term tracking) to improve question relevance over time. Integrated voice input (Whisper) and sentiment analysis for confidence assessment.

---

## Key Features

| Feature | Description |
|---|---|
| **Dynamic Probing** | AI acts as a real interviewer, asking context-aware follow-up questions based on your answers to test depth and edge cases |
| **Topic Selection** | Choose your interview focus from 20+ technical topics (React, Python, System Design, etc.) |
| **Answer Evaluation** | AI scores your complete multi-turn thread on technical accuracy, clarity, and provides improvement suggestions |
| **Feedback Loop** | Weak topics are tracked over time — questions adapt as you improve |
| **Voice Input** | Answer with your voice using Groq Whisper transcription |
| **Sentiment Analysis** | Confidence scoring via transformer-based sentiment analysis |
| **Attention Detection** | Optional OpenCV posture/face detection during interviews |
| **Chat Mode** | Full AI assistant with memory, voice, and image detection |
| **Dual Storage** | MongoDB (long-term history) + Redis (session context) with local fallback |
| **Auth System** | bcrypt-based login/register with guest preview mode |

---

## Architecture

```
app.py                    ← Dual-mode Streamlit UI (Interview Coach + Chat)
 ├── auth.py              ← User login / register (bcrypt + MongoDB)
 ├── interview_engine.py  ← Question generation + topic detection (LLM)
 ├── feedback_engine.py   ← Answer analysis + scoring (LLM + sentiment)
 ├── memory_manager.py    ← Conversation + interview persistence (Mongo/Redis/local)
 ├── ai_features.py       ← Sentiment analysis + OpenCV detection
 └── config.py            ← Environment config loading
```

---

## Tech Stack

- **Language:** Python 3.10+
- **UI:** Streamlit
- **LLM + Orchestration:** Groq, LangChain, langchain-groq
- **Speech Transcription:** Groq Whisper (`whisper-large-v3-turbo`)
- **Computer Vision:** OpenCV (`opencv-python-headless`)
- **Sentiment Analysis:** transformers + torch (DistilBERT)
- **Databases:**
  - MongoDB — long-term interview history, auth, weak topic aggregation
  - Redis — session context, live interview state
- **Auth:** bcrypt
- **Config:** python-dotenv

---

## How It Works

### Interview Coach Mode (🎯)

1. **Topic Selection** — Choose a specific technology or algorithmic topic to be interviewed on
2. **Weak Topic Detection** — MongoDB aggregation pipeline finds your historically lowest-scoring topics to prioritize weaknesses
3. **Question Generation** — LLM creates a targeted interview question 
4. **Answer Submission** — Type or speak your answer (Whisper transcription)
5. **Dynamic Probing** — Instead of generic feedback, the AI generates a follow-up question probing your reasoning
6. **Thread Evaluation** — The Feedback engine scores your *entire transcript* (original answer + follow-up defense) providing a 1-10 grade
7. **Memory Update** — Results saved to MongoDB; weak topics recalculate for next round

### Chat Mode (💬)

Original AI assistant functionality:
- Chat with memory and context history
- Voice input via Whisper
- OpenCV image/face/body detection
- Sentiment-aware responses

---

## Setup

### 1. Clone and enter the project

```powershell
git clone https://github.com/malganisridhargoud/ai-image-voice-detection-using-openCV-and-whisper.git
cd ai-image-voice-detection-using-openCV-and-whisper
```

### 2. Create and activate virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
MONGODB_URI=your_mongodb_uri
REDIS_URL=your_redis_url
PRIMARY_MODEL=llama-3.3-70b-versatile
AUDIO_MODEL=whisper-large-v3-turbo
CONTEXT_WINDOW=5
```

**Notes:**
- `GROQ_API_KEY` is **required** for interview and chat features (UI loads without it in preview mode)
- `MONGODB_URI` / `REDIS_URL` are optional — the app falls back to local in-memory storage

### 5. Run the app

```powershell
streamlit run app.py --server.port 8501
```

Open `http://localhost:8501` in your browser.

---

## Usage

1. **Login / Register** or continue as guest
2. Select mode in the sidebar: **🎯 Interview Coach** or **💬 Chat**
3. In Interview Coach:
   - Select your desired technical topic
   - Read the generated question and submit your initial answer
   - Defend your reasoning in the generated *Follow-up Question*
   - Submit → view AI feedback on your full interview thread
   - Click "Start Next Interview Topic" to continue
4. In Chat mode: type or speak, upload images for detection

---

## Project Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dual-mode UI (Interview Coach + Chat) |
| `interview_engine.py` | LLM question generation + topic detection |
| `feedback_engine.py` | Answer evaluation + scoring engine |
| `memory_manager.py` | MongoDB/Redis/local memory operations |
| `ai_features.py` | Sentiment analysis + OpenCV detection |
| `auth.py` | bcrypt login/register logic |
| `config.py` | Environment/config loading |
| `requirements.txt` | Python dependencies |

---

## Troubleshooting

- **Port busy:** Run on another port: `streamlit run app.py --server.port 8502`
- **OpenCV import error:** Ensure app runs from project venv with dependencies installed
- **No questions generated:** Verify `GROQ_API_KEY` is set in `.env`
- **Memory offline:** App works without MongoDB/Redis using local in-memory fallback
