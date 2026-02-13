# AI Assistant With Memory

A Streamlit-based AI assistant with:
- chat responses (Groq + LangChain),
- voice-to-text input (Groq Whisper),
- sentiment analysis,
- OpenCV image detection (face/body),
- conversation memory (MongoDB + Redis).

## Tech Stack

- Language: `Python 3.10+`
- UI: `Streamlit`
- LLM + orchestration: `Groq`, `LangChain`, `langchain-groq`
- Speech transcription: `Groq Whisper` (`whisper-large-v3-turbo`)
- Computer vision: `OpenCV` (`opencv-python-headless`)
- Sentiment: `transformers` + `torch`
- Databases:
  - `MongoDB` (long-term chat memory, auth data)
  - `Redis` (context window memory)
- Auth/security: `bcrypt`
- Config: `python-dotenv`

## Project Files

- `app.py` - Streamlit app entry point
- `ai_features.py` - sentiment + OpenCV detection logic
- `memory_manager.py` - MongoDB/Redis memory operations
- `auth.py` - login/register logic
- `config.py` - environment/config loading
- `requirements.txt` - Python dependencies

## Setup (Windows / PowerShell)

1. Open terminal in project folder:
```powershell
cd C:\Users\acer\Downloads\LOOPS
```

2. Create virtual environment (if not already created):
```powershell
python -m venv venv
```

3. Activate virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```

4. Install dependencies:
```powershell
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in project root with:

```env
GROQ_API_KEY=your_groq_api_key
MONGODB_URI=your_mongodb_uri
REDIS_URL=your_redis_url
PRIMARY_MODEL=llama-3.3-70b-versatile
AUDIO_MODEL=whisper-large-v3-turbo
CONTEXT_WINDOW=5
```

Notes:
- If `GROQ_API_KEY` is missing, chat/voice features are disabled (UI still loads).
- If DB values are missing, auth/memory features may run in limited mode.

## Run the App

Use either command:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

or (after venv activation):

```powershell
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open in browser:
- `http://localhost:8501`
- or `http://127.0.0.1:8501`

Do not open `http://0.0.0.0:8501` in browser; `0.0.0.0` is only a bind address.

## Quick Usage

1. Login/register, or continue as guest.
2. Type chat input or record voice input.
3. Use camera/upload image in **Image Detection** section.
4. View recent memory in sidebar.

## Troubleshooting

- Port busy (`Port 8501 is not available`):
  - stop existing Streamlit process or run on another port:
  ```powershell
  streamlit run app.py --server.port 8502
  ```

- OpenCV import error (`No module named cv2`):
  - ensure app runs from project venv and dependencies are installed.

- App not reachable:
  - use `localhost` or `127.0.0.1`, not `0.0.0.0`.
