# feedback_engine.py
"""
Answer analysis and scoring engine.
Combines transformer-based sentiment analysis with LLM evaluation
to produce structured feedback on interview answers.
"""
import logging
import re
from typing import Dict, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from ai_features import detect_sentiment
from config import GROQ_API_KEY, PRIMARY_MODEL

logger = logging.getLogger(__name__)

# ---------------------
# LLM Instance
# ---------------------
_llm: Optional[ChatGroq] = None


def _get_llm() -> ChatGroq:
    """Lazy-init the ChatGroq instance."""
    global _llm
    if _llm is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is required for feedback engine.")
        _llm = ChatGroq(
            temperature=0.4,
            model_name=PRIMARY_MODEL,
            groq_api_key=GROQ_API_KEY,
        )
    return _llm


# ---------------------
# Evaluation Prompt
# ---------------------
EVAL_PROMPT = PromptTemplate.from_template("""You are an expert interview evaluator. Analyze this multi-turn interview transcript thoroughly.

**Interview Transcript:**
{transcript}

Provide your evaluation in EXACTLY this format:

**Technical Score:** X/10
**Clarity Score:** X/10
**Overall Score:** X/10

**Strengths:**
- (list key strengths)

**Weak Areas:**
- (list areas for improvement)

**Improvement Suggestion:**
(specific, actionable advice to improve)

Be fair but constructive. If the answer is empty or irrelevant, score accordingly.""")


# ---------------------
# Score Extraction
# ---------------------
def _extract_score(evaluation: str) -> float:
    """Extract the overall numeric score from the LLM evaluation text."""
    # Try "Overall Score: X/10" pattern first
    match = re.search(r"Overall Score[:\s]*(\d+(?:\.\d+)?)\s*/\s*10", evaluation, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # Fallback: try "Technical Score" pattern
    match = re.search(r"Technical Score[:\s]*(\d+(?:\.\d+)?)\s*/\s*10", evaluation, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # Last resort: find any X/10 pattern and average them
    all_scores = re.findall(r"(\d+(?:\.\d+)?)\s*/\s*10", evaluation)
    if all_scores:
        nums = [float(s) for s in all_scores]
        return round(sum(nums) / len(nums), 1)

    return 5.0  # neutral default


# ---------------------
# Core Function
# ---------------------
def analyze_answer(qa_thread: list) -> Dict:
    """
    Analyze an interview thread and return structured feedback.
    qa_thread: list of dicts [{"role": "interviewer", "content": "..."}, {"role": "candidate", "content": "..."}]
    """
    # Combine answers for sentiment analysis
    all_answers = " ".join([msg["content"] for msg in qa_thread if msg["role"] == "candidate"])
    sentiment = detect_sentiment(all_answers)

    # Handle empty answers
    if not all_answers.strip():
        return {
            "sentiment": sentiment,
            "evaluation": "No answer was provided. Please attempt an answer — even a partial one shows thinking ability.",
            "score": 0.0,
        }

    # Format transcript
    transcript_lines = []
    for msg in qa_thread:
        role_label = "Interviewer" if msg["role"] == "interviewer" else "Candidate"
        transcript_lines.append(f"**{role_label}:** {msg['content']}")
    transcript = "\n\n".join(transcript_lines)

    # LLM-based evaluation
    llm = _get_llm()
    prompt = EVAL_PROMPT.format(transcript=transcript)

    try:
        response = llm.invoke(prompt)
        evaluation = response.content.strip()
        score = _extract_score(evaluation)

        return {
            "sentiment": sentiment,
            "evaluation": evaluation,
            "score": score,
        }

    except Exception as exc:
        logger.error("Answer evaluation failed: %s", exc)
        return {
            "sentiment": sentiment,
            "evaluation": "Evaluation temporarily unavailable. Please try again.",
            "score": 5.0,
        }
