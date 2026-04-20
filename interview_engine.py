# interview_engine.py
"""
Core interview logic: personalized question generation and topic detection.
Uses Groq LLM via LangChain for intelligent question targeting.
"""
import logging
from typing import List, Optional

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

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
            raise RuntimeError("GROQ_API_KEY is required for interview engine.")
        _llm = ChatGroq(
            temperature=0.7,
            model_name=PRIMARY_MODEL,
            groq_api_key=GROQ_API_KEY,
        )
    return _llm


# ---------------------
# Prompts
# ---------------------
QUESTION_PROMPT = PromptTemplate.from_template("""You are a senior technical interviewer conducting a mock interview.

The candidate has the following weak topics that need improvement:
{weak_topics}

Rules:
1. Ask exactly ONE interview question.
2. Focus on the weak topics listed above.
3. The question should be practical and test real understanding, not just definitions.
4. Vary difficulty — sometimes ask conceptual, sometimes coding/scenario-based.
5. Do NOT include the answer. Only output the question.
6. If no weak topics are provided, ask a general software engineering question.

Your question:""")


TOPIC_DETECT_PROMPT = PromptTemplate.from_template("""Analyze this interview Q&A and identify the primary technical topic.

Question: {question}
Answer: {answer}

Return ONLY the topic name as a short label (1-3 words). Examples: "Python OOP", "SQL Joins", "System Design", "React Hooks", "Data Structures", "REST APIs".

Topic:""")


# ---------------------
# Core Functions
# ---------------------
def generate_question(weak_topics: List[str]) -> str:
    """Generate a single interview question targeting weak areas."""
    llm = _get_llm()

    topics_str = ", ".join(weak_topics) if weak_topics else "General programming"
    prompt = QUESTION_PROMPT.format(weak_topics=topics_str)

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as exc:
        logger.error("Question generation failed: %s", exc)
        return "Explain the difference between a stack and a queue. When would you use each?"


def detect_topic(question: str, answer: str) -> str:
    """Auto-detect the topic of a Q&A pair using the LLM."""
    llm = _get_llm()

    prompt = TOPIC_DETECT_PROMPT.format(question=question, answer=answer)

    try:
        response = llm.invoke(prompt)
        topic = response.content.strip().strip('"').strip("'")
        # Sanitize: keep it short
        return topic[:50] if topic else "General"
    except Exception as exc:
        logger.warning("Topic detection failed, using fallback: %s", exc)
        return "General"
