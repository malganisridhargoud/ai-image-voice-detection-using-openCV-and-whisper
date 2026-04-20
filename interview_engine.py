# interview_engine.py
"""
Core interview logic: personalized question generation and topic detection.
Uses Groq LLM via LangChain for intelligent question targeting.
"""
import logging
from typing import List, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

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

The candidate wants to be interviewed on this topic: {chosen_topic}

Additionally, their weak areas include: {weak_topics}

Rules:
1. Ask exactly ONE interview question.
2. Focus primarily on the chosen topic.
3. The question should be practical and test real understanding, not just definitions.
4. Vary difficulty — sometimes ask conceptual, sometimes coding/scenario-based.
5. Do NOT include the answer. Only output the question.

Your question:""")


TOPIC_DETECT_PROMPT = PromptTemplate.from_template("""Analyze this interview Q&A and identify the primary technical topic.

Question: {question}
Answer: {answer}

Return ONLY the topic name as a short label (1-3 words). Examples: "Python OOP", "SQL Joins", "System Design", "React Hooks", "Data Structures", "REST APIs".

Topic:""")


FOLLOW_UP_PROMPT = PromptTemplate.from_template("""You are a senior technical interviewer.

Previous Question: {question}
Candidate's Answer: {answer}

Your task: Ask ONE follow-up question to probe deeper.
If their answer was shallow, ask them to explain further or provide an example.
If their answer was good, ask about an edge case, a trade-off, or how to scale it.
Do NOT give them the answer or say "Good job". Just ask the follow-up question directly.

Follow-up Question:""")


# ---------------------
# Available Interview Topics
# ---------------------
INTERVIEW_TOPICS = [
    "Python",
    "Java",
    "JavaScript",
    "React",
    "Node.js",
    "SQL & Databases",
    "Data Structures",
    "Algorithms",
    "System Design",
    "REST APIs",
    "OOP Concepts",
    "Git & Version Control",
    "Docker & DevOps",
    "Machine Learning",
    "Operating Systems",
    "Computer Networks",
    "HTML & CSS",
    "Cloud Computing (AWS/GCP)",
    "Cybersecurity Basics",
    "General Programming",
]


# ---------------------
# Core Functions
# ---------------------
def generate_question(weak_topics: List[str], chosen_topic: str = "") -> str:
    """Generate a single interview question targeting the chosen topic and weak areas."""
    llm = _get_llm()

    topic = chosen_topic if chosen_topic else "General programming"
    topics_str = ", ".join(weak_topics) if weak_topics else "None identified yet"
    prompt = QUESTION_PROMPT.format(chosen_topic=topic, weak_topics=topics_str)

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as exc:
        logger.error("Question generation failed: %s", exc)
        return "Explain the difference between a stack and a queue. When would you use each?"


def generate_followup(question: str, answer: str) -> str:
    """Generate a context-aware follow-up question based on the candidate's initial answer."""
    llm = _get_llm()

    prompt = FOLLOW_UP_PROMPT.format(question=question, answer=answer)

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as exc:
        logger.error("Follow-up generation failed: %s", exc)
        return "Can you elaborate more on your thought process there?"


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
