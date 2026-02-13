import logging
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_sentiment_pipeline():
    """Load transformer-based sentiment model once per process."""
    from transformers import pipeline

    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )


def detect_sentiment(text: str) -> Dict[str, str]:
    """Return sentiment label/score for a text prompt."""
    cleaned = (text or "").strip()
    if not cleaned:
        return {"label": "NEUTRAL", "score": "0.00"}

    try:
        sentiment_pipeline = _load_sentiment_pipeline()
        result = sentiment_pipeline(cleaned[:512])[0]
        return {
            "label": result.get("label", "NEUTRAL").upper(),
            "score": f"{float(result.get('score', 0.0)):.2f}",
        }
    except Exception as exc:
        logger.warning("Sentiment pipeline unavailable, using fallback: %s", exc)
        positive_words = {"great", "good", "awesome", "happy", "love", "excellent"}
        negative_words = {"bad", "sad", "angry", "hate", "terrible", "upset"}

        tokens = set(cleaned.lower().split())
        if tokens.intersection(positive_words):
            return {"label": "POSITIVE", "score": "0.55"}
        if tokens.intersection(negative_words):
            return {"label": "NEGATIVE", "score": "0.55"}
        return {"label": "NEUTRAL", "score": "0.50"}


def _run_multiscale_passes(detector, gray: np.ndarray, min_size: Tuple[int, int]):
    """Try a few detection parameter combinations and keep the richest hit set."""
    import cv2

    best = ()
    passes = (
        (1.05, 3),
        (1.1, 4),
        (1.15, 5),
    )
    for scale_factor, min_neighbors in passes:
        hits = detector.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(hits) > len(best):
            best = hits
    return best


def detect_objects_with_opencv(image_bytes: bytes) -> Tuple[np.ndarray, List[str]]:
    """Detect objects (faces and bodies) using OpenCV Haar cascades."""
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is not installed. Install opencv-python-headless.") from exc

    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    detectors = [
        ("Face", cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"), (0, 255, 0)),
        ("Profile Face", cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml"), (0, 200, 255)),
        ("Upper Body", cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml"), (255, 140, 0)),
        ("Full Body", cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml"), (255, 0, 255)),
    ]
    eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    detections: List[str] = []
    h, w = gray.shape[:2]
    min_face = (max(30, w // 14), max(30, h // 14))
    min_body = (max(40, w // 8), max(40, h // 8))

    for label, detector, color in detectors:
        if detector.empty():
            logger.warning("OpenCV cascade missing for %s", label)
            continue

        min_size = min_body if "Body" in label else min_face
        hits = _run_multiscale_passes(detector, gray, min_size)
        if len(hits) == 0:
            hits = _run_multiscale_passes(detector, gray_eq, min_size)

        for idx, (x, y, bw, bh) in enumerate(hits, start=1):
            cv2.rectangle(image, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(image, f"{label} {idx}", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            detections.append(f"{label} {idx}")

            if "Face" in label and not eye_detector.empty():
                roi_gray = gray[y:y + bh, x:x + bw]
                roi_color = image[y:y + bh, x:x + bw]
                eyes = _run_multiscale_passes(eye_detector, roi_gray, (12, 12))
                for eye_idx, (ex, ey, ew, eh) in enumerate(eyes, start=1):
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 200, 0), 2)
                    detections.append(f"{label} {idx} - Eye {eye_idx}")

    annotated = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return annotated, detections
