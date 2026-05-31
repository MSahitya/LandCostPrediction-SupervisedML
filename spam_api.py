"""
Email Spam Detection - Production REST API

Serves a trained spam classifier via FastAPI with:
  - Single and batch prediction endpoints
  - Explainable spam signals per prediction
  - Structured JSON request logging
  - Health and model-info endpoints
  - Graceful 503 when model is not yet trained

Start server:
    python spam_train.py          # train the model first
    python spam_api.py            # then start the API

Endpoints:
    POST /predict            — classify a single email
    POST /predict/batch      — classify up to 100 emails at once
    GET  /health             — liveness probe
    GET  /model/info         — model metadata + training stats
"""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ── Paths (must match spam_train.py) ─────────────────────────────────────────

MODEL_PATH = Path("spam_detector_pipeline.joblib")
METADATA_PATH = Path("spam_detector_metadata.json")

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("spam_api")

# ── Global model state ────────────────────────────────────────────────────────

_model = None
_metadata: dict = {}

# ── Spam signal detection (independent of ML model) ──────────────────────────

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_CAPS_RE = re.compile(r"[A-Z]")
_SUSPICIOUS_DOMAIN_RE = re.compile(
    r"\.(xyz|top|click|loan|work|stream|accountant|gq|ml|tk|cf|ga)$", re.I
)
_URGENCY_PATTERNS = re.compile(
    r"\b(act now|limited time|urgent|expires|immediate|don'?t miss|last chance|"
    r"final notice|respond now|click here|claim now|verify now)\b",
    re.I,
)
_MONEY_PATTERNS = re.compile(
    r"\b(free money|earn money|make money|million dollar|prize|lottery|winner|"
    r"investment opportunity|profit|guaranteed income)\b",
    re.I,
)
_PHISH_PATTERNS = re.compile(
    r"\b(verify your account|confirm your password|account suspended|"
    r"security alert|unusual activity|click to verify|enter your password)\b",
    re.I,
)


def detect_spam_signals(email: dict) -> list[str]:
    """Return a list of human-readable spam signal labels found in the email."""
    signals = []
    subject = str(email.get("subject", "") or "")
    body = str(email.get("body", "") or "")
    sender = str(email.get("sender", "") or "")
    full_text = subject + " " + body
    clean_text = _HTML_TAG_RE.sub(" ", full_text)
    text_len = max(len(clean_text.replace(" ", "")), 1)

    caps_ratio = len(_CAPS_RE.findall(clean_text)) / text_len
    if caps_ratio > 0.25:
        signals.append("excessive_capitalization")

    if full_text.count("!") > 3:
        signals.append("excessive_exclamation_marks")

    url_count = len(_URL_RE.findall(full_text))
    if url_count > 2:
        signals.append("multiple_urls")

    if full_text.count("$") > 2:
        signals.append("heavy_monetary_language")

    if _URGENCY_PATTERNS.search(full_text):
        signals.append("urgency_language")

    if _MONEY_PATTERNS.search(full_text):
        signals.append("money_offer_detected")

    if _PHISH_PATTERNS.search(full_text):
        signals.append("phishing_pattern")

    if subject.upper() == subject and len(subject) > 5:
        signals.append("subject_all_caps")

    if subject.count("!") > 1:
        signals.append("subject_excessive_punctuation")

    sender_domain = sender.split("@")[-1].lower() if "@" in sender else ""
    if _SUSPICIOUS_DOMAIN_RE.search(sender_domain):
        signals.append("suspicious_sender_domain")

    if bool(email.get("is_html")) and len(_HTML_TAG_RE.findall(full_text)) > 20:
        signals.append("heavy_html_content")

    return signals


def risk_level(spam_probability: float) -> str:
    if spam_probability >= 0.85:
        return "critical"
    if spam_probability >= 0.65:
        return "high"
    if spam_probability >= 0.40:
        return "medium"
    return "low"


# ── Pydantic models ───────────────────────────────────────────────────────────

class EmailRequest(BaseModel):
    subject: str = Field(default="", max_length=998, description="Email subject line")
    body: str = Field(default="", max_length=100_000, description="Email body text or HTML")
    sender: str = Field(default="", max_length=320, description="Sender email address")
    sender_name: Optional[str] = Field(default=None, max_length=256)
    is_html: bool = Field(default=False, description="True if body contains HTML")
    num_attachments: int = Field(default=0, ge=0, le=999)

    @field_validator("subject", "body", "sender", mode="before")
    @classmethod
    def coerce_none_to_empty(cls, v):
        return v if v is not None else ""

    model_config = {"json_schema_extra": {
        "example": {
            "subject": "URGENT: Claim your $1,000,000 prize NOW!!!",
            "body": "Congratulations! You have been selected as our lucky winner...",
            "sender": "prizes@suspicious-domain.xyz",
            "is_html": False,
        }
    }}


class SpamPrediction(BaseModel):
    is_spam: bool
    spam_probability: float
    ham_probability: float
    confidence: str          # "high" | "medium" | "low"
    risk_level: str          # "critical" | "high" | "medium" | "low"
    spam_signals: list[str]  # human-readable signals that contributed
    request_id: str
    inference_time_ms: float
    model_version: str


class BatchEmailRequest(BaseModel):
    emails: list[EmailRequest] = Field(..., min_length=1, max_length=100)


class BatchSpamPrediction(BaseModel):
    results: list[SpamPrediction]
    total: int
    spam_count: int
    ham_count: int
    request_id: str
    inference_time_ms: float


# ── Application lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _metadata
    if MODEL_PATH.exists():
        log.info("Loading model from %s ...", MODEL_PATH)
        _model = joblib.load(MODEL_PATH)
        log.info("Model loaded successfully.")
    else:
        log.warning(
            "Model file not found at '%s'. Run 'python spam_train.py' to train first. "
            "API will return 503 until the model is available.",
            MODEL_PATH,
        )

    if METADATA_PATH.exists():
        _metadata = json.loads(METADATA_PATH.read_text())

    yield

    _model = None
    _metadata = {}
    log.info("API shut down.")


# ── FastAPI application ───────────────────────────────────────────────────────

app = FastAPI(
    title="Email Spam Detector API",
    description=(
        "Real-time email spam classification using an ensemble of ML classifiers. "
        "Combines TF-IDF text features with 19 handcrafted email-structure signals."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request logging middleware ────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    rid = str(uuid.uuid4())
    request.state.request_id = rid
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    log.info(
        "method=%s path=%s status=%d duration_ms=%.1f request_id=%s",
        request.method, request.url.path, response.status_code, elapsed, rid,
    )
    response.headers["X-Request-ID"] = rid
    return response


# ── Helper: require model ─────────────────────────────────────────────────────

def _require_model():
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "model_not_loaded",
                "message": "Model is not available. Run 'python spam_train.py' first.",
            },
        )


# ── Core prediction logic ─────────────────────────────────────────────────────

def _predict_one(email_dict: dict, request_id: str) -> SpamPrediction:
    _require_model()
    t0 = time.perf_counter()

    features = [email_dict]
    proba = _model.predict_proba(features)[0]  # [ham_prob, spam_prob]
    spam_prob = float(proba[1])
    ham_prob = float(proba[0])
    is_spam = spam_prob >= 0.50

    conf_score = abs(spam_prob - 0.50) * 2  # 0→uncertain, 1→certain
    confidence = "high" if conf_score > 0.6 else ("medium" if conf_score > 0.3 else "low")

    signals = detect_spam_signals(email_dict)
    elapsed = (time.perf_counter() - t0) * 1000

    return SpamPrediction(
        is_spam=is_spam,
        spam_probability=round(spam_prob, 4),
        ham_probability=round(ham_prob, 4),
        confidence=confidence,
        risk_level=risk_level(spam_prob),
        spam_signals=signals,
        request_id=request_id,
        inference_time_ms=round(elapsed, 2),
        model_version=_metadata.get("model_version", "unknown"),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/predict",
    response_model=SpamPrediction,
    summary="Classify a single email as spam or ham",
    tags=["prediction"],
)
def predict(request: EmailRequest, req: Request):
    rid = getattr(req.state, "request_id", str(uuid.uuid4()))
    email_dict = request.model_dump()
    result = _predict_one(email_dict, rid)
    log.info(
        "predict request_id=%s is_spam=%s spam_prob=%.3f signals=%s",
        rid, result.is_spam, result.spam_probability, result.spam_signals,
    )
    return result


@app.post(
    "/predict/batch",
    response_model=BatchSpamPrediction,
    summary="Classify up to 100 emails in one call",
    tags=["prediction"],
)
def predict_batch(payload: BatchEmailRequest, req: Request):
    _require_model()
    rid = getattr(req.state, "request_id", str(uuid.uuid4()))
    t0 = time.perf_counter()

    email_dicts = [e.model_dump() for e in payload.emails]
    probas = _model.predict_proba(email_dicts)

    results = []
    for i, (email_dict, proba) in enumerate(zip(email_dicts, probas)):
        spam_prob = float(proba[1])
        ham_prob = float(proba[0])
        is_spam = spam_prob >= 0.50
        conf_score = abs(spam_prob - 0.50) * 2
        confidence = "high" if conf_score > 0.6 else ("medium" if conf_score > 0.3 else "low")

        results.append(SpamPrediction(
            is_spam=is_spam,
            spam_probability=round(spam_prob, 4),
            ham_probability=round(ham_prob, 4),
            confidence=confidence,
            risk_level=risk_level(spam_prob),
            spam_signals=detect_spam_signals(email_dict),
            request_id=f"{rid}-{i}",
            inference_time_ms=0.0,  # filled below
            model_version=_metadata.get("model_version", "unknown"),
        ))

    total_ms = round((time.perf_counter() - t0) * 1000, 2)
    per_email_ms = round(total_ms / max(len(results), 1), 2)
    for r in results:
        r.inference_time_ms = per_email_ms

    spam_count = sum(1 for r in results if r.is_spam)
    log.info(
        "batch request_id=%s n=%d spam=%d ham=%d elapsed_ms=%.1f",
        rid, len(results), spam_count, len(results) - spam_count, total_ms,
    )
    return BatchSpamPrediction(
        results=results,
        total=len(results),
        spam_count=spam_count,
        ham_count=len(results) - spam_count,
        request_id=rid,
        inference_time_ms=total_ms,
    )


@app.get(
    "/health",
    summary="Liveness and readiness probe",
    tags=["ops"],
)
def health():
    return {
        "status": "ok" if _model is not None else "degraded",
        "model_loaded": _model is not None,
        "model_version": _metadata.get("model_version", "not_trained"),
    }


@app.get(
    "/model/info",
    summary="Model metadata and training statistics",
    tags=["ops"],
)
def model_info():
    if not _metadata:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No model metadata available. Train a model first.",
        )
    return _metadata


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("spam_api:app", host="0.0.0.0", port=8001, reload=True)
