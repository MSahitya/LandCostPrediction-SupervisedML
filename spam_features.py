"""
Shared feature engineering for email spam detection.

Imported by both spam_train.py (training) and spam_api.py (inference).
Keeping this in a dedicated module is required so joblib can unpickle
the EmailFeatureExtractor class regardless of which script loads the model.
"""

import re
from typing import Optional

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# ── Patterns ──────────────────────────────────────────────────────────────────

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
HTML_TAG_RE = re.compile(r"<[^>]+>")
CAPS_RE = re.compile(r"[A-Z]")
DIGIT_RE = re.compile(r"\d")
SUSPICIOUS_DOMAIN_RE = re.compile(
    r"\.(xyz|top|click|loan|work|stream|accountant|gq|ml|tk|cf|ga)$", re.I
)
URGENCY_RE = re.compile(
    r"\b(act now|limited time|urgent|expires|immediate|don'?t miss|last chance|"
    r"final notice|respond now|click here|claim now|verify now)\b",
    re.I,
)
MONEY_RE = re.compile(
    r"\b(free money|earn money|make money|million dollar|prize|lottery|winner|"
    r"investment opportunity|profit|guaranteed income)\b",
    re.I,
)
PHISH_RE = re.compile(
    r"\b(verify your account|confirm your password|account suspended|"
    r"security alert|unusual activity|click to verify|enter your password)\b",
    re.I,
)

FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "aol.com", "mail.com", "yandex.com",
}

SPAM_KEYWORDS = [
    "free", "win", "winner", "cash", "prize", "lottery", "urgent", "act now",
    "limited time", "offer expires", "cheap", "guaranteed", "no obligation",
    "risk-free", "risk free", "money back", "earn money", "make money",
    "million dollars", "billion dollars", "credit card", "bank account",
    "verify your", "confirm your", "dear customer", "dear user",
    "congratulations", "you have been selected", "you have won", "inheritance",
    "wire transfer", "click here", "buy now", "order now", "subscribe now",
    "satisfy", "performance", "enhance", "enlarge", "pharmacy", "medication",
    "pills", "prescription", "singles in your area", "investment opportunity",
    "profit", "forex", "bitcoin", "crypto", "weight loss", "diet", "fat burn",
    "suspended", "blocked", "verify now", "account closed", "lucky draw",
    "processing fee", "claim your prize", "selected winner",
]

NUMERIC_FEATURE_NAMES = [
    "caps_ratio", "exclamation_ratio", "question_ratio",
    "url_count", "embedded_email_count", "dollar_count",
    "digit_ratio", "html_density", "spam_keyword_count",
    "is_html", "sender_free_domain", "sender_suspicious_domain",
    "sender_no_domain", "subject_empty", "subject_all_caps",
    "subject_has_re_fwd", "subject_exclamation_count",
    "avg_word_length", "body_length_normalized",
]


# ── Feature extractor (must stay importable for joblib pickle) ────────────────

class EmailFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Converts a list of email dicts into a sparse feature matrix:
      TF-IDF on (subject×2 + body)  +  19 numeric handcrafted features.

    All numeric features are MinMax-scaled to [0, 1] for compatibility with
    probability-based classifiers (ComplementNB, LogisticRegression).

    Input dict keys: subject (str), body (str), sender (str), is_html (bool).
    """

    def __init__(
        self,
        max_tfidf_features: int = 15_000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
    ):
        self.max_tfidf_features = max_tfidf_features
        self.ngram_range = ngram_range
        self.min_df = min_df

    def fit(self, X, y=None):
        texts = [self._text(e) for e in X]
        self.tfidf_ = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            stop_words="english",
        )
        self.tfidf_.fit(texts)

        numeric = np.array([self._numeric(e) for e in X], dtype=np.float32)
        self.scaler_ = MinMaxScaler()
        self.scaler_.fit(numeric)
        return self

    def transform(self, X):
        texts = [self._text(e) for e in X]
        text_feat = self.tfidf_.transform(texts)

        numeric = np.array([self._numeric(e) for e in X], dtype=np.float32)
        num_feat = csr_matrix(self.scaler_.transform(numeric))

        return hstack([text_feat, num_feat], format="csr")

    @staticmethod
    def _text(email: dict) -> str:
        subject = str(email.get("subject", "") or "")
        body = str(email.get("body", "") or "")
        return f"{subject} {subject} {body}"

    @staticmethod
    def _numeric(email: dict) -> list:
        subject = str(email.get("subject", "") or "")
        body = str(email.get("body", "") or "")
        sender = str(email.get("sender", "") or "")
        is_html = float(bool(email.get("is_html", False)))

        full_text = subject + " " + body
        clean_text = HTML_TAG_RE.sub(" ", full_text)
        text_len = max(len(clean_text), 1)
        words = clean_text.split()
        word_count = max(len(words), 1)
        text_lower = clean_text.lower()

        caps_ratio       = len(CAPS_RE.findall(clean_text)) / text_len
        digit_ratio      = len(DIGIT_RE.findall(clean_text)) / text_len
        exclamation_ratio = full_text.count("!") / word_count
        question_ratio   = full_text.count("?") / word_count
        url_count        = float(len(URL_RE.findall(full_text)))
        email_count      = float(len(EMAIL_RE.findall(full_text)))
        dollar_count     = float(full_text.count("$"))
        html_tag_count   = len(HTML_TAG_RE.findall(full_text))
        html_density     = html_tag_count / max(len(full_text), 1)
        spam_kw_count    = float(sum(1 for kw in SPAM_KEYWORDS if kw in text_lower))
        avg_word_len     = float(np.mean([len(w) for w in words])) if words else 0.0
        body_len_norm    = min(word_count / 1000.0, 1.0)

        sender_domain            = sender.split("@")[-1].lower() if "@" in sender else ""
        sender_free_domain       = float(sender_domain in FREE_EMAIL_DOMAINS)
        sender_suspicious_domain = float(bool(SUSPICIOUS_DOMAIN_RE.search(sender_domain)))
        sender_no_domain         = float("@" not in sender)

        subject_empty          = float(not subject.strip())
        subject_all_caps       = float(subject.upper() == subject and len(subject) > 3)
        subject_has_re_fwd     = float(subject.lower().startswith(("re:", "fwd:", "fw:")))
        subject_exclamations   = float(subject.count("!"))

        return [
            caps_ratio, exclamation_ratio, question_ratio,
            url_count, email_count, dollar_count,
            digit_ratio, html_density, spam_kw_count,
            is_html, sender_free_domain, sender_suspicious_domain,
            sender_no_domain, subject_empty, subject_all_caps,
            subject_has_re_fwd, subject_exclamations,
            avg_word_len, body_len_norm,
        ]


# ── Spam signal detector (rule-based, model-independent) ──────────────────────

def detect_spam_signals(email: dict) -> list[str]:
    """Return human-readable spam signal labels found in the email."""
    signals = []
    subject  = str(email.get("subject", "") or "")
    body     = str(email.get("body", "") or "")
    sender   = str(email.get("sender", "") or "")
    full_text = subject + " " + body
    clean_text = HTML_TAG_RE.sub(" ", full_text)
    text_len = max(len(clean_text.replace(" ", "")), 1)

    if len(CAPS_RE.findall(clean_text)) / text_len > 0.25:
        signals.append("excessive_capitalization")

    if full_text.count("!") > 3:
        signals.append("excessive_exclamation_marks")

    if len(URL_RE.findall(full_text)) > 2:
        signals.append("multiple_urls")

    if full_text.count("$") > 2:
        signals.append("heavy_monetary_language")

    if URGENCY_RE.search(full_text):
        signals.append("urgency_language")

    if MONEY_RE.search(full_text):
        signals.append("money_offer_detected")

    if PHISH_RE.search(full_text):
        signals.append("phishing_pattern")

    if subject.upper() == subject and len(subject) > 5:
        signals.append("subject_all_caps")

    if subject.count("!") > 1:
        signals.append("subject_excessive_punctuation")

    sender_domain = sender.split("@")[-1].lower() if "@" in sender else ""
    if SUSPICIOUS_DOMAIN_RE.search(sender_domain):
        signals.append("suspicious_sender_domain")

    if bool(email.get("is_html")) and len(HTML_TAG_RE.findall(full_text)) > 20:
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
