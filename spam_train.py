"""
Email Spam Detection - Training Pipeline

Trains multiple classification algorithms on combined TF-IDF + handcrafted
features, evaluates via stratified cross-validation, and saves the best model.

Usage:
    python spam_train.py                     # Train on built-in synthetic data
    python spam_train.py --csv emails.csv    # Train on CSV (columns: subject,body,sender,label)
    python spam_train.py --samples 5000      # Synthetic dataset size
"""

import sys
import json
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
import joblib

from spam_features import (
    EmailFeatureExtractor,
    NUMERIC_FEATURE_NAMES,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_PATH = Path("spam_detector_pipeline.joblib")
METADATA_PATH = Path("spam_detector_metadata.json")

# ── Synthetic dataset ─────────────────────────────────────────────────────────

def generate_synthetic_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Return a balanced synthetic spam/ham DataFrame for demonstration."""
    rng = np.random.default_rng(seed)

    spam_templates = [
        {
            "subject": "URGENT: You've been selected as our lucky WINNER!!!",
            "body": "Congratulations! Your email address was selected from 50 million addresses. You have WON $2,500,000 USD. To claim your prize IMMEDIATELY click: http://prize-claim.xyz. Send processing fee of $99. Act NOW - expires in 24 hours!!!",
            "sender": "no-reply@prize-claim.xyz",
        },
        {
            "subject": "FREE MONEY - Claim your $5,000,000 today!",
            "body": "DEAR WINNER, You have been approved for a loan of $5,000,000! No credit check! No obligation! 100% risk-free guaranteed! Click here: http://free-money-now.top/claim. Limited time offer. Buy now and receive BONUS cash! ACT NOW!!!",
            "sender": "offers@free-money-now.top",
        },
        {
            "subject": "Your PayPal account has been SUSPENDED - Verify NOW",
            "body": "Dear Valued Customer, Your account has been suspended due to suspicious activity. Verify your account immediately to avoid permanent closure. Click here: http://paypal-secure.click/verify. Enter your username, password and credit card details to confirm your identity.",
            "sender": "security@paypal-secure.click",
        },
        {
            "subject": "MAKE $5000/week from home - No experience needed!!!",
            "body": "Earn $500-$3000 per DAY working from home! Our proven system makes money 24/7! No experience required! Join thousands already making HUGE profits online! Click here: http://easy-income.loan. FREE training included! 100% guaranteed income! ACT NOW limited spots!!!",
            "sender": "support@easy-income.loan",
        },
        {
            "subject": "Lose 30 POUNDS in 30 days - DOCTORS HATE THIS!!!",
            "body": "Secret weight loss trick revealed! Burn fat fast without diet or exercise! Guaranteed results or FULL money back! Buy 2 get 1 FREE today only! Click: http://miracle-diet.work. Limited time discount - 90% OFF! Order now before stock runs out!!!",
            "sender": "health@miracle-diet.work",
        },
        {
            "subject": "Re: Urgent business transfer - Mr. Prince James",
            "body": "DEAR FRIEND, I am a Nigerian prince who needs your urgent help. I have $45,000,000 USD to transfer abroad. You will receive 30% commission for your bank account details. This is 100% risk-free and legitimate! Please respond immediately with your full name and bank account number.",
            "sender": "prince.james@freemail.top",
        },
        {
            "subject": "HOT SINGLES in your area are waiting for YOU!!!",
            "body": "Beautiful singles near you want to meet TONIGHT! Join for FREE: http://singles-near-you.xyz. Thousands of active members. Discreet and confidential. See who's online RIGHT NOW in your area! No subscription needed to browse profiles! Click here to view matches!",
            "sender": "matches@singles-near-you.xyz",
        },
        {
            "subject": "Cheap Viagra, Cialis - No prescription needed!!!",
            "body": "BEST PRICES on all medications! Viagra $0.89/pill, Cialis from $1.19! No prescription needed! Fast discreet shipping worldwide! 100% satisfaction guaranteed or money back! Order online: http://online-pharma.top. Save 80% vs retail pharmacy prices! Buy now!",
            "sender": "pharmacy@online-pharma.top",
        },
        {
            "subject": "FOREX ROBOT earns $10,000/day - 99.9% accuracy!!!",
            "body": "Our AI trading bot generates MASSIVE profits automatically! Turn $500 into $50,000 in 30 days! No experience needed! 99.9% accuracy rate! Risk-free guarantee! Sign up: http://forex-profits.stream. Only 10 spots remaining! Wall Street insiders HATE us! ACT NOW!!!",
            "sender": "trading@forex-profits.stream",
        },
        {
            "subject": "CONGRATULATIONS!!! You won an iPhone 15 Pro MAX!!!",
            "body": "YOU ARE TODAY'S WINNER!!! You have been randomly selected to receive a FREE iPhone 15 Pro Max! Just pay $9.99 shipping and handling! Click to claim: http://free-iphone.click. Offer valid for next 2 hours only!!! Don't miss this amazing opportunity! Claim NOW!!!",
            "sender": "giveaway@free-iphone.click",
        },
    ]

    ham_templates = [
        {
            "subject": "Meeting tomorrow at 3pm - agenda attached",
            "body": "Hi team, Just a reminder about our weekly standup tomorrow at 3pm in Conference Room B. Agenda attached. Please review the Q3 progress report beforehand. Let me know if you can't make it. Thanks, Sarah",
            "sender": "sarah.johnson@company.com",
        },
        {
            "subject": "Re: Project milestone update",
            "body": "Hi Alex, Thanks for the update. The API integration looks solid. A few questions: 1) Have we load-tested the new endpoints? 2) What's the timeline for the database migration? I'll review the PR this afternoon. Cheers, David",
            "sender": "david.lee@corp.org",
        },
        {
            "subject": "Lunch on Friday?",
            "body": "Hey! Hope you're having a good week. Are you free for lunch Friday around 12:30? There's a new Thai place that opened on Main Street - reviews look great. Let me know! Mike",
            "sender": "mike.chen@gmail.com",
        },
        {
            "subject": "Your order #ORD-2024-5678 has shipped",
            "body": "Thank you for your order! Your package has been shipped via FedEx. Tracking number: 794607230258. Estimated delivery: December 18-20. You can track your shipment at fedex.com/tracking. Questions? Contact support@shop.com",
            "sender": "orders@trusted-shop.com",
        },
        {
            "subject": "Code review feedback - authentication module",
            "body": "Hi, I've finished reviewing the auth module PR. Overall the implementation is clean. Minor suggestions: 1) Add error handling for expired tokens in line 142, 2) The session timeout logic could use a test case, 3) README update would be helpful. LGTM after those changes. Thanks!",
            "sender": "priya.sharma@corp.org",
        },
        {
            "subject": "Invoice #2024-1234 for November consulting",
            "body": "Dear John, Please find attached invoice #2024-1234 for $2,500 for consulting services in November. Payment is due within 30 days. Bank transfer details on the invoice. Please confirm receipt. Thank you for your continued business. Best, Emma Wilson",
            "sender": "emma.wilson@consulting.net",
        },
        {
            "subject": "Re: Annual report draft - comments",
            "body": "Thanks for sharing the draft. I have a few comments: The revenue projection on page 12 needs to account for the Q4 expansion. Also, the market analysis section could use more recent data points. Can we schedule a review call before Friday? Let me know your availability. Best, Tom",
            "sender": "tom.martinez@business.io",
        },
        {
            "subject": "Reminder: Performance review next Thursday",
            "body": "Hi, This is a reminder that your annual performance review is scheduled for Thursday at 2pm. Please prepare a brief self-assessment covering your key achievements and goals. The template is on the HR portal. Let me know if you need to reschedule. HR Team",
            "sender": "hr@company.com",
        },
        {
            "subject": "GitHub: Pull request review needed - #456",
            "body": "PR #456 has been submitted by @dev-team and is ready for review. Changes: Updated data pipeline, fixed memory leak in batch processor, added unit tests (coverage: 87%). Reviewers: @alice @bob. Target merge: Friday. View PR at github.com/org/repo/pull/456",
            "sender": "noreply@github.com",
        },
        {
            "subject": "Fwd: Updated requirements from client - Project Phoenix",
            "body": "Forwarding the updated spec from the client. Key changes: 1) Dashboard must support real-time updates (WebSocket), 2) Export to Excel is now required, 3) Deadline moved to Jan 15. I've highlighted the delta from v1 in red. Let's discuss in tomorrow's sync. - Alex",
            "sender": "alex.kim@agency.com",
        },
    ]

    records = []
    n_spam = int(n_samples * 0.30)
    n_ham = n_samples - n_spam

    spam_extra_endings = [
        " ACT NOW - EXPIRES SOON!!!", " LIMITED TIME OFFER!!!",
        " DON'T MISS OUT!!!",  " RESPOND IMMEDIATELY!!!",
        " CLICK HERE NOW!!!", " CLAIM BEFORE IT'S GONE!!!",
    ]
    spam_extra_subjects = [
        "!!!{s}!!!", "URGENT: {s}", "ACT NOW: {s}", "[IMPORTANT] {s}",
    ]

    for i in range(n_spam):
        t = spam_templates[i % len(spam_templates)].copy()
        if rng.random() > 0.5:
            template = rng.choice(spam_extra_subjects)
            t["subject"] = template.format(s=t["subject"])
        if rng.random() > 0.6:
            t["body"] = t["body"] + rng.choice(spam_extra_endings)
        t["is_html"] = bool(rng.random() > 0.35)
        t["label"] = 1
        records.append(t)

    ham_extra_greetings = ["Hope this finds you well. ", "Quick note: ", "Just following up - "]
    for i in range(n_ham):
        t = ham_templates[i % len(ham_templates)].copy()
        if rng.random() > 0.7:
            t["body"] = rng.choice(ham_extra_greetings) + t["body"]
        if rng.random() > 0.8:
            t["subject"] = "Re: " + t["subject"]
        t["is_html"] = bool(rng.random() > 0.75)
        t["label"] = 0
        records.append(t)

    df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)
    log.info("Synthetic dataset: %d spam | %d ham | %d total", n_spam, n_ham, len(df))
    return df


# ── Model definitions ─────────────────────────────────────────────────────────

def _build_classifiers() -> dict:
    """Return candidate classifiers (all support predict_proba after wrapping)."""
    return {
        "logistic_regression": LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs"
        ),
        "complement_naive_bayes": ComplementNB(alpha=0.1),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42
        ),
        # SVC wrapped for calibrated probabilities
        "svm_calibrated": CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", max_iter=2000), cv=3
        ),
    }


# ── Training & evaluation ─────────────────────────────────────────────────────

def train_and_select_best(
    X_emails: list[dict],
    y: np.ndarray,
    cv_folds: int = 5,
) -> tuple[Pipeline, dict]:
    """
    Cross-validate every candidate classifier, pick the best by F1-macro,
    then refit the winner on the full training set.

    Returns: (fitted_pipeline, metrics_dict)
    """
    featurizer = EmailFeatureExtractor(max_tfidf_features=15_000, ngram_range=(1, 2))
    classifiers = _build_classifiers()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    log.info("Extracting features for cross-validation ...")
    X_feat = featurizer.fit_transform(X_emails, y)

    best_name, best_f1, all_scores = None, -1.0, {}

    for name, clf in classifiers.items():
        log.info("  Cross-validating: %s", name)
        scores = cross_validate(
            clf, X_feat, y,
            cv=cv,
            scoring={"f1": "f1", "roc_auc": "roc_auc", "precision": "precision", "recall": "recall"},
            n_jobs=-1,
        )
        mean_f1 = float(np.mean(scores["test_f1"]))
        mean_auc = float(np.mean(scores["test_roc_auc"]))
        all_scores[name] = {
            "f1": round(mean_f1, 4),
            "roc_auc": round(mean_auc, 4),
            "precision": round(float(np.mean(scores["test_precision"])), 4),
            "recall": round(float(np.mean(scores["test_recall"])), 4),
        }
        log.info(
            "    %s — F1=%.4f | AUC=%.4f",
            name, mean_f1, mean_auc,
        )
        if mean_f1 > best_f1:
            best_f1, best_name = mean_f1, name

    log.info("Best classifier: %s (F1=%.4f) — refitting on full data ...", best_name, best_f1)
    best_clf = classifiers[best_name]
    best_clf.fit(X_feat, y)

    full_pipeline = Pipeline([
        ("featurizer", featurizer),
        ("classifier", best_clf),
    ])

    # Final metrics on full training set
    y_pred = best_clf.predict(X_feat)
    y_prob = (
        best_clf.predict_proba(X_feat)[:, 1]
        if hasattr(best_clf, "predict_proba")
        else None
    )
    final_metrics = {
        "best_classifier": best_name,
        "cv_scores_all_classifiers": all_scores,
        "train_classification_report": classification_report(
            y, y_pred, target_names=["ham", "spam"], output_dict=True
        ),
        "train_roc_auc": round(float(roc_auc_score(y, y_prob)), 4) if y_prob is not None else None,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_training_samples": len(y),
        "n_spam": int(y.sum()),
        "n_ham": int((y == 0).sum()),
        "n_features_tfidf": best_clf.n_features_in_ - len(NUMERIC_FEATURE_NAMES)
        if hasattr(best_clf, "n_features_in_")
        else None,
        "n_numeric_features": len(NUMERIC_FEATURE_NAMES),
        "model_version": "1.0.0",
    }

    log.info("\n%s", classification_report(y, y_pred, target_names=["ham", "spam"]))
    return full_pipeline, final_metrics


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train email spam detector")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to CSV with columns: subject, body, sender, label (0/1)")
    parser.add_argument("--samples", type=int, default=2000,
                        help="Number of synthetic samples (ignored if --csv provided)")
    args = parser.parse_args()

    if args.csv:
        log.info("Loading dataset from %s ...", args.csv)
        df = pd.read_csv(args.csv)
        required = {"subject", "body", "label"}
        if not required.issubset(df.columns):
            log.error("CSV must have columns: %s", required)
            sys.exit(1)
        df["sender"] = df.get("sender", "")
        df["is_html"] = df.get("is_html", False)
    else:
        df = generate_synthetic_dataset(n_samples=args.samples)

    X_emails = df[["subject", "body", "sender", "is_html"]].to_dict("records")
    y = df["label"].to_numpy(dtype=int)

    pipeline, metrics = train_and_select_best(X_emails, y)

    joblib.dump(pipeline, MODEL_PATH, compress=3)
    METADATA_PATH.write_text(json.dumps(metrics, indent=2))

    log.info("Model saved  → %s", MODEL_PATH)
    log.info("Metadata saved → %s", METADATA_PATH)
    log.info("Best: %s | F1=%.4f | AUC=%s",
             metrics["best_classifier"],
             metrics["cv_scores_all_classifiers"][metrics["best_classifier"]]["f1"],
             metrics["train_roc_auc"])


if __name__ == "__main__":
    main()
