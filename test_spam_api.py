"""
Quick test script for the spam detection API.
Run the API first: python spam_api.py
Then: python test_spam_api.py
"""

import json
import requests

BASE = "http://localhost:8001"

# ── 1. Health check ───────────────────────────────────────────────────────────

print("=" * 60)
print("1. HEALTH CHECK")
r = requests.get(f"{BASE}/health")
print(json.dumps(r.json(), indent=2))

# ── 2. Model info ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. MODEL INFO")
r = requests.get(f"{BASE}/model/info")
info = r.json()
print(f"  Best classifier : {info['best_classifier']}")
print(f"  Training samples: {info['n_training_samples']} ({info['n_spam']} spam / {info['n_ham']} ham)")
print(f"  Model version   : {info['model_version']}")
print(f"  Trained at      : {info['trained_at']}")

# ── 3. Single predictions ─────────────────────────────────────────────────────

test_emails = [
    {
        "label": "SPAM",
        "payload": {
            "subject": "URGENT!!! You have WON $1,000,000 - CLAIM NOW!!!",
            "body": "Congratulations! Your email was selected from 50 million addresses. You have WON $2,500,000 USD. Click: http://prize-claim.xyz. Send your bank details NOW! Expires in 24 hours!!!",
            "sender": "no-reply@prize-claim.xyz",
            "is_html": False,
        },
    },
    {
        "label": "SPAM",
        "payload": {
            "subject": "Your PayPal account SUSPENDED - Verify immediately",
            "body": "Dear Valued Customer, Your account has been suspended due to suspicious activity. Verify your account immediately to avoid permanent closure. Click here: http://paypal-secure.click/verify. Enter your username, password and credit card details.",
            "sender": "security@paypal-secure.click",
            "is_html": True,
        },
    },
    {
        "label": "HAM",
        "payload": {
            "subject": "Re: Team standup notes - Thursday",
            "body": "Hi everyone, attaching the meeting notes from today. Action items: 1) John to complete API docs by Friday, 2) Sarah to review UI mockups. Next meeting: same time next week. Thanks, Alex",
            "sender": "alex.kim@company.com",
            "is_html": False,
        },
    },
    {
        "label": "HAM",
        "payload": {
            "subject": "Invoice #2024-5678 for consulting services",
            "body": "Dear John, please find attached invoice for consulting services rendered in November. Total: $3,200. Payment due within 30 days. Let me know if you have any questions. Best regards, Emma",
            "sender": "emma@consulting.net",
            "is_html": False,
        },
    },
    {
        "label": "SPAM",
        "payload": {
            "subject": "MAKE $5000/week working from home - No experience!",
            "body": "Earn $500-$3000 per DAY working from home! Our proven system makes money 24/7! No experience required! Click: http://easy-income.loan. FREE training! 100% guaranteed income! LIMITED spots! ACT NOW!!!",
            "sender": "support@easy-income.loan",
            "is_html": False,
        },
    },
]

print("\n" + "=" * 60)
print("3. SINGLE EMAIL PREDICTIONS")
print("-" * 60)

passed = 0
for tc in test_emails:
    r = requests.post(f"{BASE}/predict", json=tc["payload"])
    result = r.json()
    prediction = "SPAM" if result["is_spam"] else "HAM"
    ok = prediction == tc["label"]
    passed += ok
    status = "PASS" if ok else "FAIL"

    print(f"[{status}] Expected={tc['label']:<4}  Got={prediction:<4}  "
          f"spam_prob={result['spam_probability']:.4f}  "
          f"risk={result['risk_level']:<8}  "
          f"conf={result['confidence']}")
    if result["spam_signals"]:
        print(f"       signals: {result['spam_signals']}")
    print()

print(f"Result: {passed}/{len(test_emails)} correct")

# ── 4. Batch prediction ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("4. BATCH PREDICTION (5 emails in one call)")
batch_payload = {"emails": [tc["payload"] for tc in test_emails]}
r = requests.post(f"{BASE}/predict/batch", json=batch_payload)
batch = r.json()
print(f"  Total       : {batch['total']}")
print(f"  Spam found  : {batch['spam_count']}")
print(f"  Ham found   : {batch['ham_count']}")
print(f"  Elapsed ms  : {batch['inference_time_ms']}")

# ── 5. Edge cases ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("5. EDGE CASES")

edge_cases = [
    ("empty email",     {"subject": "", "body": "", "sender": ""}),
    ("no sender",       {"subject": "Hello", "body": "How are you?", "sender": ""}),
    ("html spam",       {"subject": "FREE OFFER!!!", "body": "<html><body><b>CLICK HERE NOW!!!</b><a href='http://spam.xyz'>WIN $$$</a></body></html>", "sender": "spam@offer.top", "is_html": True}),
]

for name, payload in edge_cases:
    r = requests.post(f"{BASE}/predict", json=payload)
    result = r.json()
    print(f"  {name:<18} = {'SPAM' if result['is_spam'] else 'HAM':4s}  "
          f"spam_prob={result['spam_probability']:.4f}")
