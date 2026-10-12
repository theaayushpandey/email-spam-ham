"""Spam checker as a web service and a WhatsApp bot.

Run:  python app.py

POST /predict   JSON {"text": "..."}             -> JSON verdict (for testing and other apps)
POST /whatsapp  Twilio's WhatsApp webhook (form) -> TwiML reply that Twilio sends back to the user
"""
import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import joblib
from flask import Flask, Response, jsonify, request

from text_utils import transform_text

HERE = Path(__file__).parent
FEEDBACK_FILE = HERE / "feedback.csv"
MAX_LENGTH = 2000

# Load the model once at startup, not on every message
saved = joblib.load(HERE / "model.joblib")
vectorizer, model = saved["vectorizer"], saved["model"]

app = Flask(__name__)

# The last message each sender asked about, so "wrong" knows what to correct.
# Kept in memory only: it is lost on restart and never written to disk.
last_check = {}

HELP_TEXT = (
    "Hi! Forward me any SMS or message you are unsure about and I will tell you "
    "if it looks like spam.\n\nIf I get one wrong, reply WRONG and I will learn from it."
)


def classify(text):
    """Return a verdict for one message.

    spam_probability is how sure the model is about THIS message. It is not the
    model's accuracy, which is measured once on the test set in train.py.
    """
    vector = vectorizer.transform([transform_text(text)])
    if vector.nnz == 0:
        # None of the words are in the vocabulary, so the model would only be guessing
        return {"label": "unknown", "spam_probability": None}

    probability = float(model.predict_proba(vector)[0][1])
    if probability >= 0.7:
        label = "spam"
    elif probability <= 0.3:
        label = "ham"
    else:
        label = "unsure"
    return {"label": label, "spam_probability": round(probability, 3)}


def percent(probability):
    # Naive Bayes is overconfident and often outputs exactly 0 or 1, so never show 0% or 100%
    return f"{min(max(probability, 0.01), 0.99):.0%}"


def reply_text(verdict):
    label, probability = verdict["label"], verdict["spam_probability"]
    if label == "unknown":
        return (
            "I can't judge this one: it has no words I learned from. "
            "Be careful with links and never share an OTP."
        )
    if label == "spam":
        text = f"Likely SPAM ({percent(probability)} sure). Don't click links or share an OTP."
    elif label == "ham":
        text = f"Looks safe (spam chance {percent(probability)})."
    else:
        text = f"Not sure about this one (spam chance {percent(probability)}). Treat it with care."
    return text + "\n\nThis is an automated guess. Reply WRONG if I got it wrong."


def save_feedback(text, predicted):
    """Store a misclassified message for the next retraining. No phone number is saved."""
    is_new = not FEEDBACK_FILE.exists()
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["time", "text", "predicted", "correct_label"])
        correct = "ham" if predicted == "spam" else "spam"
        writer.writerow([datetime.now(timezone.utc).isoformat(), text, predicted, correct])


@app.post("/predict")
def predict():
    text = (request.get_json(silent=True) or {}).get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Send JSON like {\"text\": \"your message\"}"}), 400
    if len(text) > MAX_LENGTH:
        return jsonify({"error": f"Message is longer than {MAX_LENGTH} characters"}), 413
    return jsonify(classify(text))


@app.post("/whatsapp")
def whatsapp():
    # Twilio sends the message as form fields: Body is the text, From is the sender
    body = request.form.get("Body", "").strip()
    sender = request.form.get("From", "")
    command = body.lower()

    if not body or command in {"hi", "hello", "hey", "help", "start"}:
        answer = HELP_TEXT
    elif command == "wrong":
        previous = last_check.pop(sender, None)
        if previous:
            save_feedback(*previous)
            answer = "Thanks, I've noted that I got it wrong. It will be used to retrain me."
        else:
            answer = "Send me a message to check first, then reply WRONG if I misjudge it."
    elif len(body) > MAX_LENGTH:
        answer = "That message is too long for me. Send the main part of it."
    else:
        verdict = classify(body)
        if verdict["label"] in {"spam", "ham"}:
            last_check[sender] = (body, verdict["label"])
        answer = reply_text(verdict)

    # TwiML: Twilio reads this XML and sends the message back on WhatsApp
    twiml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{escape(answer)}</Message></Response>'
    return Response(twiml, mimetype="application/xml")


@app.get("/")
def health():
    return jsonify({"ok": True, "model": saved.get("name")})


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5001)), debug=os.environ.get("FLASK_DEBUG") == "1")
