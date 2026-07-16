from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import re
import os

app = FastAPI(title="Sentiment Analysis API")

# Allow frontend (running on a different port/origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model + vectorizer (created in Phase 8)
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)


class TextRequest(BaseModel):
    text: str


def clean_text(text: str) -> str:
    # Must match the exact cleaning steps used in training (Phase 3),
    # otherwise the vectorizer will see different-looking text.
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # remove URLs
    text = re.sub(r"@\w+", "", text)              # remove @mentions
    text = re.sub(r"[^a-z\s]", "", text)          # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()      # remove extra spaces
    return text


@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API is running"}


@app.post("/predict")
def predict_sentiment(request: TextRequest):
    if not request.text.strip():
        return {"text": request.text, "prediction": "Invalid input", "error": "Empty text"}

    cleaned = clean_text(request.text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0].lower()  # "Positive" -> "positive" etc.

    # Confidence score (optional feature from Phase 10)
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        confidence = round(float(max(proba)) * 100, 2)

    return {
        "text": request.text,
        "prediction": prediction,
        "confidence": confidence,
    }
