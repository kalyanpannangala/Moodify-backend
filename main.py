from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os
import logging
from langdetect import detect, LangDetectException

# Only download if not available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Setup
app = FastAPI()
sia = SentimentIntensityAnalyzer()

# Logging config
logging.basicConfig(level=logging.INFO)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model
class TextInput(BaseModel):
    text: str

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Root message
@app.get("/")
def read_root():
    return {"message": "Moodify backend is up and running with VADER 🔥"}

# Sentiment prediction
@app.post("/predict")
def analyze_sentiment(data: TextInput):
    text = data.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text input is empty.")

    # Language detection
    try:
        lang = detect(text)
        if lang != "en":
            raise HTTPException(status_code=422, detail="Please input English text only.")
    except LangDetectException:
        raise HTTPException(status_code=422, detail="Could not detect language.")

    # Analyze sentiment
    logging.info(f"Analyzing sentiment for: {text}")
    scores = sia.polarity_scores(text)
    compound = scores['compound']

    if compound < -0.5:
        mood = "Negative"
    elif -0.5 <= compound < -0.2:
        mood = "Mildly Negative"
    elif -0.2 <= compound <= 0.2:
        mood = "Neutral"
    elif 0.2 < compound <= 0.5:
        mood = "Mildly Positive"
    else:
        mood = "Positive"

    return {
        "text": text,
        "mood": mood,
        "score": compound,
        "raw_scores": scores
    }
