from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import logging
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer  # Optional placeholder

# Download VADER lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Initialize sentiment analyzer
try:
    sia = SentimentIntensityAnalyzer()
except Exception as e:
    logging.error(f"Failed to initialize VADER SentimentIntensityAnalyzer: {e}")
    raise RuntimeError("Sentiment Analyzer could not be initialized. Check nltk resources.")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema for sentiment prediction
class PredictRequest(BaseModel):
    text: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {"message": "Moodify backend is up and running with VADER 🔥"}

@app.post("/predict")
async def analyze_sentiment(payload: PredictRequest):
    try:
        text = payload.text.strip()
        logging.info(f"Incoming text: {text}")

        if not text:
            raise HTTPException(status_code=400, detail="Text input is empty.")

        # OPTIONAL: TF-IDF placeholder (if needed later)
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform([text])
        logging.info("TF-IDF features extracted (debug placeholder)")

        # Perform sentiment analysis
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

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Unexpected error in /predict: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error. Please try again later."}
        )
