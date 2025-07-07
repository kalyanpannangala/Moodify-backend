from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load model and labels
try:
    model = joblib.load("model.pkl")
    with open("model_labels.json", "r") as f:
        label_map = json.load(f)
    # Convert keys to int (JSON saves them as strings)
    label_map = {int(k): v for k, v in label_map.items()}
    logging.info("✅ ML model and label map loaded.")
except Exception as e:
    logging.error(f"❌ Failed to load model or labels: {e}")
    raise RuntimeError("Model or labels could not be loaded.")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can lock this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class PredictRequest(BaseModel):
    text: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {"message": "Moodify backend is live with ML model 🧠🚀"}

@app.post("/predict")
async def predict_mood(payload: PredictRequest):
    try:
        text = payload.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text input is empty.")

        prediction = model.predict([text])[0]
        mood = label_map[prediction]

        return {
            "text": text,
            "mood": mood,
            "prediction": int(prediction)
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"❌ Unexpected error in /predict: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error. Please try again later."}
        )
