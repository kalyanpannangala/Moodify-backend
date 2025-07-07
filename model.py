import joblib
import os

# Load the trained model once
MODEL_PATH = os.path.join("model", "moodify_model.pkl")
model = joblib.load(MODEL_PATH)

def predict_mood(text: str) -> str:
    prediction = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    confidence = round(max(proba), 3)
    return {
        "mood": prediction,
        "confidence": confidence,
        "all_probs": dict(zip(model.classes_, map(float, proba)))
    }
