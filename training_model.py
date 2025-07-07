# train_model.py

import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Define label mapping
label_map = {
    "Negative": 0,
    "Mildly Negative": 1,
    "Neutral": 2,
    "Mildly Positive": 3,
    "Positive": 4
}

# Load dataset
df = pd.read_csv("dataset.csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Map labels to integers
df["label"] = df["label"].map(label_map)

# Features and labels
X = df["text"]
y = df["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Save model to model.pkl
joblib.dump(model, "model.pkl")

# Save label map to model_labels.json
with open("model_labels.json", "w") as f:
    json.dump({v: k for k, v in label_map.items()}, f)

print("✅ Model saved as model.pkl")
print("✅ Labels saved as model_labels.json")
