"""
backend/app.py
Heart Attack Risk Analysis & Prediction
Author: Parnil Kashyap

Flask REST API — POST /predict
Accepts patient clinical metrics in JSON, returns risk probability and level.
Run: python backend/app.py
"""

import os
import sys
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Resolve model paths relative to project root ─────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "model")

app = Flask(__name__)
CORS(app)

# ── Load artefacts once at startup ────────────────────────────────────────────
def load_artefacts():
    model_path    = os.path.join(MODEL_DIR, "model.pkl")
    scaler_path   = os.path.join(MODEL_DIR, "scaler.pkl")
    features_path = os.path.join(MODEL_DIR, "features.pkl")
    if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
        print("[ERROR] Model artefacts not found. Run train_model.py first.")
        sys.exit(1)
    return (
        joblib.load(model_path),
        joblib.load(scaler_path),
        joblib.load(features_path),
    )

model, scaler, FEATURES = load_artefacts()
print(f"[INFO] Model loaded. Features: {FEATURES}")


# ── Helper: map probability to risk level ─────────────────────────────────────
def risk_level(prob: float) -> str:
    if prob < 0.30:
        return "Low"
    elif prob < 0.60:
        return "Moderate"
    else:
        return "High"


# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "RandomForest"}), 200


# ── Predict endpoint ──────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected JSON body (all numeric):
    {
        "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
        "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
        "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
    }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    # Build feature vector in training order
    try:
        input_values = [float(data.get(f, 0)) for f in FEATURES]
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 422

    X = np.array(input_values).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0][1])
    level = risk_level(probability)

    response = {
        "prediction": prediction,
        "risk_probability": round(probability, 4),
        "risk_level": level,
        "message": (
            "Heart disease risk detected." if prediction == 1
            else "No significant heart disease risk detected."
        ),
    }
    return jsonify(response), 200


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] Starting Flask API on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
