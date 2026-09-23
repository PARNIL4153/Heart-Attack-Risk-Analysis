"""
train_model.py
Heart Attack Risk Analysis & Prediction
Author: Parnil Kashyap

Loads data/data.csv, preprocesses it, trains a Random Forest classifier,
evaluates performance, and saves model.pkl + scaler.pkl to model/.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)

# ── 1. Load data ─────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join("data", "data.csv")
MODEL_DIR  = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, na_values=["?"])
df.columns = df.columns.str.strip()   # remove any trailing whitespace
print(f"[INFO] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ── 2. Preprocess ─────────────────────────────────────────────────────────────
# Target: 'num' (0 = no disease, 1-4 = disease) → binary
df["target"] = (df["num"] > 0).astype(int)
df.drop(columns=["num"], inplace=True)

# Coerce all columns (except target) to numeric where possible
for col in df.columns:
    if col != "target":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill numeric NaNs with column median (CoW-safe)
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())

# Drop any remaining rows with NaN (shouldn't be many)
df = df.dropna()
print(f"[INFO] After preprocessing: {df.shape[0]} rows")

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# Keep only features present in dataframe
FEATURES = [f for f in FEATURES if f in df.columns]

X = df[FEATURES]
y = df["target"]

print(f"[INFO] Class distribution:\n{y.value_counts()}")

# ── 3. Split ──────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Scale ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 5. Train ──────────────────────────────────────────────────────────────────
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
)
rf_model.fit(X_train_sc, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred = rf_model.predict(X_test_sc)
y_prob = rf_model.predict_proba(X_test_sc)[:, 1]

acc    = accuracy_score(y_test, y_pred)
auc    = roc_auc_score(y_test, y_prob)
cv_acc = cross_val_score(rf_model, scaler.transform(X), y, cv=5, scoring="accuracy").mean()

print(f"\n[RESULTS] Test Accuracy : {acc:.4f}")
print(f"[RESULTS] ROC-AUC       : {auc:.4f}")
print(f"[RESULTS] CV Accuracy   : {cv_acc:.4f}")
print(f"\n[RESULTS] Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"[RESULTS] Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# ── 7. Save artefacts ─────────────────────────────────────────────────────────
joblib.dump(rf_model, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(scaler,   os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(FEATURES, os.path.join(MODEL_DIR, "features.pkl"))

print(f"\n[INFO] Saved model.pkl, scaler.pkl, features.pkl to '{MODEL_DIR}/'")
