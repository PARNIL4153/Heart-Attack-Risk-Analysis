"""
frontend/ui.py
Heart Attack Risk Analysis & Prediction
Author: Parnil Kashyap

Streamlit Dashboard — two tabs:
  1. Interactive Risk Calculator  → calls Flask API at http://localhost:5000/predict
  2. Decision Dashboard           → EDA charts on the dataset
Run: streamlit run frontend/ui.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Attack Risk Analysis",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL   = "http://localhost:5000/predict"
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "data.csv")

# ── Load dataset (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, na_values=["?"])
    df.columns = df.columns.str.strip()
    df["target"] = (df["num"] > 0).astype(int)
    df.drop(columns=["num"], inplace=True)
    for col in df.columns:
        if col != "target":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    df = df.dropna()
    return df

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=80)
    st.title("Heart Attack Risk")
    st.markdown("**Author:** Parnil Kashyap")
    st.markdown("---")
    st.info(f"Dataset: **{len(df)} patients** loaded")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🩺 Risk Calculator", "📊 Decision Dashboard"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Interactive Risk Calculator
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🩺 Patient Risk Calculator")
    st.markdown("Enter patient clinical parameters below and click **Predict Risk**.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age      = st.slider("Age", 20, 80, 52)
        sex      = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp       = st.selectbox("Chest Pain Type (cp)",
                                [0, 1, 2, 3],
                                format_func=lambda x: {0:"Typical Angina",1:"Atypical Angina",
                                                        2:"Non-anginal",3:"Asymptomatic"}[x])
        trestbps = st.number_input("Resting BP (trestbps)", 80, 200, 125)
        chol     = st.number_input("Serum Cholesterol (chol)", 100, 600, 212)

    with col2:
        fbs      = st.selectbox("Fasting Blood Sugar > 120 (fbs)", [0, 1],
                                format_func=lambda x: "No" if x == 0 else "Yes")
        restecg  = st.selectbox("Resting ECG (restecg)",
                                [0, 1, 2],
                                format_func=lambda x: {0:"Normal",1:"ST-T abnormality",
                                                        2:"LV hypertrophy"}[x])
        thalach  = st.slider("Max Heart Rate (thalach)", 60, 220, 168)
        exang    = st.selectbox("Exercise Induced Angina (exang)", [0, 1],
                                format_func=lambda x: "No" if x == 0 else "Yes")

    with col3:
        oldpeak  = st.number_input("ST Depression (oldpeak)", 0.0, 6.2, 1.0, step=0.1)
        slope    = st.selectbox("Slope of ST segment",
                                [0, 1, 2],
                                format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])
        ca       = st.selectbox("Fluoroscopy vessels (ca)", [0, 1, 2, 3])
        thal     = st.selectbox("Thalassemia (thal)",
                                [0, 1, 2, 3],
                                format_func=lambda x: {0:"Unknown",1:"Normal",
                                                        2:"Fixed Defect",3:"Reversible Defect"}[x])

    st.markdown("---")
    if st.button("🔍 Predict Risk", use_container_width=True, type="primary"):
        payload = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
        }
        try:
            resp = requests.post(API_URL, json=payload, timeout=5)
            resp.raise_for_status()
            result = resp.json()

            prob  = result["risk_probability"]
            level = result["risk_level"]
            msg   = result["message"]

            colour = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}.get(level, "⚪")
            st.success(f"{colour} **Risk Level: {level}**  |  Probability: **{prob*100:.1f}%**")
            st.info(f"**Diagnosis:** {msg}")

            # Gauge bar
            st.markdown("#### Risk Probability Gauge")
            fig, ax = plt.subplots(figsize=(6, 0.6))
            ax.barh(0, prob, color="#e74c3c", height=0.4)
            ax.barh(0, 1 - prob, left=prob, color="#ecf0f1", height=0.4)
            ax.set_xlim(0, 1); ax.axis("off")
            ax.text(prob / 2, 0, f"{prob*100:.1f}%", va="center", ha="center",
                    color="white", fontweight="bold")
            st.pyplot(fig, use_container_width=True)

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to Flask API. Make sure `backend/app.py` is running on port 5000.")
        except Exception as exc:
            st.error(f"❌ Error: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Decision Dashboard (EDA)
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📊 Decision Dashboard — Exploratory Data Analysis")

    # Filters
    with st.expander("🔧 Demographic Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        age_range = fc1.slider("Age Range", int(df.age.min()), int(df.age.max()),
                               (int(df.age.min()), int(df.age.max())))
        sex_filter = fc2.multiselect("Sex", [0, 1], default=[0, 1],
                                     format_func=lambda x: "Female" if x == 0 else "Male")
        cp_filter  = fc3.multiselect("Chest Pain Type", [0, 1, 2, 3], default=[0, 1, 2, 3])

    mask = (
        df.age.between(*age_range) &
        df.sex.isin(sex_filter) &
        df.cp.isin(cp_filter)
    )
    fdf = df[mask]
    st.markdown(f"**Filtered patients: {len(fdf)}**")

    r1c1, r1c2 = st.columns(2)

    # Chart 1 — Target distribution
    with r1c1:
        st.subheader("Heart Disease Distribution")
        fig, ax = plt.subplots()
        counts = fdf["target"].value_counts()
        ax.pie(counts, labels=["No Disease", "Disease"],
               colors=["#2ecc71", "#e74c3c"], autopct="%1.1f%%", startangle=90)
        ax.set_title("Target Class Split")
        st.pyplot(fig)

    # Chart 2 — Age histogram by target
    with r1c2:
        st.subheader("Age Distribution by Risk")
        fig, ax = plt.subplots()
        for t, colour, label in [(0, "#2ecc71", "No Disease"), (1, "#e74c3c", "Disease")]:
            ax.hist(fdf[fdf.target == t]["age"], bins=15, alpha=0.7, color=colour, label=label)
        ax.set_xlabel("Age"); ax.set_ylabel("Count")
        ax.legend(); ax.set_title("Age vs Heart Disease")
        st.pyplot(fig)

    r2c1, r2c2 = st.columns(2)

    # Chart 3 — Cholesterol boxplot
    with r2c1:
        st.subheader("Cholesterol by Risk Class")
        fig, ax = plt.subplots()
        fdf.boxplot(column="chol", by="target", ax=ax,
                    boxprops=dict(color="#3498db"),
                    medianprops=dict(color="#e74c3c"))
        ax.set_xlabel("Target (0=No Disease, 1=Disease)")
        ax.set_ylabel("Cholesterol (mg/dl)")
        ax.set_title(""); plt.suptitle("")
        st.pyplot(fig)

    # Chart 4 — Correlation heatmap
    with r2c2:
        st.subheader("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(7, 5))
        corr = fdf.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".1f", cmap="coolwarm",
                    linewidths=0.5, ax=ax, annot_kws={"size": 7})
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)

    # Chart 5 — Max Heart Rate vs Age scatter
    st.subheader("Max Heart Rate vs Age")
    fig, ax = plt.subplots(figsize=(8, 4))
    colours = fdf["target"].map({0: "#2ecc71", 1: "#e74c3c"})
    ax.scatter(fdf["age"], fdf["thalach"], c=colours, alpha=0.6)
    ax.set_xlabel("Age"); ax.set_ylabel("Max Heart Rate (thalach)")
    ax.set_title("Max Heart Rate vs Age (Green = No Disease, Red = Disease)")
    st.pyplot(fig)

    # Raw data
    with st.expander("🗂️ View Raw Data"):
        st.dataframe(fdf.reset_index(drop=True), use_container_width=True)
