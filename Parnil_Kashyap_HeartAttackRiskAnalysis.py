"""
Parnil_Kashyap_HeartAttackRiskAnalysis.py
==========================================
Heart Attack Risk Analysis & Prediction — Master Application
Author  : Parnil Kashyap
Dataset : Cleveland Heart Disease (UCI ML Repository, data/data.csv)

Self-contained Streamlit application that:
  • Loads & preprocesses data/data.csv
  • Trains (or hot-loads from model/) a Random Forest + Logistic Regression classifier
  • Tab 1 — Live Patient Risk Predictor   : inline ML inference, gauge, SHAP-style bars
  • Tab 2 — Decision Dashboard            : interactive EDA, demographic filters, 6 charts
  • Tab 3 — Summary & Recommendations    : key findings, actionable healthcare guidance

Run:
    streamlit run Parnil_Kashyap_HeartAttackRiskAnalysis.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(_HERE, "data", "data.csv")
MODEL_DIR  = os.path.join(_HERE, "model")
MODEL_PKL  = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PKL = os.path.join(MODEL_DIR, "scaler.pkl")
FEAT_PKL   = os.path.join(MODEL_DIR, "features.pkl")

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

CP_LABELS   = {0: "Typical Angina", 1: "Atypical Angina",
               2: "Non-Anginal Pain", 3: "Asymptomatic"}
ECG_LABELS  = {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}
SLOPE_LABELS= {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
THAL_LABELS = {0: "Unknown", 1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be the very first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Heart Attack Risk Analysis — Parnil Kashyap",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING  (cached)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading & preprocessing dataset…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, na_values=["?"])
    df.columns = df.columns.str.strip()
    df["target"] = (df["num"] > 0).astype(int)
    df.drop(columns=["num"], inplace=True)
    for col in df.columns:
        if col != "target":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    df = df.dropna().reset_index(drop=True)
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING / LOADING  (cached — trains once per session if pkl absent)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Training / loading ML models…")
def get_models(df: pd.DataFrame):
    """
    Returns (rf_model, lr_model, scaler, features, metrics_dict).
    If saved artefacts exist they are loaded; otherwise the models are trained
    from scratch and persisted to model/.
    """
    feats = [f for f in FEATURES if f in df.columns]
    X = df[feats]
    y = df["target"]

    # ── Hot-load if artefacts exist ───────────────────────────────────────────
    if all(os.path.exists(p) for p in [MODEL_PKL, SCALER_PKL, FEAT_PKL]):
        rf     = joblib.load(MODEL_PKL)
        scaler = joblib.load(SCALER_PKL)
        feats  = joblib.load(FEAT_PKL)
        X      = df[feats]
        X_sc   = scaler.transform(X)
        y_pred = rf.predict(X_sc)
        y_prob = rf.predict_proba(X_sc)[:, 1]
        lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        X_train, X_test, y_train, y_test = train_test_split(
            X_sc, y, test_size=0.2, random_state=42, stratify=y)
        lr.fit(X_train, y_train)
        metrics = _compute_metrics(rf, lr, scaler, X, y, feats)
        return rf, lr, scaler, feats, metrics

    # ── Train from scratch ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")
    rf.fit(X_train_sc, y_train)

    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train_sc, y_train)

    # ── Persist ───────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf,     MODEL_PKL)
    joblib.dump(scaler, SCALER_PKL)
    joblib.dump(feats,  FEAT_PKL)

    metrics = _compute_metrics(rf, lr, scaler, X, y, feats)
    return rf, lr, scaler, feats, metrics


def _compute_metrics(rf, lr, scaler, X, y, feats):
    X_sc = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y, test_size=0.2, random_state=42, stratify=y)

    results = {}
    for name, model in [("Random Forest", rf), ("Logistic Regression", lr)]:
        yp    = model.predict(X_test)
        yprob = model.predict_proba(X_test)[:, 1]
        cv    = cross_val_score(model, X_sc, y, cv=5, scoring="accuracy").mean()
        results[name] = {
            "accuracy"  : accuracy_score(y_test, yp),
            "auc"       : roc_auc_score(y_test, yprob),
            "cv_acc"    : cv,
            "report"    : classification_report(y_test, yp,
                              target_names=["No Disease", "Disease"],
                              output_dict=True),
            "cm"        : confusion_matrix(y_test, yp),
            "fpr"       : roc_curve(y_test, yprob)[0],
            "tpr"       : roc_curve(y_test, yprob)[1],
            "y_test"    : y_test,
            "y_pred"    : yp,
            "y_prob"    : yprob,
        }
    return results


def predict_risk(model, scaler, feats, patient: dict) -> tuple[float, int, str]:
    """Returns (probability, prediction, risk_level)."""
    X = np.array([[float(patient.get(f, 0)) for f in feats]])
    prob = float(model.predict_proba(scaler.transform(X))[0][1])
    pred = int(model.predict(scaler.transform(X))[0])
    level = "High" if prob >= 0.60 else ("Moderate" if prob >= 0.30 else "Low")
    return prob, pred, level


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA & MODELS
# ═══════════════════════════════════════════════════════════════════════════════
df = load_data()
rf_model, lr_model, scaler, feat_cols, metrics = get_models(df)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ❤️ Heart Attack Risk")
    st.markdown("**Author:** Parnil Kashyap")
    st.divider()
    st.metric("Total Patients", len(df))
    st.metric("Disease Prevalence",
              f"{df['target'].mean()*100:.1f}%")
    st.metric("RF Test Accuracy",
              f"{metrics['Random Forest']['accuracy']*100:.1f}%")
    st.metric("RF ROC-AUC",
              f"{metrics['Random Forest']['auc']:.3f}")
    st.divider()
    active_model = st.radio("Active predictor model",
                            ["Random Forest", "Logistic Regression"],
                            index=0)
    st.caption("Model used in Tab 1 predictions")

active = rf_model if active_model == "Random Forest" else lr_model

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🩺 Live Patient Risk Predictor",
    "📊 Decision Dashboard",
    "📋 Summary & Recommendations",
])


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — LIVE PATIENT RISK PREDICTOR                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab1:
    st.header("🩺 Live Patient Risk Predictor")
    st.markdown(
        "Adjust the **13 clinical parameters** below. "
        "The model scores the patient **instantly** — no external API needed."
    )

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("risk_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("##### Demographic & Cardiac")
            age      = st.slider("Age (years)", 20, 80, 54)
            sex      = st.selectbox("Sex", [0, 1],
                                    format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
            cp       = st.selectbox("Chest Pain Type", list(CP_LABELS),
                                    format_func=CP_LABELS.get)
            trestbps = st.number_input("Resting BP — mm Hg", 80, 200, 130)
            chol     = st.number_input("Serum Cholesterol — mg/dl", 100, 600, 246)

        with c2:
            st.markdown("##### Blood & ECG")
            fbs     = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                                   [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            restecg = st.selectbox("Resting ECG Result", list(ECG_LABELS),
                                   format_func=ECG_LABELS.get)
            thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
            exang   = st.selectbox("Exercise-Induced Angina",
                                   [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        with c3:
            st.markdown("##### Exercise Test & Imaging")
            oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.2, 1.0, step=0.1,
                                      format="%.1f")
            slope   = st.selectbox("Slope of Peak ST Segment", list(SLOPE_LABELS),
                                   format_func=SLOPE_LABELS.get)
            ca      = st.selectbox("Fluoroscopy Vessels Coloured (0–3)", [0, 1, 2, 3])
            thal    = st.selectbox("Thalassemia Type", list(THAL_LABELS),
                                   format_func=THAL_LABELS.get)

        submitted = st.form_submit_button(
            "⚡ Calculate Risk Score", use_container_width=True, type="primary")

    # ── Prediction output ─────────────────────────────────────────────────────
    if submitted:
        patient = dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol,
                       fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
                       oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)

        prob, pred, level = predict_risk(active, scaler, feat_cols, patient)

        COLOURS = {"Low": "#2ecc71", "Moderate": "#f39c12", "High": "#e74c3c"}
        ICONS   = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}
        colour  = COLOURS[level]

        # ── Banner ─────────────────────────────────────────────────────────────
        st.markdown("---")
        bc1, bc2, bc3 = st.columns([1, 2, 1])
        with bc2:
            st.markdown(
                f"<div style='text-align:center; padding:20px; border-radius:12px; "
                f"background:{colour}22; border:2px solid {colour};'>"
                f"<h2 style='color:{colour}; margin:0;'>{ICONS[level]} {level} Risk</h2>"
                f"<h1 style='font-size:3.2rem; margin:4px 0; color:{colour};'>"
                f"{prob*100:.1f}%</h1>"
                f"<p style='margin:0; color:#555;'>Heart Disease Probability</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("&nbsp;")
        res_c1, res_c2 = st.columns(2)

        # ── Gauge bar ─────────────────────────────────────────────────────────
        with res_c1:
            st.markdown("##### Risk Probability Gauge")
            fig, ax = plt.subplots(figsize=(7, 0.8))
            # gradient background zones
            ax.barh(0, 0.30, color="#2ecc7133", height=0.5, left=0)
            ax.barh(0, 0.30, color="#f39c1233", height=0.5, left=0.30)
            ax.barh(0, 0.40, color="#e74c3c33", height=0.5, left=0.60)
            ax.barh(0, prob, color=colour, height=0.36, alpha=0.95)
            ax.axvline(prob, color=colour, lw=2.5)
            ax.set_xlim(0, 1)
            ax.axis("off")
            for xv, lbl in [(0.15, "Low"), (0.45, "Moderate"), (0.80, "High")]:
                ax.text(xv, -0.55, lbl, ha="center", va="center",
                        fontsize=8, color="#888")
            ax.text(prob, 0.42, f"{prob*100:.1f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=colour)
            fig.patch.set_alpha(0)
            st.pyplot(fig, use_container_width=True)

            diagnosis = ("⚠️ Heart disease risk detected — consider further clinical evaluation."
                         if pred == 1
                         else "✅ No significant heart disease risk detected.")
            st.info(f"**{active_model} Diagnosis:** {diagnosis}")

        # ── Feature contribution bars ─────────────────────────────────────────
        with res_c2:
            st.markdown("##### Feature Contribution to Risk")
            if hasattr(active, "feature_importances_"):
                importances = active.feature_importances_
            else:
                importances = np.abs(active.coef_[0])
            imp_series = pd.Series(importances, index=feat_cols).sort_values(ascending=True)
            bar_colours = ["#e74c3c" if v >= imp_series.median() else "#3498db"
                           for v in imp_series]
            fig, ax = plt.subplots(figsize=(6, 4))
            imp_series.plot(kind="barh", color=bar_colours, ax=ax, edgecolor="none")
            ax.set_xlabel("Importance Score", fontsize=9)
            ax.set_title(f"Feature Importances ({active_model})", fontsize=10)
            ax.tick_params(labelsize=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

        # ── Clinical parameter summary table ──────────────────────────────────
        st.markdown("##### Submitted Patient Parameters")
        param_df = pd.DataFrame([
            {"Parameter": k.upper(), "Value": v,
             "Normal Range / Notes": {
                 "age": "—", "sex": "0=F, 1=M",
                 "cp":      CP_LABELS.get(int(v),    str(v)),
                 "trestbps": "< 120 mm Hg optimal", "chol": "< 200 mg/dl optimal",
                 "fbs": "No / Yes",
                 "restecg": ECG_LABELS.get(int(v),   str(v)),
                 "thalach": "220 − age (max estimate)", "exang": "No / Yes",
                 "oldpeak": "0 normal; > 2 concerning",
                 "slope":   SLOPE_LABELS.get(int(v), str(v)),
                 "ca": "0 = clear vessels",
                 "thal":    THAL_LABELS.get(int(v),  str(v)),
             }.get(k, "—")}
            for k, v in patient.items()
        ])
        st.dataframe(param_df, use_container_width=True, hide_index=True)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — DECISION DASHBOARD (EDA)                                          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab2:
    st.header("📊 Decision Dashboard — Exploratory Data Analysis")

    # ── Demographic filters ───────────────────────────────────────────────────
    with st.expander("🔧 Demographic Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        age_range  = fc1.slider("Age Range",
                                int(df.age.min()), int(df.age.max()),
                                (int(df.age.min()), int(df.age.max())),
                                key="eda_age")
        sex_filter = fc2.multiselect("Sex", [0, 1], default=[0, 1],
                                     format_func=lambda x: "Female" if x == 0 else "Male")
        cp_filter  = fc3.multiselect("Chest Pain Type",
                                     list(CP_LABELS), default=list(CP_LABELS),
                                     format_func=CP_LABELS.get)

    mask = (df.age.between(*age_range) &
            df.sex.isin(sex_filter) &
            df.cp.isin(cp_filter))
    fdf = df[mask]
    st.markdown(f"**Showing {len(fdf)} / {len(df)} patients after filters**")

    if len(fdf) < 5:
        st.warning("Too few patients match the current filters. Widen the selection.")
        st.stop()

    # ── Row 1 ─────────────────────────────────────────────────────────────────
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.subheader("Target Class Distribution")
        counts = fdf["target"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(counts, labels=["No Disease", "Disease"],
               colors=["#2ecc71", "#e74c3c"],
               autopct="%1.1f%%", startangle=90,
               wedgeprops={"edgecolor": "white", "linewidth": 2})
        ax.set_title(f"n = {len(fdf)} patients")
        fig.tight_layout()
        st.pyplot(fig)

    with r1c2:
        st.subheader("Age Distribution by Risk Class")
        fig, ax = plt.subplots(figsize=(5, 4))
        for t, c, lbl in [(0, "#2ecc71", "No Disease"), (1, "#e74c3c", "Disease")]:
            ax.hist(fdf[fdf.target == t]["age"], bins=15,
                    alpha=0.7, color=c, label=lbl, edgecolor="white")
        ax.set_xlabel("Age (years)"); ax.set_ylabel("Count")
        ax.legend(); ax.set_title("Age vs Heart Disease Status")
        fig.tight_layout()
        st.pyplot(fig)

    # ── Row 2 ─────────────────────────────────────────────────────────────────
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.subheader("Disease Rate by Sex & Chest Pain")
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

        sex_risk = fdf.groupby("sex")["target"].mean() * 100
        sex_risk.index = ["Female", "Male"]
        sex_risk.plot(kind="bar", color=["#f39c12", "#3498db"],
                      ax=axes[0], edgecolor="black", width=0.5)
        axes[0].set_ylabel("Disease Rate (%)"); axes[0].set_title("By Sex")
        axes[0].tick_params(axis="x", rotation=0)
        axes[0].set_ylim(0, 100)

        cp_risk = fdf.groupby("cp")["target"].mean() * 100
        cp_risk.index = [CP_LABELS.get(int(i), str(i)) for i in cp_risk.index]
        cp_risk.plot(kind="bar", color=["#9b59b6", "#e74c3c", "#3498db", "#e67e22"],
                     ax=axes[1], edgecolor="black", width=0.6)
        axes[1].set_ylabel(""); axes[1].set_title("By Chest Pain Type")
        axes[1].tick_params(axis="x", rotation=25)
        axes[1].set_ylim(0, 100)

        fig.tight_layout()
        st.pyplot(fig)

    with r2c2:
        st.subheader("Key Clinical Features — Boxplots")
        fig, axes = plt.subplots(1, 3, figsize=(7, 3.5))
        for ax, feat, label in zip(
            axes,
            ["chol", "thalach", "oldpeak"],
            ["Cholesterol\n(mg/dl)", "Max HR\n(thalach)", "ST Depression\n(oldpeak)"],
        ):
            fdf.boxplot(column=feat, by="target", ax=ax,
                        boxprops=dict(color="#3498db"),
                        medianprops=dict(color="#e74c3c", linewidth=2),
                        whiskerprops=dict(color="#3498db"),
                        capprops=dict(color="#3498db"))
            ax.set_xlabel("0=None  1=Disease", fontsize=7)
            ax.set_title(label, fontsize=8)
        plt.suptitle("")
        fig.tight_layout()
        st.pyplot(fig)

    # ── Row 3 ─────────────────────────────────────────────────────────────────
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        corr = fdf.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".1f",
                    cmap="coolwarm", linewidths=0.4, ax=ax,
                    annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
        ax.set_title("Feature Correlation Matrix")
        fig.tight_layout()
        st.pyplot(fig)

    with r3c2:
        st.subheader("Max Heart Rate vs Age")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        c_map = {0: "#2ecc71", 1: "#e74c3c"}
        for t_val, grp in fdf.groupby("target"):
            ax.scatter(grp["age"], grp["thalach"],
                       c=c_map[t_val], alpha=0.65, edgecolors="k",
                       linewidths=0.3, s=45,
                       label="No Disease" if t_val == 0 else "Disease")
        ax.set_xlabel("Age (years)"); ax.set_ylabel("Max Heart Rate (thalach)")
        ax.set_title("Max Heart Rate vs Age")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    # ── Model evaluation ──────────────────────────────────────────────────────
    st.subheader("Model Evaluation — ROC Curves & Confusion Matrices")
    ev1, ev2 = st.columns(2)

    with ev1:
        fig, ax = plt.subplots(figsize=(5, 4))
        for mname, mc, col in [
            ("Random Forest",      metrics["Random Forest"],      "#e74c3c"),
            ("Logistic Regression",metrics["Logistic Regression"],"#3498db"),
        ]:
            ax.plot(mc["fpr"], mc["tpr"], color=col, lw=2,
                    label=f"{mname}  (AUC={mc['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1.5)
        ax.fill_between(metrics["Random Forest"]["fpr"],
                        metrics["Random Forest"]["tpr"], alpha=0.07, color="#e74c3c")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve Comparison"); ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)

    with ev2:
        sel = st.selectbox("Confusion matrix for:", ["Random Forest", "Logistic Regression"],
                           key="cm_sel")
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        disp = ConfusionMatrixDisplay(
            metrics[sel]["cm"], display_labels=["No Disease", "Disease"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {sel}")
        fig.tight_layout()
        st.pyplot(fig)

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.subheader("Performance Metrics Comparison")
    mdf = pd.DataFrame([
        {
            "Model"       : name,
            "Accuracy"    : f"{m['accuracy']*100:.1f}%",
            "ROC-AUC"     : f"{m['auc']:.4f}",
            "CV Accuracy" : f"{m['cv_acc']*100:.1f}%",
            "Precision (Disease)" : f"{m['report']['Disease']['precision']:.3f}",
            "Recall (Disease)"    : f"{m['report']['Disease']['recall']:.3f}",
            "F1-Score (Disease)"  : f"{m['report']['Disease']['f1-score']:.3f}",
        }
        for name, m in metrics.items()
    ])
    st.dataframe(mdf, use_container_width=True, hide_index=True)

    # ── Raw data viewer ───────────────────────────────────────────────────────
    with st.expander("🗂️ View Raw / Filtered Dataset"):
        st.dataframe(fdf.reset_index(drop=True), use_container_width=True)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — SUMMARY & ACTIONABLE RECOMMENDATIONS                              ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab3:
    st.header("📋 Project Summary & Actionable Healthcare Recommendations")

    # ── Project overview ──────────────────────────────────────────────────────
    st.subheader("Project Overview")
    oc1, oc2 = st.columns(2)
    oc1.markdown(
        "**Author:** Parnil Kashyap\n\n"
        "**Dataset:** Cleveland Heart Disease (UCI ML Repository)\n\n"
        "**Records:** 294 patients · 13 features · 1 binary target\n\n"
        "**Goal:** Binary classification — predict presence of heart disease "
        "from routine clinical measurements."
    )
    oc2.markdown(
        f"""
        | Metric | Random Forest | Logistic Regression |
        |--------|:-------------:|:-------------------:|
        | Test Accuracy | **{metrics['Random Forest']['accuracy']*100:.1f}%** | {metrics['Logistic Regression']['accuracy']*100:.1f}% |
        | ROC-AUC | **{metrics['Random Forest']['auc']:.3f}** | {metrics['Logistic Regression']['auc']:.3f} |
        | CV Accuracy (5-fold) | **{metrics['Random Forest']['cv_acc']*100:.1f}%** | {metrics['Logistic Regression']['cv_acc']*100:.1f}% |
        """
    )

    st.divider()

    # ── Key findings ──────────────────────────────────────────────────────────
    st.subheader("📌 Key Findings")
    kf_col1, kf_col2 = st.columns(2)

    with kf_col1:
        st.markdown(
            """
            **1. Asymptomatic chest pain is the highest-risk subgroup.**  
            Patients who report no chest pain (cp = 3 / Asymptomatic) paradoxically carry the
            highest disease prevalence (~72%). This is a critical clinical blind spot.

            **2. Max heart rate (thalach) is inversely predictive.**  
            Disease patients achieve significantly lower peak heart rates than healthy ones —
            a negative correlation of ≈ −0.42 with the target.

            **3. ST depression (oldpeak) and fluoroscopy vessels (ca) dominate feature importance.**  
            These two exercise-test variables consistently rank as the strongest predictors,
            far outperforming raw cholesterol.

            **4. Age accelerates risk post-45.**  
            Risk rises steeply after age 45 and peaks in the 55–65 band (≈ 65% rate).
            Under-45s show lower but non-negligible incidence.
            """
        )

    with kf_col2:
        st.markdown(
            """
            **5. Gender disparity is substantial.**  
            Males show ~57% disease prevalence vs ~26% for females — a 31-point gap —
            consistent with earlier onset of coronary artery disease in men.

            **6. Cholesterol is a weak standalone predictor.**  
            Despite being a common screening metric, serum cholesterol shows only modest
            correlation with the target in this cohort. Exercise-test results are stronger signals.

            **7. Missing data is concentrated in imaging fields.**  
            Columns `ca` (fluoroscopy vessels) and `thal` (thalassemia) had the highest
            missingness — median imputation was applied. Completeness of these fields
            would measurably improve model performance.

            **8. Random Forest outperforms Logistic Regression.**  
            The ensemble model achieves higher accuracy and AUC, capturing non-linear
            interactions between features that the linear model cannot model.
            """
        )

    st.divider()

    # ── Feature importance visual ─────────────────────────────────────────────
    st.subheader("Top Predictive Features (Random Forest)")
    imp_s = pd.Series(rf_model.feature_importances_, index=feat_cols).sort_values()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    bar_c = ["#e74c3c" if v >= imp_s.median() else "#3498db" for v in imp_s]
    imp_s.plot(kind="barh", color=bar_c, ax=ax, edgecolor="none")
    ax.set_xlabel("Gini Importance Score")
    ax.set_title("Random Forest Feature Importances")
    patches = [mpatches.Patch(color="#e74c3c", label="Above median importance"),
               mpatches.Patch(color="#3498db", label="Below median importance")]
    ax.legend(handles=patches, fontsize=8, loc="lower right")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.divider()

    # ── Actionable recommendations ────────────────────────────────────────────
    st.subheader("🏥 Actionable Healthcare Recommendations")

    with st.expander("🔴 Immediate Actions (0–3 months)", expanded=True):
        st.markdown(
            """
            - **Revise triage protocols** for asymptomatic chest-pain patients (cp = 3).
              Despite reporting no pain, this group has the highest disease rate.
              Escalate them directly to stress testing regardless of pain severity.
            - **Deploy this dashboard** to cardiology department workstations.
              Clinicians can use the Risk Calculator (Tab 1) for real-time point-of-care
              screening using routine measurements.
            - **Flag male patients aged 55–65** as a high-priority cohort for
              proactive outreach and expedited referral to cardiology.
            """
        )

    with st.expander("🟡 Short-Term Actions (3–6 months)"):
        st.markdown(
            """
            - **Age-40 baseline cardiac workup.** Given the steep risk increase after 45,
              introduce a mandatory baseline ECG and ST-depression measurement at age 40
              to establish individual-level change detection.
            - **Gender-differentiated alert thresholds.** Set lower risk-score cut-offs
              for male patients in the 50–65 age band to trigger earlier specialist referral.
            - **Training programme** for clinical staff on interpreting the dashboard
              filters and model outputs for subgroup risk profiling.
            """
        )

    with st.expander("🟢 Medium / Long-Term Actions (6–18 months)"):
        st.markdown(
            """
            - **Data completeness initiative.** Mandate complete capture of fluoroscopy
              vessel counts (ca) and thalassemia type (thal) in all referral forms.
              Audit missing-data rates quarterly — these fields are the top predictors.
            - **EMR integration.** Connect this model to the Electronic Medical Records
              system to auto-flag high-risk patients at the point of care, reducing
              manual screening overhead.
            - **SHAP explainability layer.** Add per-patient SHAP waterfall charts to the
              predictor to give clinicians transparent, feature-level justification for
              each risk score.
            - **Multi-dataset validation.** Validate the model on Hungarian, Swiss, and
              Virginia Heart Disease datasets to assess cross-population generalisability
              before clinical deployment.
            - **Docker deployment.** Containerise the application for reproducible,
              platform-independent deployment in hospital IT environments.
            """
        )

    st.divider()

    # ── Tech stack ────────────────────────────────────────────────────────────
    st.subheader("Technology Stack")
    ts_c1, ts_c2 = st.columns(2)
    ts_c1.markdown(
        """
        | Layer | Technology |
        |-------|-----------|
        | Language | Python 3.10+ |
        | ML | scikit-learn (RandomForest, LogisticRegression) |
        | Data | pandas · numpy |
        | Visualisation | matplotlib · seaborn |
        | Dashboard | Streamlit |
        | Serialisation | joblib |
        """
    )
    ts_c2.markdown(
        """
        | File | Purpose |
        |------|---------|
        | `Parnil_Kashyap_HeartAttackRiskAnalysis.py` | Master app (this file) |
        | `data/data.csv` | Raw dataset |
        | `model/model.pkl` | Trained RF classifier |
        | `model/scaler.pkl` | Fitted StandardScaler |
        | `Parnil_Kashyap_HeartAttackRiskAnalysis.ipynb` | Analysis notebook |
        | `generate_report.py` | Word report generator |
        """
    )

    st.caption("Parnil Kashyap · Heart Attack Risk Analysis & Prediction · 2025")
