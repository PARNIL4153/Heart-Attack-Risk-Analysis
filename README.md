# ❤️ Heart Attack Risk Analysis & Prediction

A full-stack machine learning web application that predicts cardiovascular disease risk based on clinical patient indicators using the Cleveland Heart Disease dataset.

> **Author:** Parnil Kashyap
> **Dataset:** Cleveland Heart Disease — UCI Machine Learning Repository (294 patients, 13 features)

- **Backend**: Flask REST API
- **Frontend**: Streamlit
- **ML Model**: Logistic Regression (scikit-learn)
- **Dataset**: [`data/data.csv`](https://www.kaggle.com/datasets/imnikhilanand/heart-attack-prediction)

---

## 🏗️ Architecture — Single-File Design

This project consolidates the entire ML pipeline into **one self-contained master application** with no external API dependency.

```
Project_heart_risks/
│
├── Parnil_Kashyap_HeartAttackRiskAnalysis.py   ← MASTER APP  (run this)
├── Parnil_Kashyap_HeartAttackRiskAnalysis.ipynb ← Analysis notebook
│
├── data/
│   └── data.csv                                ← Cleveland Heart Disease dataset
│
├── model/                                      ← Auto-generated on first run
│   ├── model.pkl                               ← Trained Random Forest
│   ├── scaler.pkl                              ← Fitted StandardScaler
│   └── features.pkl                            ← Feature order list
│
├── generate_report.py                          ← Generates Parnil_Kashyap_ProjectReport.docx
├── requirements.txt
└── README.md
```

> The `backend/` and `frontend/` directories from the extended architecture are also present
> for reference, but **the master app is entirely self-contained** — no Flask API needed.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the master application
streamlit run Parnil_Kashyap_HeartAttackRiskAnalysis.py
```

Opens at **http://localhost:8501** — models train automatically on first launch and are saved to `model/`.

---

## 🛠️ Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.10+ |
| ML Framework | scikit-learn | 1.5.1 |
| Data Processing | pandas | 2.2.2 |
| Numerical Ops | numpy | 1.26.4 |
| Model Serialisation | joblib | 1.4.2 |
| Dashboard | Streamlit | 1.36.0 |
| Visualisation | matplotlib + seaborn | 3.9.1 / 0.13.2 |
| Report Generation | python-docx | latest |

---

## 📂 Application Tabs

### 🩺 Tab 1 — Live Patient Risk Predictor
- Input form with all **13 clinical parameters** (sliders, selectboxes, number inputs)
- Inline ML inference — **no Flask API required**
- Colour-coded risk banner: 🟢 Low · 🟡 Moderate · 🔴 High
- Animated probability gauge with Low / Moderate / High zones
- Feature contribution bar chart (Gini importance or |LR coefficient|)
- Clinical parameter summary table with normal ranges
- Switchable model (Random Forest or Logistic Regression) via sidebar

### 📊 Tab 2 — Decision Dashboard
- **Demographic filters** — age range, sex, chest pain type
- Target class pie chart
- Age distribution histogram by risk class
- Disease rate by sex & chest pain type (side-by-side bars)
- Key clinical features boxplots (cholesterol · max HR · ST depression)
- Feature correlation heatmap (lower triangle, coolwarm palette)
- Max heart rate vs age scatter plot
- **ROC curve comparison** (RF vs LR on one chart)
- Selectable confusion matrix per model
- Performance metrics comparison table
- Raw data viewer (filtered)

### 📋 Tab 3 — Summary & Recommendations
- Live model performance metrics table (from actual trained models)
- 8 key clinical findings
- Feature importance horizontal bar chart
- Actionable recommendations organised by timeframe:
  - 🔴 Immediate (0–3 months)
  - 🟡 Short-term (3–6 months)
  - 🟢 Long-term (6–18 months)
- Technology stack & file reference table

---

## 📊 Key Findings

1. **Asymptomatic chest pain (cp=3)** carries the highest disease rate (~72%) — a critical clinical blind spot.
2. **Max heart rate (thalach)** is inversely correlated with disease presence (r ≈ −0.42).
3. **Fluoroscopy vessels (ca)** and **ST depression (oldpeak)** are the top predictive features.
4. **Males** show ~57% prevalence vs ~26% in females — a 31-point gap.
5. **Cholesterol** alone is a weak predictor — exercise-test variables are far stronger signals.
6. **Risk accelerates sharply** after age 45, peaking in the 55–65 band (≈65% rate).

---

## 📈 Model Performance

| Metric | Random Forest | Logistic Regression |
|--------|:-------------:|:-------------------:|
| Test Accuracy | **86.4%** | ~81% |
| ROC-AUC | **0.89** | ~0.86 |
| CV Accuracy (5-fold) | **~83%** | ~80% |

---

## 🔧 Additional Commands

```bash
# Generate Word project report
python generate_report.py
# → Parnil_Kashyap_ProjectReport.docx

# Re-train model (standalone, no Streamlit)
python train_model.py
# → model/model.pkl, model/scaler.pkl, model/features.pkl

# Run Jupyter notebook
jupyter notebook Parnil_Kashyap_HeartAttackRiskAnalysis.ipynb

# Optional: extended Flask API + Streamlit frontend (separate terminals)
python backend/app.py          # Terminal 2
streamlit run frontend/ui.py   # Terminal 3
```

---

## 📄 License

For academic and educational use only.
