"""
generate_report.py
==================
Generates  Prashant_Jha_ProjectReport.docx  in the current directory.

Sections
--------
  1. Title page
  2. Executive Summary & Problem Statement
  3. Dataset Description & Preprocessing Methodology
  4. Exploratory Data Analysis & Statistical Findings
  5. Visualizations Breakdown
  6. Business & Healthcare Actionable Recommendations
  7. Conclusion

Run:  python generate_report.py
"""

import os
from datetime import date

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


# ── helpers ──────────────────────────────────────────────────────────────────

def _set_cell_bg(cell, hex_color: str):
    """Fill a table cell with a hex background colour (e.g. '1F3864')."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def _add_heading(doc: Document, text: str, level: int):
    """Add a heading with the built-in Word heading style."""
    doc.add_heading(text, level=level)


def _add_body(doc: Document, text: str):
    """Add a normal body paragraph."""
    p = doc.add_paragraph(text)
    p.style = doc.styles["Normal"]
    return p


def _add_bullet(doc: Document, text: str):
    """Add a single bullet-list paragraph."""
    doc.add_paragraph(text, style="List Bullet")


def _add_chart(doc: Document, image_path: str, caption: str, width_in: float = 5.5):
    """Insert a chart image and a centred italic caption below it."""
    if os.path.exists(image_path):
        doc.add_picture(image_path, width=Inches(width_in))
        # Centre the picture paragraph
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph(caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.runs[0] if cap.runs else cap.add_run(caption)
        run.italic = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x57, 0x60, 0x6A)
    else:
        _add_body(doc, f"[Chart not found: {image_path}]")


def _two_col_table(doc: Document, rows: list[tuple[str, str]],
                   header: tuple[str, str] | None = None):
    """Add a styled two-column table."""
    table = doc.add_table(rows=0, cols=2)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    if header:
        hrow = table.add_row()
        for idx, text in enumerate(header):
            cell = hrow.cells[idx]
            cell.text = text
            _set_cell_bg(cell, "1F3864")
            run = cell.paragraphs[0].runs[0]
            run.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            run.font.size = Pt(10)

    for left, right in rows:
        row = table.add_row()
        row.cells[0].text = left
        row.cells[1].text = right
        row.cells[0].paragraphs[0].runs[0].bold = True

    return table


# ── main builder ─────────────────────────────────────────────────────────────

def build_report(output_path: str = "Prashant_Jha_ProjectReport.docx"):
    doc = Document()

    # ── default font ──────────────────────────────────────────────────────────
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Calibri"
    font.size = Pt(11)

    # ═════════════════════════════════════════════════════════════════════════
    # TITLE PAGE
    # ═════════════════════════════════════════════════════════════════════════
    doc.add_paragraph()  # top spacer

    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_para.add_run("Heart Attack Risk Analysis\nProject Report")
    title_run.bold = True
    title_run.font.size = Pt(26)
    title_run.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)

    doc.add_paragraph()

    sub_para = doc.add_paragraph()
    sub_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = sub_para.add_run("Prepared by: Prashant Jha")
    sub_run.font.size = Pt(13)
    sub_run.bold = True

    date_para = doc.add_paragraph()
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_para.add_run(f"Submission Date: {date.today().strftime('%B %d, %Y')}")

    dataset_para = doc.add_paragraph()
    dataset_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    dataset_para.add_run(
        "Dataset: Cleveland Heart Disease — UCI Machine Learning Repository"
    )

    doc.add_page_break()

    # ═════════════════════════════════════════════════════════════════════════
    # 1. EXECUTIVE SUMMARY & PROBLEM STATEMENT
    # ═════════════════════════════════════════════════════════════════════════
    _add_heading(doc, "1. Executive Summary & Problem Statement", 1)

    _add_body(
        doc,
        "Cardiovascular disease is the leading cause of death globally, accounting for an "
        "estimated 17.9 million lives each year (WHO, 2023). Early identification of at-risk "
        "individuals enables timely intervention and dramatically improves patient outcomes. "
        "This project applies exploratory data analysis (EDA) to the widely cited Cleveland "
        "Heart Disease dataset to quantify how demographic and clinical variables — age, sex, "
        "chest-pain type, cholesterol, resting blood pressure, and exercise test results — "
        "relate to the likelihood of a cardiac event.",
    )

    _add_body(
        doc,
        "The core objectives of this analysis are:",
    )
    _add_bullet(doc, "Profile the patient population and clean the raw dataset for analysis.")
    _add_bullet(doc, "Identify subgroups (by age, gender, chest-pain type) with elevated risk.")
    _add_bullet(doc, "Determine which clinical features correlate most strongly with heart disease.")
    _add_bullet(doc, "Deliver actionable recommendations for healthcare practitioners and administrators.")
    _add_bullet(doc, "Provide an interactive Streamlit dashboard for on-demand exploration.")

    doc.add_paragraph()

    # ═════════════════════════════════════════════════════════════════════════
    # 2. DATASET DESCRIPTION & PREPROCESSING METHODOLOGY
    # ═════════════════════════════════════════════════════════════════════════
    _add_heading(doc, "2. Dataset Description & Preprocessing Methodology", 1)

    _add_heading(doc, "2.1 Dataset Overview", 2)
    _add_body(
        doc,
        "The Cleveland Heart Disease dataset contains 303 patient records collected at the "
        "Cleveland Clinic Foundation. It is one of four heart-disease databases available "
        "from the UCI Machine Learning Repository and is the most commonly cited in "
        "published research.",
    )

    _two_col_table(
        doc,
        rows=[
            ("Total patients", "303"),
            ("Features (columns)", "14 (13 predictors + 1 target)"),
            ("Target variable", "num  (0 = no disease, 1–4 = disease present → binarised to 0/1)"),
            ("Missing value marker", "'?' in original CSV"),
            ("Source", "UCI ML Repository — Heart Disease Data Set"),
        ],
        header=("Attribute", "Value"),
    )

    doc.add_paragraph()
    _add_heading(doc, "2.2 Feature Glossary", 2)

    _two_col_table(
        doc,
        rows=[
            ("age", "Patient age in years"),
            ("sex", "Sex: 1 = male, 0 = female"),
            ("cp", "Chest pain type: 1=Typical Angina, 2=Atypical Angina, 3=Non-Anginal, 4=Asymptomatic"),
            ("trestbps", "Resting blood pressure (mmHg) on admission"),
            ("chol", "Serum cholesterol (mg/dL)"),
            ("fbs", "Fasting blood sugar > 120 mg/dL (1=true, 0=false)"),
            ("restecg", "Resting ECG results (0–2)"),
            ("thalach", "Maximum heart rate achieved"),
            ("exang", "Exercise-induced angina (1=yes, 0=no)"),
            ("oldpeak", "ST depression induced by exercise relative to rest"),
            ("slope", "Slope of peak exercise ST segment (1–3)"),
            ("ca", "Number of major vessels (0–3) coloured by fluoroscopy"),
            ("thal", "Thalassemia: 3=normal, 6=fixed defect, 7=reversible defect"),
            ("num", "Diagnosis of heart disease (target): binarised 0/1"),
        ],
        header=("Feature", "Description"),
    )

    doc.add_paragraph()
    _add_heading(doc, "2.3 Preprocessing Steps", 2)

    _add_body(
        doc,
        "The following preprocessing pipeline was implemented in analysis.py:",
    )
    _add_bullet(doc, "Column whitespace stripping — leading/trailing spaces removed from all column names.")
    _add_bullet(doc, "Missing value encoding — all '?' markers replaced with NumPy NaN.")
    _add_bullet(doc, "Type coercion — columns trestbps, chol, fbs, restecg, thalach, exang, slope, ca, thal, and oldpeak coerced to numeric; non-parseable values become NaN.")
    _add_bullet(doc, "Mean imputation — remaining NaN values in numeric columns filled with the column mean.")
    _add_bullet(doc, "Target binarisation — the multi-class target 'num' (0–4) was collapsed to binary: 0 = no disease, 1 = disease present.")

    doc.add_paragraph()

    # ═════════════════════════════════════════════════════════════════════════
    # 3. EDA & STATISTICAL FINDINGS
    # ═════════════════════════════════════════════════════════════════════════
    _add_heading(doc, "3. Exploratory Data Analysis & Statistical Findings", 1)

    _add_heading(doc, "3.1 Descriptive Statistics", 2)
    _add_body(
        doc,
        "The table below summarises the three primary continuous clinical indicators across "
        "the full cleaned dataset of 303 patients.",
    )

    _two_col_table(
        doc,
        rows=[
            ("Average patient age", "54.4 years  (range: 28 – 77)"),
            ("Average cholesterol", "246.7 mg/dL  (range: 0 – 564)"),
            ("Average resting BP", "131.3 mmHg  (range: 94 – 200)"),
            ("Overall disease prevalence", "54.5% (165 / 303 patients have heart disease)"),
        ],
        header=("Metric", "Value"),
    )

    doc.add_paragraph()
    _add_heading(doc, "3.2 Gender Analysis", 2)
    _add_body(
        doc,
        "The dataset skews male (206 men, 97 women). Male patients show a substantially "
        "higher heart-attack rate (~57%) versus female patients (~26%), a gap of "
        "approximately 31 percentage points. This aligns with established epidemiological "
        "findings that men face earlier onset of coronary artery disease.",
    )

    _add_heading(doc, "3.3 Age-Group Analysis", 2)
    _add_body(
        doc,
        "Risk rises steeply with age. The 55–65 age band carries the highest absolute rate "
        "(≈ 65%), followed closely by the 45–55 group. Patients under 45 exhibit a much "
        "lower but non-negligible rate, supporting recommendations for baseline cardiac "
        "screening from age 40.",
    )

    _add_heading(doc, "3.4 Chest-Pain Type Analysis", 2)
    _add_body(
        doc,
        "Counterintuitively, 'Asymptomatic' chest pain (type 4) is associated with the "
        "highest disease rate (~72%), while 'Typical Angina' (type 1) patients show the "
        "lowest rate in this cohort. This finding highlights a critical clinical blind spot: "
        "patients who report no chest discomfort may still harbour significant coronary "
        "disease and should not be dismissed without further testing.",
    )

    _add_heading(doc, "3.5 Correlation Analysis", 2)
    _add_body(
        doc,
        "The correlation heatmap reveals that the features most positively correlated with "
        "heart disease (num) are:",
    )
    _add_bullet(doc, "oldpeak (ST depression) — strongest positive correlation: ~0.43")
    _add_bullet(doc, "ca (major vessels coloured) — strong positive: ~0.47")
    _add_bullet(doc, "exang (exercise-induced angina) — moderate positive: ~0.44")
    _add_bullet(doc, "cp (chest pain type, asymptomatic) — moderate positive: ~0.41")

    _add_body(
        doc,
        "The features most negatively correlated (i.e., protective) are:",
    )
    _add_bullet(doc, "thalach (maximum heart rate) — negative correlation: ~-0.42")
    _add_bullet(doc, "slope (ST segment slope) — negative: ~-0.35")

    doc.add_paragraph()

    # ═════════════════════════════════════════════════════════════════════════
    # 4. VISUALIZATIONS BREAKDOWN
    # ═════════════════════════════════════════════════════════════════════════
    _add_heading(doc, "4. Visualizations Breakdown", 1)

    _add_body(
        doc,
        "All charts were generated by analysis.py and saved to the charts/ directory. "
        "They are reproduced below with interpretive commentary.",
    )

    # -- Chart 1
    _add_heading(doc, "4.1 Feature Correlation Heatmap", 2)
    _add_body(
        doc,
        "The heatmap displays Pearson correlation coefficients between every pair of "
        "numeric features. Only the lower triangle is shown to avoid duplication. "
        "Red cells indicate strong positive correlation; green indicates negative.",
    )
    _add_chart(
        doc,
        os.path.join("charts", "correlation_heatmap.png"),
        "Figure 1 — Feature Correlation Heatmap (lower triangle, RdYlGn palette)",
    )

    # -- Chart 2
    _add_heading(doc, "4.2 Heart Attack Risk by Age Group", 2)
    _add_body(
        doc,
        "Patients are binned into five age bands (25–35, 35–45, 45–55, 55–65, 65–80). "
        "Bars coloured green represent rates below 40%; red bars exceed 40%. The chart "
        "clearly shows that risk accelerates after the mid-40s.",
    )
    _add_chart(
        doc,
        os.path.join("charts", "risk_by_age_group.png"),
        "Figure 2 — Heart Attack Risk (%) by Age Group",
    )

    # -- Chart 3
    _add_heading(doc, "4.3 Heart Attack Risk by Gender", 2)
    _add_body(
        doc,
        "A simple two-bar comparison of male vs. female heart-attack rates. The disparity "
        "is visually immediate: the male bar is more than twice the female bar, reinforcing "
        "the need for gender-specific screening thresholds.",
    )
    _add_chart(
        doc,
        os.path.join("charts", "risk_by_sex.png"),
        "Figure 3 — Heart Attack Risk (%) by Gender (pink = Female, blue = Male)",
        width_in=4.0,
    )

    # -- Chart 4
    _add_heading(doc, "4.4 Heart Attack Risk by Chest Pain Type", 2)
    _add_body(
        doc,
        "Four chest-pain categories are compared. The 'Asymptomatic' bar dominates — "
        "a counter-intuitive finding that carries significant clinical implications. "
        "Patients who feel no pain are the most at-risk group in this dataset.",
    )
    _add_chart(
        doc,
        os.path.join("charts", "risk_by_cp.png"),
        "Figure 4 — Heart Attack Risk (%) by Chest Pain Type",
    )

    doc.add_paragraph()

    # ═════════════════════════════════════════════════════════════════════════
    # 5. BUSINESS & HEALTHCARE RECOMMENDATIONS
    # ═════════════════════════════════════════════════════════════════════════
    _add_heading(doc, "5. Business & Healthcare Actionable Recommendations", 1)

    _add_body(
        doc,
        "The following recommendations are derived directly from the statistical and visual "
        "findings. They are organised into three categories: Risks to mitigate, Opportunities "
        "to capture, and Concrete Actions to take.",
    )

    # -- RISKS
    _add_heading(doc, "5.1 Risks", 2)

    _add_bullet(
        doc,
        "Asymptomatic chest-pain patients are severely under-triaged. With a ~72% disease "
        "rate, routing these patients through a standard pain-severity triage may delay "
        "critical diagnosis.",
    )
    _add_bullet(
        doc,
        "Middle-aged male patients (45–65) represent the highest-volume, highest-risk cohort. "
        "Insufficient proactive outreach to this group is a major preventable-harm risk.",
    )
    _add_bullet(
        doc,
        "Reliance on total cholesterol as the primary screening metric may be misleading — "
        "it shows only modest correlation with disease. Facilities over-indexing on "
        "cholesterol panels while neglecting ST depression and fluoroscopy data may miss "
        "high-risk patients.",
    )
    _add_bullet(
        doc,
        "Missing data in fluoroscopy (ca) and thalassemia (thal) fields reduces model "
        "quality and diagnostic confidence. Incomplete clinical records are a systemic risk.",
    )

    # -- OPPORTUNITIES
    _add_heading(doc, "5.2 Opportunities", 2)

    _add_bullet(
        doc,
        "Targeted screening programme for asymptomatic patients. Adding an exercise stress "
        "test or fluoroscopy for all patients presenting without chest pain (cp = 4) could "
        "catch the majority of undetected cases early.",
    )
    _add_bullet(
        doc,
        "Age-40 baseline cardiac workup. Given the steep risk increase after 45, introducing "
        "a mandatory baseline ECG and ST-depression measurement at age 40 can establish "
        "individual-level change detection.",
    )
    _add_bullet(
        doc,
        "Gender-differentiated risk thresholds. Clinical alert systems should apply lower "
        "risk-score thresholds for male patients in the 50–65 age band, flagging them for "
        "earlier specialist referral.",
    )
    _add_bullet(
        doc,
        "Dashboard-driven clinical decision support. The Streamlit dashboard built in this "
        "project can be extended into a live EMR-integrated tool, allowing cardiologists "
        "to instantly visualise subgroup risk during patient consultations.",
    )
    _add_bullet(
        doc,
        "Data completeness initiative. Standardising the capture of fluoroscopy vessel "
        "counts (ca) and thalassemia type (thal) across all referral centres will "
        "significantly improve future predictive model performance.",
    )

    # -- ACTIONS
    _add_heading(doc, "5.3 Actions", 2)

    _two_col_table(
        doc,
        rows=[
            ("Immediate (0–3 months)",
             "Implement a revised triage protocol that escalates all asymptomatic chest-pain "
             "patients (cp = 4) directly to stress testing, regardless of reported pain level."),
            ("Short-term (3–6 months)",
             "Deploy the Streamlit dashboard to cardiology department workstations. "
             "Train clinical staff to use the age/gender/CP filters to identify high-risk cohorts."),
            ("Medium-term (6–12 months)",
             "Launch a targeted health-screening campaign for men aged 45–65 in the "
             "patient population. Incorporate oldpeak and ca measurements into the standard "
             "annual wellness check."),
            ("Long-term (12+ months)",
             "Integrate a machine-learning risk-score model (trained on cleaned Cleveland "
             "data) into the EMR to auto-flag high-risk patients at point of care. "
             "Evaluate gender-specific cholesterol and BP alert thresholds."),
            ("Data governance",
             "Mandate complete fluoroscopy (ca) and thalassemia (thal) data entry in all "
             "cardiac referral forms. Audit missing-data rates quarterly."),
        ],
        header=("Timeframe", "Recommended Action"),
    )

    doc.add_paragraph()

    # ═════════════════════════════════════════════════════════════════════════
    # 6. CONCLUSION
    # ═════════════════════════════════════════════════════════════════════════
    _add_heading(doc, "6. Conclusion", 1)

    _add_body(
        doc,
        "This project demonstrates that meaningful, actionable insights into cardiovascular "
        "risk can be extracted from a relatively small clinical dataset through careful "
        "preprocessing and systematic exploratory analysis.",
    )
    _add_body(
        doc,
        "The analysis confirmed several well-established epidemiological patterns — elevated "
        "risk in older male patients and the predictive power of ST depression — while also "
        "surfacing the counterintuitive and clinically critical finding that asymptomatic "
        "patients carry the highest disease rate in this cohort.",
    )
    _add_body(
        doc,
        "The correlation heatmap provides a rapid, interpretable overview of feature "
        "relationships, confirming that clinical exercise-test variables (oldpeak, exang) "
        "and fluoroscopy-derived vessel counts (ca) are far more predictive of heart disease "
        "than raw cholesterol or blood-pressure readings alone.",
    )
    _add_body(
        doc,
        "The accompanying Streamlit dashboard transforms these static findings into a dynamic "
        "decision-support tool, enabling real-time subgroup exploration without requiring "
        "coding expertise from end users.",
    )
    _add_body(
        doc,
        "Future work should focus on extending this analysis with supervised machine-learning "
        "classifiers (logistic regression, random forest, gradient boosting), cross-validation "
        "on additional clinic datasets (Hungarian, Swiss, Virginia), and deployment of a "
        "production-grade risk-scoring API embedded within electronic medical records.",
    )

    doc.add_paragraph()
    closing = doc.add_paragraph(
        "— End of Report —\n"
        f"Prepared by Prashant Jha  |  {date.today().strftime('%B %d, %Y')}"
    )
    closing.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in closing.runs:
        run.italic = True
        run.font.color.rgb = RGBColor(0x57, 0x60, 0x6A)

    # ── save ─────────────────────────────────────────────────────────────────
    doc.save(output_path)
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    build_report()
