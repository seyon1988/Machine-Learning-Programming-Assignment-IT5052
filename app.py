"""
Heart Disease Prediction — Professional Streamlit Application
Supports Logistic Regression and Random Forest models.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


# Page config (must be first Streamlit call)
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)



# Constants
MODEL_DIR = Path(__file__).parent / "output" / "model"
print(f"model path:{MODEL_DIR.resolve()}")

FEATURE_ORDER = [
    "Age", "Gender", "Blood Pressure", "Cholesterol Level",
    "Exercise Habits", "Smoking", "Family Heart Disease", "Diabetes",
    "BMI", "High Blood Pressure", "Low HDL Cholesterol", "High LDL Cholesterol",
    "Alcohol Consumption", "Stress Level", "Sleep Hours", "Sugar Consumption",
    "Triglyceride Level", "Fasting Blood Sugar", "CRP Level", "Homocysteine Level",
]

BINARY_MAP   = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
ORDINAL_MAP  = {"Low": 0, "Medium": 1, "High": 2}

# Reference stats for StandardScaler fallback (population-level estimates)
SCALER_STATS = {
    "Age":                  {"mean": 54.5,  "std": 15.0},
    "Gender":               {"mean": 0.50,  "std": 0.50},
    "Blood Pressure":       {"mean": 122.0, "std": 21.0},
    "Cholesterol Level":    {"mean": 200.0, "std": 50.0},
    "Exercise Habits":      {"mean": 1.0,   "std": 0.82},
    "Smoking":              {"mean": 0.50,  "std": 0.50},
    "Family Heart Disease": {"mean": 0.50,  "std": 0.50},
    "Diabetes":             {"mean": 0.50,  "std": 0.50},
    "BMI":                  {"mean": 27.5,  "std": 5.5},
    "High Blood Pressure":  {"mean": 0.50,  "std": 0.50},
    "Low HDL Cholesterol":  {"mean": 0.50,  "std": 0.50},
    "High LDL Cholesterol": {"mean": 0.50,  "std": 0.50},
    "Alcohol Consumption":  {"mean": 1.0,   "std": 0.82},
    "Stress Level":         {"mean": 1.0,   "std": 0.82},
    "Sleep Hours":          {"mean": 7.0,   "std": 1.5},
    "Sugar Consumption":    {"mean": 1.0,   "std": 0.82},
    "Triglyceride Level":   {"mean": 150.0, "std": 80.0},
    "Fasting Blood Sugar":  {"mean": 100.0, "std": 30.0},
    "CRP Level":            {"mean": 3.0,   "std": 3.5},
    "Homocysteine Level":   {"mean": 10.0,  "std": 5.0},
}


# CSS Styling
st.markdown("""
<style>
/* ─── Global ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ─── Hero banner ────────────────────────── */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.hero-icon { font-size: 3.5rem; }
.hero h1   { font-size: 1.9rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
.hero p    { margin: 0.3rem 0 0; font-size: 0.95rem; opacity: 0.75; }

/* ─── Section cards ──────────────────────── */
.section-card {
    background: #ffffff;
    border: 1px solid #e8ecf0;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.section-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #0f3460;
    margin-bottom: 0.9rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e8ecf0;
}

/* ─── Result cards ───────────────────────── */
.result-high {
    background: linear-gradient(135deg, #fff5f5, #ffe0e0);
    border: 1.5px solid #fc8181;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
}
.result-low {
    background: linear-gradient(135deg, #f0fff4, #c6f6d5);
    border: 1.5px solid #68d391;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
}
.result-label  { font-size: 0.75rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; color: #555; margin-bottom: 0.3rem; }
.result-value  { font-size: 2.2rem; font-weight: 800; margin: 0; }
.result-sub    { font-size: 0.85rem; color: #666; margin-top: 0.3rem; }

.stat-box {
    background: #f7f9fc;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.stat-label { font-size: 0.72rem; color: #888; font-weight: 500; letter-spacing: 0.5px; text-transform: uppercase; }
.stat-value { font-size: 1.4rem; font-weight: 700; color: #1a202c; }

/* ─── Badge ──────────────────────────────── */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-blue  { background: #ebf4ff; color: #2b6cb0; }
.badge-green { background: #f0fff4; color: #276749; }
.badge-red   { background: #fff5f5; color: #9b2c2c; }

/* ─── Divider ────────────────────────────── */
hr.styled { border: none; border-top: 1px solid #e8ecf0; margin: 1rem 0; }

/* ─── Sidebar ────────────────────────────── */
[data-testid="stSidebar"] { background: #f7f9fc; }
.sidebar-logo { text-align: center; margin-bottom: 1.5rem; }
.sidebar-logo .logo-icon { font-size: 2.5rem; }
.sidebar-logo h2 { font-size: 1rem; font-weight: 700; color: #0f3460; margin: 0.3rem 0 0; }

.model-card {
    background: white;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    border: 2px solid transparent;
    cursor: pointer;
    transition: border 0.2s;
}
.model-active { border-color: #0f3460; }

/* ─── Disclaimer ─────────────────────────── */
.disclaimer {
    background: #fffbeb;
    border: 1px solid #f6e05e;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
    color: #744210;
    margin-top: 1rem;
}

/* ─── Predict button ─────────────────────── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0f3460, #1a5276);
    color: white;
    border: none;
    padding: 0.75rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    border-radius: 10px;
    width: 100%;
    transition: opacity 0.2s;
}
div[data-testid="stButton"] > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)



# Model / Scaler Loading
@st.cache_resource(show_spinner=False)
def load_models():
    lr_path = MODEL_DIR / "logistic_regression_model.pkl"
    rf_path = MODEL_DIR / "random_forest_model.pkl"
    sc_path = MODEL_DIR / "scaler.pkl"

    models, scaler = {}, None

    if lr_path.exists():
        models["Logistic Regression"] = joblib.load(lr_path)
    if rf_path.exists():
        models["Random Forest"]       = joblib.load(rf_path)
    if sc_path.exists():
        scaler = joblib.load(sc_path)

    return models, scaler


def manual_scale(values: np.ndarray) -> np.ndarray:
    """Fallback StandardScaler using population reference stats."""
    scaled = []
    for feat, val in zip(FEATURE_ORDER, values):
        s = SCALER_STATS[feat]
        scaled.append((val - s["mean"]) / s["std"])
    return np.array(scaled, dtype=float)


def preprocess(raw: dict, scaler) -> np.ndarray:
    """Encode and scale a single patient record."""
    binary_cols  = ["Gender", "Smoking", "Family Heart Disease", "Diabetes",
                    "High Blood Pressure", "Low HDL Cholesterol", "High LDL Cholesterol"]
    ordinal_cols = ["Exercise Habits", "Alcohol Consumption", "Stress Level", "Sugar Consumption"]

    encoded = {}
    for feat, val in raw.items():
        if feat in binary_cols:
            encoded[feat] = BINARY_MAP[val]
        elif feat in ordinal_cols:
            encoded[feat] = ORDINAL_MAP[val]
        else:
            encoded[feat] = float(val)

    vec = np.array([encoded[f] for f in FEATURE_ORDER], dtype=float)

    if scaler is not None:
        vec = scaler.transform(vec.reshape(1, -1)).flatten()
    else:
        vec = manual_scale(vec)

    return vec.reshape(1, -1)


# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon">🫀</div>
        <h2>CardioRisk AI</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("###   Select Model")
    models, scaler = load_models()

    if not models:
        st.error("No models found in models/ folder.")
        st.stop()

    model_choice = st.selectbox(
        "Prediction Engine",
        list(models.keys()),
        help="Choose between Logistic Regression and Random Forest."
    )
    selected_model = models[model_choice]

    st.markdown("---")
    st.markdown("### ℹ️ Model Info")

    if model_choice == "Logistic Regression":
        st.markdown('<span class="badge badge-blue">Linear Classifier</span>', unsafe_allow_html=True)
        st.caption("Optimised with L2 regularisation (C=10). Best for interpretability.")
    else:
        st.markdown('<span class="badge badge-green">Ensemble Method</span>', unsafe_allow_html=True)
        st.caption("100 decision trees. Captures complex non-linear patterns.")

    st.markdown("---")
    if scaler is None:
        st.warning("⚠️ scaler.pkl not found — using reference statistics for scaling. Save your scaler for best accuracy.")
    else:
        st.success("✅ Scaler loaded successfully.")

    st.markdown("""
    <div class="disclaimer">
        ⚕️ <strong>Clinical Disclaimer</strong><br>
        This tool is for educational/research purposes only. It does not constitute medical advice. Always consult a qualified healthcare professional.
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <div class="hero-icon">🫀</div>
    <div>
        <h1>Heart Disease Risk Predictor</h1>
        <p>Enter patient clinical data below to generate a personalised cardiovascular risk assessment using machine learning.</p>
    </div>
</div>
""", unsafe_allow_html=True)


col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    # ── Section 1: Demographics & Vitals ──────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Demographics & Vitals</div>', unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    age            = d1.number_input("Age (years)",         min_value=10,  max_value=100, value=50, step=1)
    gender         = d2.selectbox("Gender",                 ["Male", "Female"])
    bmi            = d3.number_input("BMI",                 min_value=10.0, max_value=55.0, value=26.0, step=0.1, format="%.1f")

    v1, v2, v3 = st.columns(3)
    blood_pressure = v1.number_input("Blood Pressure (mmHg)", min_value=60, max_value=240, value=120)
    sleep_hours    = v2.number_input("Sleep Hours / day",   min_value=2.0, max_value=14.0, value=7.0, step=0.5, format="%.1f")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 2: Laboratory Results ─────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧪 Laboratory Results</div>', unsafe_allow_html=True)

    l1, l2 = st.columns(2)
    cholesterol      = l1.number_input("Cholesterol (mg/dL)",          min_value=50,  max_value=500, value=200)
    triglycerides    = l2.number_input("Triglyceride Level (mg/dL)",   min_value=20,  max_value=1000, value=150)

    l3, l4 = st.columns(2)
    fasting_bs       = l3.number_input("Fasting Blood Sugar (mg/dL)", min_value=50,  max_value=400, value=100)
    crp              = l4.number_input("CRP Level (mg/L)",             min_value=0.0, max_value=50.0, value=2.5, step=0.1, format="%.1f")

    homocysteine     = st.number_input("Homocysteine Level (µmol/L)", min_value=1.0, max_value=60.0, value=10.0, step=0.5, format="%.1f")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 3: Risk Factors ────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚠️ Clinical Risk Factors</div>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    smoking          = r1.selectbox("Smoking",              ["No", "Yes"])
    family_hd        = r2.selectbox("Family Heart Disease", ["No", "Yes"])
    diabetes         = r3.selectbox("Diabetes",             ["No", "Yes"])

    r4, r5, r6 = st.columns(3)
    high_bp          = r4.selectbox("High Blood Pressure",  ["No", "Yes"])
    low_hdl          = r5.selectbox("Low HDL Cholesterol",  ["No", "Yes"])
    high_ldl         = r6.selectbox("High LDL Cholesterol", ["No", "Yes"])

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 4: Lifestyle ───────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏃 Lifestyle Factors</div>', unsafe_allow_html=True)

    ls1, ls2, ls3, ls4 = st.columns(4)
    exercise         = ls1.selectbox("Exercise Habits",     ["Low", "Medium", "High"])
    alcohol          = ls2.selectbox("Alcohol Consumption", ["Low", "Medium", "High"])
    stress           = ls3.selectbox("Stress Level",        ["Low", "Medium", "High"])
    sugar            = ls4.selectbox("Sugar Consumption",   ["Low", "Medium", "High"])

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Predict Button ─────────────────────────
    predict_clicked = st.button("🔍 Predict Heart Disease Risk", use_container_width=True)



# Results Panel (right column)
with col_right:
    st.markdown("### 📊 Prediction Results")

    if not predict_clicked:
        st.info("Fill in the patient data on the left and click *Predict* to see the risk assessment.")

        # Show feature importance placeholder
        st.markdown("---")
        st.markdown("##### About the Features")
        fi_info = {
            "Age": "Older patients carry higher baseline risk",
            "Cholesterol Level": "Key lipid marker for artery disease",
            "BMI": "Obesity increases cardiac workload",
            "CRP Level": "Inflammation marker linked to atherosclerosis",
            "Blood Pressure": "Sustained high BP damages arteries",
        }
        for feat, desc in fi_info.items():
            st.markdown(f"*{feat}* — {desc}")

    else:
        # ── Build input dict ──────────────────
        raw_input = {
            "Age":                  age,
            "Gender":               gender,
            "Blood Pressure":       blood_pressure,
            "Cholesterol Level":    cholesterol,
            "Exercise Habits":      exercise,
            "Smoking":              smoking,
            "Family Heart Disease": family_hd,
            "Diabetes":             diabetes,
            "BMI":                  bmi,
            "High Blood Pressure":  high_bp,
            "Low HDL Cholesterol":  low_hdl,
            "High LDL Cholesterol": high_ldl,
            "Alcohol Consumption":  alcohol,
            "Stress Level":         stress,
            "Sleep Hours":          sleep_hours,
            "Sugar Consumption":    sugar,
            "Triglyceride Level":   triglycerides,
            "Fasting Blood Sugar":  fasting_bs,
            "CRP Level":            crp,
            "Homocysteine Level":   homocysteine,
        }

        X_input = preprocess(raw_input, scaler)

        prediction  = selected_model.predict(X_input)[0]
        probability = selected_model.predict_proba(X_input)[0]
        risk_prob   = probability[1] * 100   # probability of heart disease
        safe_prob   = probability[0] * 100

        # ── Result Card ───────────────────────
        if prediction == 1:
            st.markdown(f"""
            <div class="result-high">
                <div class="result-label">Risk Assessment</div>
                <div class="result-value" style="color:#c53030;">⚠️ HIGH RISK</div>
                <div class="result-sub">Heart Disease Likely Detected</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low">
                <div class="result-label">Risk Assessment</div>
                <div class="result-value" style="color:#276749;">✅ LOW RISK</div>
                <div class="result-sub">No Significant Risk Detected</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr class='styled'>", unsafe_allow_html=True)

        # ── Probability Stats ─────────────────
        sc1, sc2 = st.columns(2)
        sc1.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Disease Probability</div>
            <div class="stat-value" style="color:#e53e3e;">{risk_prob:.1f}%</div>
        </div>""", unsafe_allow_html=True)
        sc2.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Healthy Probability</div>
            <div class="stat-value" style="color:#38a169;">{safe_prob:.1f}%</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gauge Chart ───────────────────────
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_prob,
            number={"suffix": "%", "font": {"size": 28, "color": "#1a202c"}},
            title={"text": "Cardiovascular Risk Score", "font": {"size": 14, "color": "#555"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#ccc"},
                "bar":  {"color": "#e53e3e" if prediction == 1 else "#38a169", "thickness": 0.25},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#eee",
                "steps": [
                    {"range": [0,  33], "color": "#c6f6d5"},
                    {"range": [33, 66], "color": "#fefcbf"},
                    {"range": [66, 100],"color": "#fed7d7"},
                ],
                "threshold": {
                    "line": {"color": "#1a202c", "width": 3},
                    "thickness": 0.8,
                    "value": risk_prob
                }
            }
        ))
        gauge.update_layout(
            height=260,
            margin=dict(t=40, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(gauge, use_container_width=True)

        # ── Feature Importance (RF only) ──────
        if model_choice == "Random Forest" and hasattr(selected_model, "feature_importances_"):
            st.markdown("##### 📌 Top Feature Importances")
            importances = selected_model.feature_importances_
            fi_df = pd.DataFrame({
                "Feature":    FEATURE_ORDER,
                "Importance": importances
            }).sort_values("Importance", ascending=True).tail(10)

            fig_fi = px.bar(
                fi_df, x="Importance", y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=["#bee3f8", "#2b6cb0"],
            )
            fig_fi.update_layout(
                height=300,
                margin=dict(t=10, b=10, l=0, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                coloraxis_showscale=False,
                yaxis_title="",
                xaxis_title="Importance Score",
                font=dict(size=11),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        # ── LR Coefficients ───────────────────
        elif model_choice == "Logistic Regression" and hasattr(selected_model, "coef_"):
            st.markdown("##### 📌 Top Feature Coefficients")
            coefs = selected_model.coef_[0]
            coef_df = pd.DataFrame({
                "Feature":     FEATURE_ORDER,
                "Coefficient": coefs
            })
            coef_df["Abs"] = coef_df["Coefficient"].abs()
            top10 = coef_df.sort_values("Abs", ascending=True).tail(10)
            top10["Color"] = top10["Coefficient"].apply(lambda x: "#fc8181" if x > 0 else "#68d391")

            fig_coef = go.Figure(go.Bar(
                x=top10["Coefficient"],
                y=top10["Feature"],
                orientation="h",
                marker_color=top10["Color"].tolist(),
            ))
            fig_coef.add_vline(x=0, line_width=1, line_dash="dash", line_color="#888")
            fig_coef.update_layout(
                height=300,
                margin=dict(t=10, b=10, l=0, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Coefficient",
                yaxis_title="",
                font=dict(size=11),
            )
            st.plotly_chart(fig_coef, use_container_width=True)

        # ── Patient Summary Table ─────────────
        with st.expander("📋 View Patient Data Summary"):
            summary = pd.DataFrame({
                "Feature": list(raw_input.keys()),
                "Value":   [str(v) for v in raw_input.values()]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)



# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.78rem;'>"
    "CardioRisk AI · Built with Streamlit · For Research & Educational Use Only"
    "</p>",
    unsafe_allow_html=True
)