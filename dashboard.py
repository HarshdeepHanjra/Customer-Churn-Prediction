# dashboard.py
# ============================================================
#  Customer Churn Intelligence Platform
#  Author : Harshdeep Singh
#  Stack  : Python · scikit-learn · Plotly · Streamlit
#
#  Install deps:
#      pip install streamlit pandas numpy scikit-learn plotly joblib
#
#  Run:
#      streamlit run dashboard.py
#
#  Expected files (optional — the app falls back to a live-trained
#  synthetic demo dataset if any of these are missing, so it runs
#  out of the box even before you plug in your own pipeline):
#      saved_models/churn_model.pkl
#      Dataset/X_train.csv, Dataset/X_test.csv, Dataset/y_test.csv
#      results/model_comparison.csv, results/business_metrics.csv
#      results/shap_feature_importance.csv   (optional)
# ============================================================

import os
import time
import warnings

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
#  FILE LOCATIONS (edit these if your project layout differs)
# ──────────────────────────────────────────────────────────
MODEL_PATH = "saved_models/churn_model.pkl"
X_TRAIN_PATH = "Dataset/X_train.csv"
X_TEST_PATH = "Dataset/X_test.csv"
Y_TEST_PATH = "Dataset/y_test.csv"
COMPARISON_PATH = "results/model_comparison.csv"
BUSINESS_PATH = "results/business_metrics.csv"
IMPORTANCE_PATH = "results/shap_feature_importance.csv"

# ──────────────────────────────────────────────────────────
#  THEME CSS
# ──────────────────────────────────────────────────────────
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-base: #0a0c10;
    --bg-card: #11141c;
    --bg-card2: #161a24;
    --bg-hover: #1c2130;
    --border: #22273a;
    --border-light: #2a3048;
    --accent-blue: #3b82f6;
    --accent-teal: #14b8a6;
    --accent-purple: #8b5cf6;
    --accent-amber: #f59e0b;
    --accent-red: #ef4444;
    --accent-green: #22c55e;
    --text-primary: #e8eaf0;
    --text-secondary: #8b909e;
    --text-muted: #4a5168;
    --font-sans: 'Inter', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
    --radius: 16px;
    --radius-sm: 10px;
    --shadow: 0 8px 32px rgba(0,0,0,0.6);
}

html, body, [class*="css"] {
    font-family: var(--font-sans) !important;
}
.stApp {
    background:
        radial-gradient(1200px 600px at 15% -10%, rgba(59,130,246,0.07), transparent 60%),
        radial-gradient(1000px 500px at 100% 0%, rgba(139,92,246,0.06), transparent 55%),
        var(--bg-base) !important;
}
section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
.block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1440px !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stRadio label {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
}

/* Scoped to the sidebar nav only, so other radio widgets (e.g. the
   "Select by" toggle on the Customer Analysis page) keep their
   normal, horizontal-friendly layout instead of being forced vertical. */
section[data-testid="stSidebar"] .stRadio > div {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 10px 16px !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    border: 1px solid transparent !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: var(--bg-hover) !important;
    color: var(--text-primary) !important;
    border-color: var(--border) !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] [aria-checked="true"] {
    background: rgba(59,130,246,0.15) !important;
    border-color: rgba(59,130,246,0.3) !important;
    color: var(--accent-blue) !important;
}

/* Metric cards — targets the current Streamlit testid (stMetric).
   Older Streamlit builds used "metric-container"; both are covered. */
div[data-testid="stMetric"], [data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 18px 22px !important;
    box-shadow: var(--shadow) !important;
    transition: transform 0.2s ease, border-color 0.2s ease !important;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    border-color: var(--border-light) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 12px !important;
}

.stButton button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 28px !important;
    transition: all 0.3s ease !important;
}
.stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(59,130,246,0.4) !important;
}
.stDownloadButton button {
    background: var(--bg-card2) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton button:hover {
    border-color: var(--accent-blue) !important;
    color: var(--accent-blue) !important;
}

.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-size: 13px !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--bg-card) !important;
    padding: 6px !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(59,130,246,0.15) !important;
    color: var(--accent-blue) !important;
}

.stAlert {
    border-radius: var(--radius) !important;
    border-left-width: 4px !important;
}
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(to right, transparent, var(--border), transparent) !important;
    margin: 24px 0 !important;
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────
def section_header(title: str, subtitle: str = "", accent: str = "#3b82f6") -> None:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px">
        <div style="width:4px;height:28px;background:{accent};border-radius:4px;flex-shrink:0;"></div>
        <div>
            <p style="margin:0;font-size:18px;font-weight:700;color:#e8eaf0;">{title}</p>
            {'<p style="margin:0;font-size:12px;color:#8b909e;margin-top:2px">'+subtitle+'</p>' if subtitle else ''}
        </div>
    </div>""", unsafe_allow_html=True)

def divider() -> None:
    st.markdown('<hr>', unsafe_allow_html=True)

def page_title(title: str, subtitle: str) -> None:
    st.markdown(f"""
    <h1 style="margin:0 0 4px;font-size:28px;font-weight:800;color:#e8eaf0;">{title}</h1>
    <p style="margin:0 0 20px;font-size:14px;color:#8b909e;">{subtitle}</p>""", unsafe_allow_html=True)

def demo_banner(missing: dict) -> None:
    """Shown on every page while running on the synthetic fallback dataset."""
    files = "".join(f"<code>{v}</code>&nbsp;&nbsp;" for v in missing.values())
    st.markdown(f"""
    <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.35);
                border-radius:12px;padding:12px 18px;margin-bottom:20px;display:flex;
                align-items:flex-start;gap:10px;font-size:12.5px;color:#f59e0b;line-height:1.5;">
        <span style="font-size:15px">🟡</span>
        <span><b>Demo Mode</b> — one or more expected files weren't found, so a synthetic
        dataset was generated and a model was trained live for this session, just so the
        dashboard is fully explorable. Add your real files at the paths below (or edit the
        constants at the top of the script) to switch to your live data automatically:
        <br>{files}</span>
    </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  PLOTLY THEME
# ──────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#8b909e", size=11),
    colorway=["#3b82f6", "#14b8a6", "#8b5cf6", "#f59e0b", "#ef4444", "#22c55e"],
    xaxis=dict(gridcolor="#1c2130", linecolor="#22273a", tickcolor="#22273a", zerolinecolor="#1c2130"),
    yaxis=dict(gridcolor="#1c2130", linecolor="#22273a", tickcolor="#22273a", zerolinecolor="#1c2130"),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#8b909e")),
    hoverlabel=dict(bgcolor="#161a24", bordercolor="#22273a", font_color="#e8eaf0", font_size=12),
)

def apply_plotly_theme(fig, title: str = "", height: int = 350) -> go.Figure:
    # NOTE: `weight` on a title font is only supported on newer Plotly
    # versions and raises a hard error on older ones. Bold via HTML tags
    # in the title text instead — that renders correctly on every version.
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(color="#e8eaf0", size=14),
            x=0, xanchor="left",
        ),
        height=height,
    )
    return fig

# ──────────────────────────────────────────────────────────
#  DATA / MODEL LOADERS  (your real pipeline output)
# ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_data():
    try:
        X_test = pd.read_csv(X_TEST_PATH)
        y_test = pd.read_csv(Y_TEST_PATH).values.ravel()
        X_train = pd.read_csv(X_TRAIN_PATH)
        return X_train, X_test, y_test
    except Exception:
        return None, None, None

@st.cache_data(show_spinner=False)
def load_results():
    try:
        comparison = pd.read_csv(COMPARISON_PATH)
        business = pd.read_csv(BUSINESS_PATH)
        try:
            importance = pd.read_csv(IMPORTANCE_PATH)
        except Exception:
            importance = None
        return comparison, business, importance
    except Exception:
        return None, None, None

def missing_paths() -> dict:
    required = {
        "Model": MODEL_PATH, "X_train": X_TRAIN_PATH, "X_test": X_TEST_PATH,
        "y_test": Y_TEST_PATH, "Model comparison": COMPARISON_PATH, "Business metrics": BUSINESS_PATH,
    }
    return {k: os.path.abspath(v) for k, v in required.items() if not os.path.exists(v)}

# ──────────────────────────────────────────────────────────
#  SYNTHETIC DEMO ENVIRONMENT (used only when real files are missing)
# ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def generate_demo_environment(n_samples: int = 2000, seed: int = 42):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n_samples, p=[0.55, 0.25, 0.20])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n_samples)
    internet = rng.choice(["DSL", "Fiber optic", "No"], n_samples, p=[0.35, 0.45, 0.20])
    tenure = rng.integers(0, 73, n_samples)
    monthly = np.round(rng.uniform(18, 120, n_samples), 2)
    total = np.round(tenure * monthly * rng.uniform(0.85, 1.05, n_samples), 2)

    services = {}
    for s in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        p_yes = rng.uniform(0.25, 0.55)
        yes_no = rng.choice(["Yes", "No"], n_samples, p=[p_yes, 1 - p_yes])
        services[s] = np.where(internet == "No", "No internet service", yes_no)

    df = pd.DataFrame({
        "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total,
        "Contract": contract, "PaymentMethod": payment, "InternetService": internet,
        **services,
    })

    # Latent risk signal: month-to-month / fiber / high bill / low tenure raise
    # churn risk; long contracts / long tenure lower it — mirrors real telco patterns.
    risk = (
        (contract == "Month-to-month") * 0.35 +
        (internet == "Fiber optic") * 0.20 +
        (monthly > 80) * 0.15 +
        (tenure < 12) * 0.25 -
        (contract == "Two year") * 0.30 -
        (tenure > 48) * 0.15
    )
    prob_churn = 1 / (1 + np.exp(-(risk * 4 - 2.3)))
    churn = (rng.uniform(0, 1, n_samples) < prob_churn).astype(int)

    enc_df = df.copy()
    for col, mapping in [
        ("Contract", {"Month-to-month": 0, "One year": 1, "Two year": 2}),
        ("PaymentMethod", {"Electronic check": 0, "Mailed check": 1, "Bank transfer": 2, "Credit card": 3}),
        ("InternetService", {"DSL": 0, "Fiber optic": 1, "No": 2}),
    ]:
        enc_df[col] = enc_df[col].map(mapping)
    for s in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        enc_df[s] = (enc_df[s] == "Yes").astype(int)

    X, y = enc_df, churn
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=seed, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # Real (not fabricated) comparison metrics — a couple of fast extra
    # models are trained on the same split purely for the benchmark table.
    models_to_compare = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": model,
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
    }
    rows = []
    for name, m in models_to_compare.items():
        t0 = time.time()
        if name != "Random Forest":
            m.fit(X_train, y_train)
        train_time = time.time() - t0
        p = m.predict(X_test)
        pr = m.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, p),
            "Precision": precision_score(y_test, p, zero_division=0),
            "Recall": recall_score(y_test, p, zero_division=0),
            "F1-Score": f1_score(y_test, p, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, pr),
            "Train_Time": round(train_time, 3),
        })
    comparison_df = pd.DataFrame(rows)

    importance_df = pd.DataFrame({
        "Feature": X_train.columns, "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    preds_test = model.predict(X_test)
    revenue_at_risk = float((X_test.loc[preds_test == 1, "MonthlyCharges"] * 12).sum())
    predicted_churners = int((preds_test == 1).sum())
    business_df = pd.DataFrame([{
        "Revenue_at_Risk": revenue_at_risk,
        "Potential_Revenue_Saved": revenue_at_risk * 0.35,
        "Predicted_Churners": predicted_churners,
    }])

    return model, X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_test, comparison_df, business_df, importance_df

# ──────────────────────────────────────────────────────────
#  ENCODING PIPELINE (for the manual "Predict Churn" form)
# ──────────────────────────────────────────────────────────
def encode_features(customer: dict) -> dict:
    enc = {}
    enc["Contract"] = {"Month-to-month": 0, "One year": 1, "Two year": 2}.get(
        customer.get("Contract", "Month-to-month"), 0
    )
    enc["PaymentMethod"] = {
        "Electronic check": 0, "Mailed check": 1, "Bank transfer": 2, "Credit card": 3
    }.get(customer.get("PaymentMethod", "Electronic check"), 0)
    enc["InternetService"] = {"DSL": 0, "Fiber optic": 1, "No": 2}.get(
        customer.get("InternetService", "DSL"), 0
    )
    for svc in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]:
        val = customer.get(svc, "No")
        enc[svc] = 1 if val == "Yes" else 0
    for num in ["tenure", "MonthlyCharges", "TotalCharges"]:
        enc[num] = float(customer.get(num, 0) or 0)
    return enc

def prepare_input(customer: dict, feature_names: list) -> pd.DataFrame:
    enc = encode_features(customer)
    df = pd.DataFrame([enc])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]

# ──────────────────────────────────────────────────────────
#  LOAD EVERYTHING (falls back to the demo environment if needed)
# ──────────────────────────────────────────────────────────
with st.spinner("🚀 Initialising Churn Intelligence Platform…"):
    model = load_model()
    X_train, X_test, y_test = load_data()
    comparison, business, importance = load_results()

    demo_mode = model is None or X_test is None or comparison is None or business is None
    missing = missing_paths() if demo_mode else {}

    if demo_mode:
        model, X_train, X_test, y_test, comparison, business, importance = generate_demo_environment()

model_ok = model is not None
data_ok = X_test is not None

# ──────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 20px">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">
            <div style="width:36px;height:36px;background:linear-gradient(135deg,#3b82f6,#8b5cf6);
                        border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px;">⚡</div>
            <div>
                <p style="margin:0;font-size:18px;font-weight:800;color:#e8eaf0;line-height:1.1;">ChurnIQ</p>
                <p style="margin:0;font-size:10px;color:#4a5168;letter-spacing:0.08em;font-weight:600">INTELLIGENCE PLATFORM</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📊 Overview", "🔍 Customer Analysis", "📈 Model Performance", "💼 Business Impact", "🔮 Predict Churn"],
        label_visibility="collapsed",
    )

    st.markdown('<div style="height:1px;background:linear-gradient(to right, transparent, #22273a, transparent);margin:16px 0"></div>', unsafe_allow_html=True)

    model_name = type(model).__name__ if model else "Not loaded"
    acc_str = ""
    if comparison is not None and not comparison.empty:
        try:
            acc_str = f"{comparison.iloc[0]['Accuracy']*100:.1f}%"
        except Exception:
            pass

    st.markdown(f"""
    <div style="background:linear-gradient(135deg, #11141c, #161a24);border:1px solid #22273a;
                border-radius:12px;padding:16px 18px;font-size:12px;">
        <p style="margin:0 0 12px;font-size:10px;text-transform:uppercase;letter-spacing:0.08em;
                  color:#4a5168;font-weight:700">System Status</p>
        <div style="display:flex;flex-direction:column;gap:8px">
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#8b909e;">Model</span>
                <span style="color:#e8eaf0;font-weight:600;">{model_name}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#8b909e;">Status</span>
                <span style="color:{'#22c55e' if model_ok else '#ef4444'};font-weight:700;">
                    {'● Active' if model_ok else '● Offline'}
                </span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#8b909e;">Data source</span>
                <span style="color:{'#f59e0b' if demo_mode else '#22c55e'};font-weight:700;">
                    {'● Demo' if demo_mode else '● Live'}
                </span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#8b909e;">Accuracy</span>
                <span style="color:#3b82f6;font-weight:700;">{acc_str or 'N/A'}</span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    if data_ok:
        st.markdown(f"""
        <p style="margin:12px 2px 0;font-size:11px;color:#4a5168;">
            {len(X_train):,} train rows · {len(X_test):,} test rows
        </p>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "📊 Overview":
    page_title("Customer Churn Dashboard", "Real-time churn intelligence · Live predictions · Business insights")
    if demo_mode:
        demo_banner(missing)

    if not model_ok or not data_ok:
        st.error("❌ Model or data files not found, and the demo fallback failed to initialise.")
        st.stop()

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Total Customers", f"{len(X_test):,}")
    with k2:
        acr = y_test.mean() * 100
        st.metric("Actual Churn", f"{acr:.1f}%")
    with k3:
        pcr = preds.mean() * 100
        st.metric("Predicted Churn", f"{pcr:.1f}%")
    with k4:
        high_risk = int((probs > 0.70).sum())
        st.metric("High-Risk (>70%)", f"{high_risk:,}")
    with k5:
        f1 = f1_score(y_test, preds)
        st.metric("F1 Score", f"{f1:.3f}")

    divider()

    c1, c2, c3 = st.columns([1.1, 1.1, 0.8])
    with c1:
        section_header("Churn Probability Distribution", "", "#3b82f6")
        bins = np.linspace(0, 1, 21)
        hist, edges = np.histogram(probs, bins=bins)
        mids = (edges[:-1] + edges[1:]) / 2
        colors = ["#22c55e" if m < 0.4 else "#f59e0b" if m < 0.7 else "#ef4444" for m in mids]
        fig = go.Figure(go.Bar(x=mids, y=hist, width=0.045, marker_color=colors,
                              hovertemplate="Prob: %{x:.2f}<br>Customers: %{y}<extra></extra>"))
        fig.add_vline(x=0.5, line=dict(color="#f59e0b", width=1.5, dash="dash"))
        fig.add_vline(x=0.7, line=dict(color="#ef4444", width=1.5, dash="dash"))
        apply_plotly_theme(fig, height=320)
        fig.update_layout(bargap=0.05, xaxis_title="Churn Probability", yaxis_title="# Customers")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        section_header("Actual vs Predicted", "", "#14b8a6")
        cat_labels = ["No Churn", "Churn"]
        actual_counts = [int((y_test == 0).sum()), int((y_test == 1).sum())]
        pred_counts = [int((preds == 0).sum()), int((preds == 1).sum())]
        fig2 = go.Figure()
        fig2.add_bar(name="Actual", x=cat_labels, y=actual_counts, marker_color=["#14b8a6", "#ef4444"])
        fig2.add_bar(name="Predicted", x=cat_labels, y=pred_counts, marker_color=["#3b82f6", "#f59e0b"])
        apply_plotly_theme(fig2, height=320)
        fig2.update_layout(barmode="group", yaxis_title="# Customers")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with c3:
        section_header("Risk Segments", "", "#8b5cf6")
        low = int((probs < 0.4).sum())
        medium = int(((probs >= 0.4) & (probs < 0.7)).sum())
        high = int((probs >= 0.7).sum())
        fig3 = go.Figure(go.Pie(
            labels=["Low (<40%)", "Medium (40–70%)", "High (>70%)"],
            values=[low, medium, high],
            hole=0.6,
            marker=dict(colors=["#22c55e", "#f59e0b", "#ef4444"], line=dict(color="#0a0c10", width=2)),
            textinfo="percent",
            textfont=dict(size=11, color="#e8eaf0"),
        ))
        apply_plotly_theme(fig3, height=320)
        fig3.update_layout(legend=dict(orientation="v", x=0.5, xanchor="center", y=-0.15, yanchor="top"))
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    if importance is not None and not importance.empty:
        divider()
        subtitle = "Random Forest feature importance (demo)" if demo_mode else "SHAP-based feature importance"
        section_header("Top Churn Drivers", subtitle, "#f59e0b")
        top10 = importance.head(10).sort_values("Importance")
        fig4 = go.Figure(go.Bar(
            x=top10["Importance"], y=top10["Feature"],
            orientation="h",
            marker=dict(color=top10["Importance"], colorscale=[[0, "#1c2130"], [0.5, "#3b82f6"], [1, "#8b5cf6"]],
                       showscale=False),
            text=top10["Importance"].round(3),
            textposition="outside",
            textfont=dict(size=11, color="#8b909e"),
        ))
        apply_plotly_theme(fig4, height=340)
        fig4.update_layout(xaxis_title="Importance Score")
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════
#  PAGE 2 — CUSTOMER ANALYSIS
# ══════════════════════════════════════════════════════════
elif page == "🔍 Customer Analysis":
    page_title("Customer Analysis", "Drill down into individual customer risk profiles")
    if demo_mode:
        demo_banner(missing)

    if not model_ok or not data_ok:
        st.error("❌ Model or data not loaded.")
        st.stop()

    if "cid" not in st.session_state:
        st.session_state.cid = 0

    all_probs = model.predict_proba(X_test)[:, 1]

    def _pick_random():
        st.session_state.cid = int(np.random.randint(0, len(X_test)))

    def _pick_from_top_risk():
        st.session_state.cid = st.session_state.cid_select

    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        mode = st.radio("Select by", ["Customer ID", "Highest Risk"], horizontal=True, label_visibility="collapsed")
        if mode == "Customer ID":
            st.number_input(f"Customer ID (0 – {len(X_test)-1})", 0, len(X_test) - 1, key="cid")
        else:
            top_ids = list(np.argsort(-all_probs)[:20])
            if st.session_state.cid not in top_ids:
                st.session_state.cid = top_ids[0]
            st.selectbox(
                "Top 20 highest-risk customers", options=top_ids,
                format_func=lambda x: f"Customer #{x} — {all_probs[x]*100:.1f}% risk",
                key="cid_select", index=top_ids.index(st.session_state.cid),
                on_change=_pick_from_top_risk,
            )
        st.button("🎲 Random Customer", use_container_width=True, on_click=_pick_random)

    cid = st.session_state.cid
    cdata = X_test.iloc[cid:cid + 1]

    try:
        prob = float(model.predict_proba(cdata)[0, 1])
        pred = int(model.predict(cdata)[0])
        conf = max(prob, 1 - prob)
    except Exception:
        prob, pred, conf = 0.5, 0, 0.5

    with col_info:
        risk_color = "#ef4444" if prob > 0.7 else "#f59e0b" if prob > 0.4 else "#22c55e"
        risk_label = "HIGH RISK" if prob > 0.7 else "MEDIUM RISK" if prob > 0.4 else "LOW RISK"
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, #11141c, #161a24);border:1px solid {risk_color}44;
                    border-left:4px solid {risk_color};border-radius:12px;padding:20px 24px;
                    display:flex;align-items:center;gap:24px;">
            <div style="text-align:center;min-width:100px">
                <div style="font-size:48px;font-weight:800;color:{risk_color};
                            font-family:'JetBrains Mono',monospace;line-height:1">{prob*100:.1f}%</div>
                <div style="font-size:10px;color:#4a5168;letter-spacing:0.08em;margin-top:4px;font-weight:600">CHURN PROB</div>
            </div>
            <div style="width:1px;height:70px;background:#22273a"></div>
            <div style="display:flex;flex-direction:column;gap:10px">
                <div>
                    <span style="background:{risk_color}22;color:{risk_color};border:1px solid {risk_color}44;
                          padding:4px 14px;border-radius:20px;font-size:11px;font-weight:700">{risk_label}</span>
                </div>
                <div style="font-size:15px;color:#e8eaf0;font-weight:500">
                    {'⚠️ Will likely churn' if pred == 1 else '✅ Will likely stay'}
                </div>
                <div style="font-size:12px;color:#8b909e">Confidence: <span style="color:#e8eaf0;font-weight:600">{conf*100:.1f}%</span></div>
            </div>
        </div>""", unsafe_allow_html=True)

    divider()

    t1, t2 = st.columns([1.5, 1])
    with t1:
        section_header("Feature Values", f"Customer #{cid}", "#3b82f6")
        fdf = cdata.T.reset_index()
        fdf.columns = ["Feature", "Value"]
        fdf["Value"] = fdf["Value"].round(4)
        st.dataframe(fdf, use_container_width=True, hide_index=True, height=360)

    with t2:
        section_header("Risk Gauge", "", "#8b5cf6")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(prob * 100, 1),
            number=dict(suffix="%", font=dict(size=36, color="#e8eaf0")),
            delta=dict(reference=50, increasing=dict(color="#ef4444"), decreasing=dict(color="#22c55e")),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#22273a", tickfont=dict(color="#8b909e", size=10)),
                bar=dict(color=risk_color, thickness=0.25),
                bgcolor="#161a24",
                borderwidth=0,
                steps=[
                    dict(range=[0, 40], color="rgba(34,197,94,0.15)"),
                    dict(range=[40, 70], color="rgba(245,158,11,0.15)"),
                    dict(range=[70, 100], color="rgba(239,68,68,0.15)"),
                ],
                threshold=dict(line=dict(color=risk_color, width=3), thickness=0.75, value=prob * 100),
            ),
        ))
        apply_plotly_theme(fig_g, height=280)
        st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})

        st.markdown("**Recommended Actions:**")
        if prob > 0.7:
            st.error("🚨 Immediate intervention required! Call customer within 24 hours.")
        elif prob > 0.4:
            st.warning("⚠️ Proactive engagement recommended. Send personalized email.")
        else:
            st.success("✅ Customer in good standing. Continue regular engagement.")

# ══════════════════════════════════════════════════════════
#  PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    page_title("Model Performance", "Accuracy, ROC curves, confusion matrix & model benchmarking")
    if demo_mode:
        demo_banner(missing)

    if not model_ok or not data_ok:
        st.error("❌ Model or data not loaded.")
        st.stop()

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    tab1, tab2, tab3 = st.tabs(["📊 Metrics Comparison", "📉 ROC & PR Curves", "🔲 Confusion Matrix"])

    with tab1:
        if comparison is not None and not comparison.empty:
            section_header("Model Comparison Matrix", "", "#3b82f6")
            display_df = comparison.copy()
            for c in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
                if c in display_df.columns:
                    display_df[c] = display_df[c].map(lambda x: f"{x*100:.2f}%" if x <= 1 else f"{x:.2f}%")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            divider()
            c1, c2 = st.columns(2)
            with c1:
                section_header("Metrics Radar", "Multi-metric comparison", "#8b5cf6")
                numeric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
                avail = [c for c in numeric_cols if c in comparison.columns]

                fig_radar = go.Figure()
                colors_r = ["#3b82f6", "#14b8a6", "#8b5cf6", "#f59e0b", "#ef4444"]

                # enumerate() gives a clean positional index for color-cycling —
                # using the dataframe's own index (as the original did) breaks
                # if that index isn't a plain 0..n-1 range (e.g. after filtering).
                for i, (_, row) in enumerate(comparison.iterrows()):
                    vals = [float(row[c]) for c in avail]
                    color = colors_r[i % len(colors_r)]
                    fillcolor_rgba = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)"
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=avail + [avail[0]],
                        fill="toself",
                        name=str(row.get("Model", "Model")),
                        line=dict(color=color),
                        fillcolor=fillcolor_rgba,
                    ))

                apply_plotly_theme(fig_radar, height=360)
                fig_radar.update_layout(polar=dict(
                    bgcolor="#161a24",
                    radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9, color="#4a5168"), gridcolor="#22273a"),
                    angularaxis=dict(tickfont=dict(size=10, color="#8b909e"), gridcolor="#22273a"),
                ))
                st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

            with c2:
                section_header("Training Time", "", "#f59e0b")
                if "Train_Time" in comparison.columns and "Model" in comparison.columns:
                    bar_colors = [colors_r[i % len(colors_r)] for i in range(len(comparison))]
                    fig_t = go.Figure(go.Bar(
                        x=comparison["Model"].astype(str),
                        y=comparison["Train_Time"],
                        marker_color=bar_colors,
                        text=comparison["Train_Time"].round(3).astype(str) + "s",
                        textposition="outside",
                        textfont=dict(size=11, color="#8b909e"),
                    ))
                    apply_plotly_theme(fig_t, height=360)
                    fig_t.update_layout(yaxis_title="Seconds")
                    st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar": False})
        else:
            st.warning("⚠️ No comparison data found.")

    with tab2:
        section_header("ROC Curve", "", "#14b8a6")
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.3f}",
                                    line=dict(color="#3b82f6", width=2.5),
                                    fill="tozeroy", fillcolor="rgba(59,130,246,0.08)"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                    line=dict(color="#4a5168", width=1.5, dash="dash")))
        apply_plotly_theme(fig_roc, height=380)
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})

        divider()
        section_header("Precision-Recall Curve", "", "#8b5cf6")
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc = auc(rec, prec)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"PR-AUC = {pr_auc:.3f}",
                                    line=dict(color="#8b5cf6", width=2.5),
                                    fill="tozeroy", fillcolor="rgba(139,92,246,0.08)"))
        apply_plotly_theme(fig_pr, height=340)
        fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig_pr, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        section_header("Confusion Matrix", "", "#f59e0b")
        cm = confusion_matrix(y_test, preds)
        labels = ["No Churn", "Churn"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            text=cm, texttemplate="%{text}",
            textfont=dict(size=20, color="#e8eaf0"),
            colorscale=[[0, "#161a24"], [0.5, "#1e3a5f"], [1, "#3b82f6"]],
            showscale=True,
            colorbar=dict(tickfont=dict(color="#8b909e"), outlinecolor="#22273a"),
        ))
        apply_plotly_theme(fig_cm, height=360)
        fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

        divider()
        section_header("Classification Report", "", "#22c55e")
        report = classification_report(y_test, preds, target_names=["No Churn", "Churn"], output_dict=True)
        rdf = pd.DataFrame(report).T.round(3).drop(columns=["support"], errors="ignore")
        st.dataframe(rdf, use_container_width=True)

# ══════════════════════════════════════════════════════════
#  PAGE 4 — BUSINESS IMPACT
# ══════════════════════════════════════════════════════════
elif page == "💼 Business Impact":
    page_title("Business Impact", "Revenue at risk · Retention ROI · Campaign planning")
    if demo_mode:
        demo_banner(missing)

    if business is not None and not business.empty:
        rev_risk = float(business["Revenue_at_Risk"].iloc[0])
        pot_savings = float(business["Potential_Revenue_Saved"].iloc[0])
        pred_churn = int(business["Predicted_Churners"].iloc[0])

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Revenue at Risk", f"${rev_risk:,.0f}")
        with k2:
            st.metric("Potential Savings", f"${pot_savings:,.0f}")
        with k3:
            st.metric("Predicted Churners", f"{pred_churn:,}")
        with k4:
            avg_clv = rev_risk / pred_churn if pred_churn > 0 else 0
            st.metric("Avg. CLV at Risk", f"${avg_clv:,.0f}")

        divider()
        section_header("ROI Calculator", "", "#3b82f6")
        c1, c2 = st.columns([1.2, 1])

        with c1:
            ret_rate = st.slider("Expected Retention Rate (%)", 5, 95, 30, 5)
            camp_cost = st.number_input("Campaign Cost ($)", 0, 500000, 10000, 1000)
            cost_per_c = st.number_input("Cost per Customer Contact ($)", 0, 500, 15, 5)

            total_campaign = camp_cost + (cost_per_c * pred_churn)
            expected_save = rev_risk * (ret_rate / 100)
            net_benefit = expected_save - total_campaign
            roi = (net_benefit / total_campaign * 100) if total_campaign > 0 else 0

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Expected Savings", f"${expected_save:,.0f}")
            with m2:
                st.metric("Total Campaign Cost", f"${total_campaign:,.0f}")
            with m3:
                st.metric("Net ROI", f"{roi:.1f}%")

        with c2:
            fig_roi = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(roi, 1),
                number=dict(suffix="%", font=dict(size=38, color="#e8eaf0")),
                delta=dict(reference=0, increasing=dict(color="#22c55e"), decreasing=dict(color="#ef4444")),
                title=dict(text="Return on Investment", font=dict(color="#8b909e", size=13)),
                gauge=dict(
                    axis=dict(range=[-50, 300], tickcolor="#22273a", tickfont=dict(color="#4a5168", size=9)),
                    bar=dict(color="#3b82f6" if roi > 0 else "#ef4444", thickness=0.25),
                    bgcolor="#161a24",
                    borderwidth=0,
                    steps=[
                        dict(range=[-50, 0], color="rgba(239,68,68,0.15)"),
                        dict(range=[0, 100], color="rgba(245,158,11,0.15)"),
                        dict(range=[100, 300], color="rgba(34,197,94,0.15)"),
                    ],
                ),
            ))
            apply_plotly_theme(fig_roi, height=300)
            st.plotly_chart(fig_roi, use_container_width=True, config={"displayModeBar": False})

        if data_ok and model_ok:
            divider()
            section_header("Retention Campaign Export", "Ready-to-use target list for outreach", "#22c55e")
            probs_all = model.predict_proba(X_test)[:, 1]
            risk_df = X_test.copy()
            risk_df.insert(0, "Customer_ID", risk_df.index)
            risk_df["Churn_Probability"] = probs_all
            risk_df["Risk_Segment"] = np.select(
                [probs_all >= 0.7, probs_all >= 0.4], ["High", "Medium"], default="Low"
            )
            high_risk_export = risk_df[risk_df["Risk_Segment"] == "High"].sort_values(
                "Churn_Probability", ascending=False
            )
            ec1, ec2 = st.columns([2, 1])
            with ec1:
                st.markdown(
                    f'<p style="color:#8b909e;font-size:13px;">'
                    f'{len(high_risk_export):,} customers currently flagged as <b style="color:#ef4444">High Risk</b> '
                    f'(&gt;70% churn probability) — export for your retention team.</p>',
                    unsafe_allow_html=True,
                )
            with ec2:
                st.download_button(
                    "📥 Download High-Risk List (CSV)",
                    data=high_risk_export.to_csv(index=False).encode("utf-8"),
                    file_name="high_risk_customers.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    else:
        st.warning("⚠️ Business metrics not found.")

# ══════════════════════════════════════════════════════════
#  PAGE 5 — PREDICT CHURN
# ══════════════════════════════════════════════════════════
else:
    page_title("Predict Churn", "Enter customer details for an instant churn probability estimate")
    if demo_mode:
        demo_banner(missing)

    if not model_ok or not data_ok:
        st.error("❌ Model not loaded.")
        st.stop()

    feature_names = X_test.columns.tolist()

    # Internet Service lives OUTSIDE the form on purpose: st.form only reruns
    # on submit, so a selectbox inside it can't reactively change what options
    # appear elsewhere (the add-on service list below depends on this value).
    # Keeping it here makes that dependency live and correct as you type.
    section_header("Plan & Contract", "", "#14b8a6")
    r0c1, r0c2, r0c3 = st.columns(3)
    with r0c1:
        contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    with r0c2:
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    with r0c3:
        internet_svc = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    divider()

    with st.form("churn_form", clear_on_submit=False):
        section_header("Subscription Details", "", "#3b82f6")
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            tenure = st.number_input("Tenure (months)", 0, 100, 12, 1)
        with r1c2:
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.5, 0.5)
        with r1c3:
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, round(tenure * monthly_charges, 2), 10.0)

        divider()
        section_header("Add-On Services", "", "#8b5cf6")
        svc_opts = ["Yes", "No"] if internet_svc != "No" else ["No internet service"]
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        with s1:
            online_sec = st.selectbox("Online Security", svc_opts)
        with s2:
            online_bk = st.selectbox("Online Backup", svc_opts)
        with s3:
            device_prot = st.selectbox("Device Protection", svc_opts)
        with s4:
            tech_sup = st.selectbox("Tech Support", svc_opts)
        with s5:
            stream_tv = st.selectbox("Streaming TV", svc_opts)
        with s6:
            stream_mv = st.selectbox("Streaming Movies", svc_opts)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡ Predict Churn Risk", use_container_width=True)

    if submitted:
        customer = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contract_type,
            "PaymentMethod": payment_method,
            "InternetService": internet_svc,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_bk,
            "DeviceProtection": device_prot,
            "TechSupport": tech_sup,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_mv,
        }

        try:
            input_df = prepare_input(customer, feature_names)
            prob = float(model.predict_proba(input_df)[0, 1])
            pred = int(model.predict(input_df)[0])
            conf = max(prob, 1 - prob)

            risk_color = "#ef4444" if prob > 0.7 else "#f59e0b" if prob > 0.4 else "#22c55e"
            risk_label = "HIGH RISK" if prob > 0.7 else "MEDIUM RISK" if prob > 0.4 else "LOW RISK"
            risk_emoji = "🚨" if prob > 0.7 else "⚠️" if prob > 0.4 else "✅"

            divider()
            section_header("Prediction Result", "", risk_color)

            st.markdown(f"""
            <div style="background:linear-gradient(135deg, #11141c, #161a24);border:1px solid {risk_color}55;
                        border-radius:16px;padding:28px 32px;display:flex;align-items:center;gap:32px;margin-bottom:20px;">
                <div style="text-align:center">
                    <div style="font-size:64px;font-weight:800;color:{risk_color};
                                font-family:'JetBrains Mono',monospace;line-height:1">{prob*100:.1f}%</div>
                    <div style="font-size:11px;color:#4a5168;letter-spacing:0.08em;margin-top:6px;font-weight:600">CHURN PROBABILITY</div>
                </div>
                <div style="width:1px;height:80px;background:#22273a"></div>
                <div style="display:flex;flex-direction:column;gap:12px">
                    <div>
                        <span style="background:{risk_color}22;color:{risk_color};border:1px solid {risk_color}44;
                              padding:4px 16px;border-radius:20px;font-size:13px;font-weight:700">{risk_label}</span>
                    </div>
                    <div style="font-size:16px;color:#e8eaf0;font-weight:600">{risk_emoji}  {'Will likely churn' if pred==1 else 'Will likely stay'}</div>
                    <div style="font-size:12px;color:#8b909e">Confidence: <span style="color:#e8eaf0;font-weight:600">{conf*100:.1f}%</span></div>
                </div>
            </div>""", unsafe_allow_html=True)

            ac1, ac2 = st.columns([1.2, 1])
            with ac1:
                section_header("Recommended Actions", "", risk_color)
                if prob > 0.7:
                    st.error("""
**Immediate Intervention Required**

1. 📞 Call customer within 24 hours
2. 🎁 Offer personalised retention discount (15–25% off)
3. 📦 Propose contract upgrade
4. 👥 Escalate to senior retention specialist
                    """)
                elif prob > 0.4:
                    st.warning("""
**Proactive Engagement Recommended**

1. 📧 Send personalised re-engagement email within 72h
2. 🎁 Offer small loyalty incentive
3. 📊 Monitor usage patterns for 2 weeks
                    """)
                else:
                    st.success("""
**Customer in Good Standing**

1. 🎉 Enrol in loyalty / rewards programme
2. 📧 Monthly satisfaction check-in
3. 📈 Upsell higher-value plan
                    """)

            with ac2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    number=dict(suffix="%", font=dict(size=34, color="#e8eaf0")),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickcolor="#22273a", tickfont=dict(color="#4a5168", size=9)),
                        bar=dict(color=risk_color, thickness=0.28),
                        bgcolor="#161a24",
                        borderwidth=0,
                        steps=[
                            dict(range=[0, 40], color="rgba(34,197,94,0.15)"),
                            dict(range=[40, 70], color="rgba(245,158,11,0.15)"),
                            dict(range=[70, 100], color="rgba(239,68,68,0.15)"),
                        ],
                    ),
                ))
                apply_plotly_theme(fig_gauge, height=260)
                st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

        except Exception as ex:
            st.error(f"❌ Prediction failed: {ex}")