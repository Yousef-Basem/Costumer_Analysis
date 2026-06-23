import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

st.title("📡 Telco Customer Churn — Prediction App")
st.markdown("Enter customer details to get predictions from 2 pre-trained models.")
st.markdown("---")

# ── Load pre-trained models (trained once, in the notebook) ─────────────────
# These .pkl files must sit in the same folder as this script.
MODEL_FILES = {
    "log_model": BASE_DIR / "log_model.pkl",
    "lr_tc": BASE_DIR / "lr_tc.pkl",
    "sc_cls": BASE_DIR / "sc_cls.pkl",
    "sc_target": BASE_DIR / "sc_target.pkl",
    "sc_tc": BASE_DIR / "sc_tc.pkl",
}

@st.cache_resource
def load_models():
    import joblib
    import os

    missing = [f for f in MODEL_FILES.values() if not os.path.exists(f)]
    if missing:
        st.error(
            "Missing model file(s): " + ", ".join(missing) +
            ". Make sure all .pkl files from the training notebook are in this folder."
        )
        st.stop()

    models = {name: joblib.load(path) for name, path in MODEL_FILES.items()}
    return models


@st.cache_data
def load_reference_data(_sc_target):
    # Used only for the comparison charts (dataset distributions), not for training.
    # tenure / MonthlyCharges / TotalCharges in this file are scaled 0-1 (done in the EDA notebook),
    # so we inverse-transform them once here to show charts in real-world units.
    df = pd.read_csv(BASE_DIR / "telco_ml_ready.csv")
    real_vals = _sc_target.inverse_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])
    df['tenure_real'] = real_vals[:, 0]
    df['MonthlyCharges_real'] = real_vals[:, 1]
    df['TotalCharges_real'] = real_vals[:, 2]
    return df


models = load_models()
log_model = models["log_model"]
lr_tc     = models["lr_tc"]
sc_cls    = models["sc_cls"]
sc_tc     = models["sc_tc"]
sc_target = models["sc_target"]

df_ref = load_reference_data(sc_target)

# The exact one-hot column layout used when these models were trained.
# This MUST match the column order produced by pd.get_dummies in the notebook.
ALL_COLUMNS = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn',
    'gender_Female', 'gender_Male',
    'SeniorCitizen_No', 'SeniorCitizen_Yes',
    'Partner_No', 'Partner_Yes',
    'Dependents_No', 'Dependents_Yes',
    'PhoneService_No', 'PhoneService_Yes',
    'MultipleLines_No', 'MultipleLines_Yes',
    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No', 'OnlineSecurity_Yes',
    'OnlineBackup_No', 'OnlineBackup_Yes',
    'DeviceProtection_No', 'DeviceProtection_Yes',
    'TechSupport_No', 'TechSupport_Yes',
    'StreamingTV_No', 'StreamingTV_Yes',
    'StreamingMovies_No', 'StreamingMovies_Yes',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_No', 'PaperlessBilling_Yes',
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
]

CLS_COLS = [c for c in ALL_COLUMNS if c != 'Churn']
TC_COLS  = [c for c in ALL_COLUMNS if c != 'TotalCharges']

# ── Sidebar Inputs ────────────────────────────────────────────────────────────
st.sidebar.header("👤 Customer Profile")

gender         = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior         = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner        = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents     = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])
tenure         = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone          = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes"])
internet       = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_sec     = st.sidebar.selectbox("Online Security", ["No", "Yes"])
online_bkp     = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
device_prot    = st.sidebar.selectbox("Device Protection", ["No", "Yes"])
tech_support   = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
streaming_tv   = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
streaming_mv   = st.sidebar.selectbox("Streaming Movies", ["No", "Yes"])
contract       = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless      = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment        = st.sidebar.selectbox("Payment Method", [
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
])
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
total_charges   = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, 1500.0)

predict_clicked = st.sidebar.button("🔮 Predict", type="primary", use_container_width=True)


# ── Build a one-hot input row matching training-time columns ────────────────
def build_input_row():
    row = {col: False for col in ALL_COLUMNS}
    row['tenure'] = tenure
    row['MonthlyCharges'] = monthly_charges
    row['TotalCharges'] = total_charges
    row['Churn'] = 0  # placeholder, dropped before each model uses it

    row[f'gender_{gender}'] = True
    row[f'SeniorCitizen_{senior}'] = True
    row[f'Partner_{partner}'] = True
    row[f'Dependents_{dependents}'] = True
    row[f'PhoneService_{phone}'] = True
    row[f'MultipleLines_{multiple_lines}'] = True
    row[f'InternetService_{internet}'] = True
    row[f'OnlineSecurity_{online_sec}'] = True
    row[f'OnlineBackup_{online_bkp}'] = True
    row[f'DeviceProtection_{device_prot}'] = True
    row[f'TechSupport_{tech_support}'] = True
    row[f'StreamingTV_{streaming_tv}'] = True
    row[f'StreamingMovies_{streaming_mv}'] = True
    row[f'Contract_{contract}'] = True
    row[f'PaperlessBilling_{paperless}'] = True
    row[f'PaymentMethod_{payment}'] = True

    return pd.DataFrame([row])[ALL_COLUMNS]


def render_card(title, value_html, subtitle, color):
    st.markdown(f"""
    <div style='background-color:{color}22; border-left: 5px solid {color};
                padding:20px; border-radius:8px;'>
        <h4 style='color:{color}'>{title}</h4>
        <h2 style='color:{color}'>{value_html}</h2>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# sc_target was fit on 3 columns together: [tenure, MonthlyCharges, TotalCharges].
# To undo scaling for just one predicted value, we must pass all 3 columns at once
# (using the user's real-world inputs for the other two), then pull out the one we need.
TENURE_IDX, MONTHLY_IDX, TOTAL_IDX = 0, 1, 2


def unscale_total_charges(tc_pred_scaled, tenure_real, monthly_real):
    """Convert a scaled TotalCharges prediction back to real dollars."""
    scaled_inputs = sc_target.transform([[tenure_real, monthly_real, 0]])[0]
    monthly_scaled = scaled_inputs[MONTHLY_IDX]
    row = [0, monthly_scaled, tc_pred_scaled]
    real = sc_target.inverse_transform([row])[0]
    return real[TOTAL_IDX]


# ── Predictions ───────────────────────────────────────────────────────────────
if not predict_clicked and "has_predicted" not in st.session_state:
    st.info("Set the customer profile in the sidebar, then click **Predict**.")
    st.stop()

if predict_clicked:
    st.session_state["has_predicted"] = True

input_df = build_input_row()

X_cls_input = input_df[CLS_COLS]
X_cls_scaled = sc_cls.transform(X_cls_input)
churn_pred = log_model.predict(X_cls_scaled)[0]
churn_prob = log_model.predict_proba(X_cls_scaled)[0]

X_tc_input = input_df[TC_COLS]
X_tc_scaled = sc_tc.transform(X_tc_input)
tc_pred_scaled = lr_tc.predict(X_tc_scaled)[0]
tc_real = unscale_total_charges(tc_pred_scaled, tenure, monthly_charges)

# ── Results Section ───────────────────────────────────────────────────────────
st.subheader("🔮 Prediction Results")

col1, col2 = st.columns(2)

with col1:
    churn_label = "🔴 Will Churn" if churn_pred == 1 else "🟢 Will NOT Churn"
    color = "#ff4b4b" if churn_pred == 1 else "#21c354"
    render_card(
        "Model 1 — Churn", churn_label, f"Churn probability: <b>{churn_prob[1]*100:.1f}%</b>", color
    )

with col2:
    render_card(
        "Model 2 — Total Charges", f"${max(tc_real, 0):,.0f}", "Predicted lifetime spend", "#1e88e5"
    )

st.markdown("---")

# ── Charts Section ────────────────────────────────────────────────────────────
st.subheader("📊 Visual Analysis")

chart1, chart2 = st.columns(2)

with chart1:
    st.markdown("**Churn Probability**")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        [churn_prob[1], churn_prob[0]],
        labels=["Churn", "No Churn"],
        autopct='%1.1f%%',
        startangle=90,
        colors=["darkred", "steelblue"],
    )
    ax.set_title("Churn vs No Churn")
    st.pyplot(fig)
    plt.close(fig)

with chart2:
    st.markdown("**Monthly Charges vs Dataset**")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.hist(df_ref['MonthlyCharges_real'], bins=30, color='steelblue', alpha=0.7, label='All Customers')
    ax.axvline(monthly_charges, color='darkred', linewidth=2, linestyle='--', label='This Customer')
    ax.set_title("Monthly Charges Distribution")
    ax.set_xlabel("Monthly Charges ($)")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")

# ── Feature importance for churn model ───────────────────────────────────────
st.subheader("📌 What Influences Churn Most?")
coef_df = pd.DataFrame({
    'Feature': CLS_COLS,
    'Coefficient': log_model.coef_[0],
}).sort_values('Coefficient', key=abs, ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['darkred' if c > 0 else 'steelblue' for c in coef_df['Coefficient']]
ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
ax.set_title('Top 10 Features Driving Churn (Logistic Regression)')
ax.set_xlabel('Coefficient Value')
ax.invert_yaxis()
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown("---")
st.caption("Built with Streamlit · Telco Customer Churn Dataset · Models loaded from pre-trained .pkl files")
