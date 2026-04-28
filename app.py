import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_raw_data():
    path = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    if not os.path.exists(path):
        st.error("Dataset not found. Place file in data/ folder.")
        st.stop()
    df = pd.read_csv(path)
    df['Attrition_Num'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    return df


@st.cache_data
def load_cleaned_data():
    path = "data/cleaned_hr_data.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_model():
    mpath = "models/rf_attrition_model.pkl"
    fpath = "models/feature_names.pkl"

    if not os.path.exists(mpath) or not os.path.exists(fpath):
        return None, None

    model = joblib.load(mpath)
    feature_names = joblib.load(fpath)

    return model, feature_names


def risk_level(prob):
    if prob >= 0.75:
        return "Critical"
    elif prob >= 0.60:
        return "High"
    elif prob >= 0.40:
        return "Medium"
    else:
        return "Low"


# Sidebar Navigation
with st.sidebar:
    st.title("📊 HR Attrition Dashboard")

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Exploratory Analysis",
            "Predict Risk",
            "At-Risk Watch List"
        ]
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":

    df = load_raw_data()

    st.title("HR Attrition Overview")

    total = len(df)
    left = df['Attrition_Num'].sum()
    rate = left / total * 100

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Employees", total)
    col2.metric("Employees Left", left)
    col3.metric("Attrition Rate", f"{rate:.1f}%")

    st.subheader("Attrition by Department")

    dept = (
        df.groupby('Department')['Attrition_Num']
        .mean()
        .sort_values()
        * 100
    )

    fig, ax = plt.subplots()
    ax.barh(dept.index, dept.values)
    ax.set_xlabel("Attrition Rate (%)")
    st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Exploratory Analysis":

    df = load_raw_data()

    st.title("Exploratory Data Analysis")

    feature = st.selectbox(
        "Select Feature",
        [
            "JobRole",
            "MaritalStatus",
            "BusinessTravel",
            "EducationField",
            "Gender"
        ]
    )

    grp = (
        df.groupby(feature)['Attrition_Num']
        .mean()
        .sort_values()
        * 100
    )

    fig, ax = plt.subplots()
    ax.barh(grp.index.astype(str), grp.values)
    ax.set_xlabel("Attrition Rate (%)")
    st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT RISK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict Risk":

    st.title("Predict Employee Attrition Risk")

    model, feature_names = load_model()

    if model is None:
        st.warning("Model not found. Train model first.")
        st.stop()

    with st.form("predict_form"):

        age = st.slider("Age", 18, 60, 30)
        income = st.number_input("Monthly Income", 1000, 20000, 3500)
        years = st.slider("Years at Company", 0, 40, 3)
        overtime = st.selectbox("Overtime", ["No", "Yes"])

        submit = st.form_submit_button("Predict")

    if submit:

        input_data = {
            "Age": age,
            "MonthlyIncome": income,
            "YearsAtCompany": years,
            "OverTime": 1 if overtime == "Yes" else 0
        }

        X = pd.DataFrame([input_data])

        X = X.reindex(columns=feature_names, fill_value=0)

        prob = model.predict_proba(X)[0][1]

        st.success(f"Attrition Probability: {prob * 100:.1f}%")
        st.info(f"Risk Level: {risk_level(prob)}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — WATCH LIST (FIXED)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "At-Risk Watch List":

    st.title("At-Risk Employee Watch List")

    model, feature_names = load_model()
    df_raw = load_raw_data()
    df_clean = load_cleaned_data()

    if model is None or df_clean is None:
        st.warning("Model or cleaned data not found. Run notebooks 01 and 03 first.")
        st.stop()

    # Prepare features
    if 'Attrition' in df_clean.columns:
        X = df_clean.drop('Attrition', axis=1)
    else:
        X = df_clean.copy()

    # Match training features
    X = X.reindex(columns=feature_names, fill_value=0)

    # Predict probabilities for ALL employees
    probs = model.predict_proba(X)[:, 1]

    watch = df_raw[[
        'Age',
        'Department',
        'JobRole',
        'MonthlyIncome',
        'YearsAtCompany',
        'OverTime',
        'JobSatisfaction',
        'Attrition'
    ]].copy()

    watch['Attrition_Probability'] = (probs * 100).round(1)
    watch['Risk_Level'] = [risk_level(p) for p in probs]

    threshold = st.slider(
        "Minimum Risk Probability (%)",
        0,
        100,
        60
    )

    filtered = (
        watch[
            watch['Attrition_Probability'] >= threshold
        ]
        .sort_values(
            'Attrition_Probability',
            ascending=False
        )
        .reset_index(drop=True)
    )

    st.metric(
        "Employees Flagged",
        len(filtered)
    )

    if len(filtered) == 0:
        st.info("No employees match the current filter.")
    else:

        st.dataframe(
            filtered.style
            .background_gradient(
                subset=['Attrition_Probability'],
                cmap='RdYlGn_r'
            )
            .format({
                'Attrition_Probability': '{:.1f}%'
            }),
            use_container_width=True
        )

        csv = filtered.to_csv(index=False)

        st.download_button(
            "Download Watch List CSV",
            data=csv,
            file_name="at_risk_employees.csv",
            mime="text/csv"
        )
