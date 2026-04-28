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

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main { background-color: #F8F7F4; }

    .block-container { padding: 2rem 2.5rem 2rem 2.5rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1A1A2E;
    }
    section[data-testid="stSidebar"] * { color: #E8E6E0 !important; }
    section[data-testid="stSidebar"] .stRadio > label { color: #A8A6A0 !important; }
    section[data-testid="stSidebar"] hr { border-color: #2E2E4E; }

    /* Metric cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E8E6E0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card .label {
        font-size: 12px;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 600;
        color: #1A1A2E;
        line-height: 1;
    }
    .metric-card .sub {
        font-size: 12px;
        color: #AAA;
        margin-top: 4px;
    }

    /* Risk badges */
    .badge-critical { background:#FDE8E8; color:#B91C1C; padding:3px 10px; border-radius:99px; font-size:12px; font-weight:600; }
    .badge-high     { background:#FEF3C7; color:#B45309; padding:3px 10px; border-radius:99px; font-size:12px; font-weight:600; }
    .badge-medium   { background:#DBEAFE; color:#1D4ED8; padding:3px 10px; border-radius:99px; font-size:12px; font-weight:600; }
    .badge-low      { background:#D1FAE5; color:#065F46; padding:3px 10px; border-radius:99px; font-size:12px; font-weight:600; }

    /* Section headers */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1A1A2E;
        border-left: 3px solid #E63946;
        padding-left: 10px;
        margin: 1.5rem 0 1rem 0;
    }

    /* Prediction result box */
    .result-box {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 2rem;
        border: 1px solid #E8E6E0;
        text-align: center;
    }
    .result-prob { font-size: 3.5rem; font-weight: 700; line-height: 1; margin-bottom: 6px; }
    .result-label { font-size: 1rem; color: #888; }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_raw_data():
    path = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    if not os.path.exists(path):
        st.error("Dataset not found. Place `WA_Fn-UseC_-HR-Employee-Attrition.csv` in the `data/` folder.")
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
    if not os.path.exists(mpath):
        return None, None
    return joblib.load(mpath), joblib.load(fpath)

def risk_badge(level):
    cls = f"badge-{level.lower()}"
    return f'<span class="{cls}">{level}</span>'

def risk_level(prob):
    if prob >= 0.75: return "Critical"
    if prob >= 0.60: return "High"
    if prob >= 0.40: return "Medium"
    return "Low"

def risk_color(prob):
    if prob >= 0.75: return "#E63946"
    if prob >= 0.60: return "#F4A261"
    if prob >= 0.40: return "#457B9D"
    return "#2A9D8F"

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 HR Attrition\n**People Analytics Dashboard**")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠  Overview",
        "🔍  Exploratory Analysis",
        "🤖  Predict Risk",
        "🚨  At-Risk Watch List",
    ])
    st.markdown("---")
    st.markdown("**Dataset**\nIBM HR Analytics\n1,470 employees · 35 features")
    st.markdown("**Model**\nRandom Forest Classifier\nSHAP Explainability")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    df = load_raw_data()

    st.markdown("# HR Attrition Analytics")
    st.markdown("Predict who is likely to leave — and understand why.")
    st.markdown("---")

    # KPI row
    total = len(df)
    left  = df['Attrition_Num'].sum()
    rate  = left / total * 100
    avg_income_left   = df[df['Attrition']=='Yes']['MonthlyIncome'].median()
    avg_income_stayed = df[df['Attrition']=='No']['MonthlyIncome'].median()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Total Employees</div>
            <div class="value">{total:,}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Employees Left</div>
            <div class="value" style="color:#E63946">{left}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Attrition Rate</div>
            <div class="value">{rate:.1f}%</div>
            <div class="sub">Company average</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        diff = avg_income_stayed - avg_income_left
        st.markdown(f"""<div class="metric-card">
            <div class="label">Income Gap</div>
            <div class="value">${diff:,.0f}</div>
            <div class="sub">Stayed vs Left (median)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Attrition by Department</div>', unsafe_allow_html=True)
        dept = df.groupby('Department')['Attrition_Num'].mean().sort_values() * 100
        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ['#E63946' if v == dept.max() else '#A8DADC' for v in dept.values]
        ax.barh(dept.index, dept.values, color=colors, height=0.5)
        ax.axvline(rate, color='#888', linestyle='--', linewidth=1, label=f'Avg {rate:.1f}%')
        ax.set_xlabel('Attrition Rate (%)')
        ax.legend(fontsize=9)
        for i, v in enumerate(dept.values):
            ax.text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=9, fontweight='500')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">Overtime Impact</div>', unsafe_allow_html=True)
        ot = df.groupby('OverTime')['Attrition_Num'].mean() * 100
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(ot.index, ot.values, color=['#2A9D8F', '#E63946'], width=0.4)
        ax.set_ylabel('Attrition Rate (%)')
        ax.set_ylim(0, 60)
        for bar, v in zip(bars, ot.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{v:.1f}%', ha='center', fontweight='600', fontsize=12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown('<div class="section-title">Top Findings</div>', unsafe_allow_html=True)
    findings = [
        ("🔴 Overtime", "Employees working overtime are **3× more likely** to leave"),
        ("🟠 Income Gap", f"Employees who left earned **${diff:,.0f} less** (median) than those who stayed"),
        ("🟡 Tenure Risk", "Highest attrition in the **first 2 years** of employment"),
        ("🟢 Sales Dept", "Sales department has the **highest attrition rate** across all departments"),
    ]
    fc1, fc2 = st.columns(2)
    for i, (title, body) in enumerate(findings):
        col = fc1 if i % 2 == 0 else fc2
        with col:
            st.info(f"**{title}** — {body}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Exploratory Analysis":
    df = load_raw_data()

    st.markdown("# Exploratory Analysis")
    st.markdown("Dig into the patterns behind employee attrition.")
    st.markdown("---")

    feature = st.selectbox("Explore attrition by:", [
        "JobRole", "MaritalStatus", "BusinessTravel",
        "EducationField", "Gender", "JobSatisfaction",
        "WorkLifeBalance", "EnvironmentSatisfaction"
    ])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f'<div class="section-title">Attrition Rate by {feature}</div>', unsafe_allow_html=True)
        grp = df.groupby(feature)['Attrition_Num'].mean().sort_values() * 100
        fig, ax = plt.subplots(figsize=(6, max(3, len(grp) * 0.55)))
        colors = ['#E63946' if v >= grp.mean() else '#A8DADC' for v in grp.values]
        ax.barh(grp.index.astype(str), grp.values, color=colors, height=0.55)
        ax.axvline(grp.mean(), color='#888', linestyle='--', linewidth=1)
        ax.set_xlabel('Attrition Rate (%)')
        for i, v in enumerate(grp.values):
            ax.text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">Income Distribution: Stayed vs Left</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        for attr, color, label in [('No', '#2A9D8F', 'Stayed'), ('Yes', '#E63946', 'Left')]:
            df[df['Attrition'] == attr]['MonthlyIncome'].plot.kde(
                ax=ax, color=color, label=label, linewidth=2.5)
        ax.set_xlabel('Monthly Income ($)')
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown('<div class="section-title">Age & Tenure vs Attrition</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for attr, color, label in [('No', '#2A9D8F', 'Stayed'), ('Yes', '#E63946', 'Left')]:
            df[df['Attrition'] == attr]['Age'].plot.kde(ax=ax, color=color, label=label, linewidth=2)
        ax.set_xlabel('Age')
        ax.legend()
        ax.set_title('Age Distribution')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for attr, color, label in [('No', '#2A9D8F', 'Stayed'), ('Yes', '#E63946', 'Left')]:
            df[df['Attrition'] == attr]['YearsAtCompany'].plot.kde(ax=ax, color=color, label=label, linewidth=2)
        ax.set_xlabel('Years at Company')
        ax.legend()
        ax.set_title('Tenure Distribution')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    num_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears',
                'YearsSinceLastPromotion', 'JobSatisfaction', 'WorkLifeBalance',
                'DistanceFromHome', 'NumCompaniesWorked', 'Attrition_Num']
    fig, ax = plt.subplots(figsize=(10, 6))
    mask = np.triu(np.ones(len(num_cols), dtype=bool))
    sns.heatmap(df[num_cols].corr(), mask=mask, annot=True, fmt='.2f',
                cmap='RdYlGn', center=0, ax=ax, linewidths=0.5, cbar_kws={'shrink': 0.7})
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT RISK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Predict Risk":
    st.markdown("# Predict Employee Attrition Risk")
    st.markdown("Fill in employee details to get a real-time risk score.")
    st.markdown("---")

    model, feature_names = load_model()

    if model is None:
        st.warning("⚠️ Model not found. Run `03_model_building.ipynb` first to train and save the model.")
        st.info("Place the saved `rf_attrition_model.pkl` and `feature_names.pkl` files in the `models/` folder.")
        st.stop()

    with st.form("predict_form"):
        st.markdown('<div class="section-title">Employee Profile</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.slider("Age", 18, 60, 30)
            monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 3500, step=500)
            years_at_company = st.slider("Years at Company", 0, 40, 3)
            years_since_promo = st.slider("Years Since Last Promotion", 0, 15, 2)

        with c2:
            overtime = st.selectbox("Works Overtime?", ["No", "Yes"])
            job_satisfaction = st.select_slider("Job Satisfaction", [1, 2, 3, 4],
                                                 format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
            work_life_balance = st.select_slider("Work-Life Balance", [1, 2, 3, 4],
                                                  format_func=lambda x: {1:"Bad",2:"Good",3:"Better",4:"Best"}[x])
            environment_sat = st.select_slider("Environment Satisfaction", [1, 2, 3, 4],
                                                format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x])

        with c3:
            distance = st.slider("Distance From Home (km)", 1, 30, 5)
            num_companies = st.slider("Num Companies Worked", 0, 9, 1)
            business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
            total_working_years = st.slider("Total Working Years", 0, 40, 5)

        submitted = st.form_submit_button("🔍 Calculate Risk Score", use_container_width=True)

    if submitted:
        input_data = {
            'Age': age,
            'MonthlyIncome': monthly_income,
            'YearsAtCompany': years_at_company,
            'YearsSinceLastPromotion': years_since_promo,
            'OverTime': 1 if overtime == "Yes" else 0,
            'JobSatisfaction': job_satisfaction,
            'WorkLifeBalance': work_life_balance,
            'EnvironmentSatisfaction': environment_sat,
            'DistanceFromHome': distance,
            'NumCompaniesWorked': num_companies,
            'BusinessTravel': {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}[business_travel],
            'TotalWorkingYears': total_working_years,
        }

        emp_df = pd.DataFrame([input_data]).reindex(columns=feature_names, fill_value=0)
        prob = model.predict_proba(emp_df)[0][1]
        level = risk_level(prob)
        color = risk_color(prob)

        st.markdown("---")
        rc1, rc2, rc3 = st.columns([1, 1, 1])

        with rc2:
            st.markdown(f"""
            <div class="result-box">
                <div class="result-prob" style="color:{color}">{prob*100:.1f}%</div>
                <div class="result-label">Attrition Probability</div>
                <br>
                {risk_badge(level)}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Risk Factors Identified</div>', unsafe_allow_html=True)
        flags = []
        if overtime == "Yes":         flags.append(("🔴 Overtime", "Working overtime significantly increases attrition risk"))
        if monthly_income < 3000:     flags.append(("🔴 Low Income", f"Monthly income of ${monthly_income:,} is below the risk threshold"))
        if years_at_company <= 2:     flags.append(("🟠 Early Tenure", "Employees in their first 2 years are highest-risk"))
        if job_satisfaction <= 2:     flags.append(("🟠 Low Satisfaction", "Low job satisfaction correlates strongly with attrition"))
        if work_life_balance <= 2:    flags.append(("🟡 Work-Life Balance", "Poor work-life balance increases attrition probability"))
        if distance > 20:             flags.append(("🟡 Long Commute", "Large distance from home adds attrition pressure"))
        if years_since_promo >= 4:    flags.append(("🟡 No Promotion", "4+ years without promotion suggests stagnation"))
        if business_travel == "Travel_Frequently": flags.append(("🟡 Frequent Travel", "Frequent travel adds stress and attrition risk"))

        if flags:
            for title, desc in flags:
                st.warning(f"**{title}** — {desc}")
        else:
            st.success("✅ No major risk factors detected for this employee profile.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — WATCH LIST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨  At-Risk Watch List":
    st.markdown("# At-Risk Employee Watch List")
    st.markdown("Employees ranked by predicted probability of leaving.")
    st.markdown("---")

    model, feature_names = load_model()
    df_raw = load_raw_data()
    df_clean = load_cleaned_data()

    if model is None or df_clean is None:
        st.warning("⚠️ Model or cleaned data not found. Run notebooks 01 and 03 first.")
        st.stop()
    # Convert input to DataFrame
     X = pd.DataFrame([input_data])

     # Match training features
     X = X.reindex(columns=feature_names, fill_value=0)

    # Predict probability
    probs = model.predict_proba(X)[:, 1]

    watch = df_raw[['Age', 'Department', 'JobRole', 'MonthlyIncome',
                    'YearsAtCompany', 'OverTime', 'JobSatisfaction', 'Attrition']].copy()
    watch['Attrition_Probability'] = (probs * 100).round(1)
    watch['Risk_Level'] = [risk_level(p) for p in probs]

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        dept_filter = st.multiselect("Department", df_raw['Department'].unique(),
                                      default=list(df_raw['Department'].unique()))
    with fc2:
        risk_filter = st.multiselect("Risk Level", ["Critical", "High", "Medium", "Low"],
                                      default=["Critical", "High"])
    with fc3:
        threshold = st.slider("Min Probability (%)", 0, 100, 60)

    filtered = watch[
        (watch['Department'].isin(dept_filter)) &
        (watch['Risk_Level'].isin(risk_filter)) &
        (watch['Attrition_Probability'] >= threshold)
    ].sort_values('Attrition_Probability', ascending=False).reset_index(drop=False)
    filtered.rename(columns={'index': 'Employee_ID'}, inplace=True)

    # Summary row
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Employees Flagged</div>
            <div class="value">{len(filtered)}</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        critical = len(filtered[filtered['Risk_Level'] == 'Critical'])
        st.markdown(f"""<div class="metric-card">
            <div class="label">Critical Risk</div>
            <div class="value" style="color:#E63946">{critical}</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        high = len(filtered[filtered['Risk_Level'] == 'High'])
        st.markdown(f"""<div class="metric-card">
            <div class="label">High Risk</div>
            <div class="value" style="color:#F4A261">{high}</div>
        </div>""", unsafe_allow_html=True)
    with s4:
        avg_prob = filtered['Attrition_Probability'].mean() if len(filtered) > 0 else 0
        st.markdown(f"""<div class="metric-card">
            <div class="label">Avg Probability</div>
            <div class="value">{avg_prob:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    if len(filtered) == 0:
        st.info("No employees match the current filters.")
    else:
        display_cols = ['Employee_ID', 'Age', 'Department', 'JobRole',
                        'MonthlyIncome', 'YearsAtCompany', 'OverTime',
                        'Attrition_Probability', 'Risk_Level', 'Attrition']
        st.dataframe(
            filtered[display_cols].style
                .background_gradient(subset=['Attrition_Probability'], cmap='RdYlGn_r')
                .format({'Attrition_Probability': '{:.1f}%',
                         'MonthlyIncome': '${:,.0f}'}),
            use_container_width=True,
            height=420
        )

        csv = filtered[display_cols].to_csv(index=False)
        st.download_button(
            "⬇️ Download Watch List CSV",
            data=csv,
            file_name="at_risk_employees.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown("---")
        st.markdown('<div class="section-title">Risk Distribution by Department</div>', unsafe_allow_html=True)
        dept_risk = watch[watch['Risk_Level'].isin(['Critical', 'High'])]\
            .groupby('Department')['Risk_Level'].value_counts().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        dept_risk.plot(kind='bar', ax=ax,
                       color=['#E63946', '#F4A261'],
                       edgecolor='white', width=0.5)
        ax.set_xlabel('')
        ax.set_xticklabels(dept_risk.index, rotation=0)
        ax.legend(title='Risk Level')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
