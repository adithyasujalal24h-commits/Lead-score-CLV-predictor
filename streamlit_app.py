# professional_streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px

# ------------------ SET PAGE CONFIG ------------------
st.set_page_config(
    page_title="B2B AI Lead Scoring & CLV Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ DATA GENERATION ------------------
np.random.seed(42)
N = 500
data = pd.DataFrame({
    "annual_revenue": np.random.uniform(1, 100, N),
    "employee_count": np.random.randint(10, 5000, N),
    "engagement_score": np.random.randint(1, 10, N),
    "email_opens": np.random.randint(0, 20, N),
    "past_purchases": np.random.randint(0, 15, N),
})
data["converted"] = np.where(
    (data["engagement_score"] > 5) &
    (data["email_opens"] > 4) &
    (data["annual_revenue"] > 10), 1, 0
)
data["clv"] = (
    data["annual_revenue"]*5000 +
    data["past_purchases"]*2000 +
    data["engagement_score"]*3000 +
    np.random.randint(2000,25000,N)
)

X = data.drop(["converted","clv"], axis=1)
y_class = data["converted"]
y_reg = data["clv"]

X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2)

# ------------------ ML MODELS ------------------
lead_model = RandomForestClassifier(n_estimators=200)
lead_model.fit(X_train, y_train)

clv_model = RandomForestRegressor(n_estimators=200)
clv_model.fit(X_train_r, y_train_r)

# ------------------ SIDEBAR INPUT ------------------
st.sidebar.header("Enter Company Information")

annual_revenue = st.sidebar.number_input("Annual Revenue (Millions USD)", 0.0, 500.0, 10.0)
employee_count = st.sidebar.number_input("Employee Count", 1, 10000, 50)
engagement_score = st.sidebar.slider("Engagement Score (1-10)", 1, 10, 5)
email_opens = st.sidebar.number_input("Number of Email Opens", 0, 50, 5)
past_purchases = st.sidebar.number_input("Past Purchases", 0, 20, 2)

# ------------------ PREDICTION BUTTON ------------------
if st.sidebar.button("Predict Lead & CLV"):

    features = np.array([annual_revenue, employee_count, engagement_score, email_opens, past_purchases]).reshape(1,-1)
    
    lead_prob = round(lead_model.predict_proba(features)[0][1]*100,2)
    clv_value = round(clv_model.predict(features)[0],2)
    
    st.markdown("## Predicted Results")
    
    # ------------------ KPI CARDS ------------------
    col1, col2 = st.columns(2)
    col1.metric("Lead Conversion Probability", f"{lead_prob} %")
    col2.metric("Predicted CLV", f"${clv_value:,.2f}")
    
    # ------------------ GRAPHS ------------------
    st.markdown("### Visual Overview")
    metrics_df = pd.DataFrame({
        "Metric": ["Lead Probability", "CLV"],
        "Value": [lead_prob, clv_value]
    })
    
    fig_bar = px.bar(metrics_df, x="Metric", y="Value", text="Value",
                     color="Metric", color_discrete_map={"Lead Probability":"#1f77b4", "CLV":"#ff7f0e"})
    fig_bar.update_traces(texttemplate="%{text}", textposition="outside")
    fig_bar.update_layout(yaxis_title="", xaxis_title="", showlegend=False)
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # ------------------ ADDITIONAL INSIGHTS ------------------
    st.markdown("### Input Parameters Visualization")
    input_df = pd.DataFrame({
        "Feature": ["Annual Revenue", "Employee Count", "Engagement Score", "Email Opens", "Past Purchases"],
        "Value": [annual_revenue, employee_count, engagement_score, email_opens, past_purchases]
    })
    fig_input = px.pie(input_df, names="Feature", values="Value", title="Input Feature Contribution")
    st.plotly_chart(fig_input, use_container_width=True)
    
    st.markdown("---")
    st.markdown("**Note:** Lead probability is based on engagement and historical data patterns. CLV is predicted using company size, engagement, and past purchases.")

# ------------------ APP FOOTER ------------------
st.markdown("""
<div style="
    text-align: center;
    font-size: 0.8em;
    color: #6c757d;
    margin-top: 40px;
    padding: 10px;
    border-top: 1px solid #e6e6e6;
">
Created by <b>Adithya Sujalal</b> | <i>B2B AI Marketing App</i>
</div>
""", unsafe_allow_html=True)
