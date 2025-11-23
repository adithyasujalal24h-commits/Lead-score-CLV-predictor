# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# ------------------ DATA ------------------
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

# ------------------ MODELS ------------------
lead_model = RandomForestClassifier(n_estimators=200)
lead_model.fit(X_train, y_train)

clv_model = RandomForestRegressor(n_estimators=200)
clv_model.fit(X_train_r, y_train_r)

# ------------------ STREAMLIT UI ------------------
st.title("AI B2B Lead Scoring & CLV Prediction")

st.sidebar.header("Enter Company Data")
annual_revenue = st.sidebar.number_input("Annual Revenue (Millions)", min_value=0.0, value=10.0)
employee_count = st.sidebar.number_input("Employee Count", min_value=1, value=50)
engagement_score = st.sidebar.slider("Engagement Score (1-10)", 1, 10, 5)
email_opens = st.sidebar.number_input("Email Opens", min_value=0, value=5)
past_purchases = st.sidebar.number_input("Past Purchases", min_value=0, value=2)

if st.button("Predict"):
    features = np.array([annual_revenue, employee_count, engagement_score, email_opens, past_purchases]).reshape(1,-1)
    lead_prob = round(lead_model.predict_proba(features)[0][1]*100,2)
    clv_value = round(clv_model.predict(features)[0],2)
    
    st.success(f"Lead Conversion Probability: {lead_prob}%")
    st.success(f"Predicted Customer Lifetime Value (CLV): ${clv_value}")
