# base of the app
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Intelligence Dashboard", layout="wide")

st.title("💳 Credit Card Churn Intelligence System")

# Load model
model = joblib.load("model.pkl")

# Load data
df = pd.read_csv("../dataset/processed/credit_card_features.csv")

# Feature list
features = [
    "Transaction_Velocity",
    "Engagement",
    "Avg_Utilization_Ratio",
    "Months_Inactive_12_mon",
    "Transaction_Drift",
    "Spend_Drift",
    "Engagement_Drift",
    "Behavioral_Risk_Score"
]

X = df[features]

# Predictions
df["Churn_Prob"] = model.predict_proba(X)[:,1]

# sidebars
st.sidebar.header("Filters")

risk_filter = st.sidebar.selectbox(
    "Select Risk Level",
    ["All", "High Risk", "Medium Risk", "Low Risk"]
)
# RISK SEGMENTATION
df["Risk_Segment"] = pd.cut(
    df["Churn_Prob"],
    bins=[0, 0.3, 0.6, 1],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

if risk_filter != "All":
    df = df[df["Risk_Segment"] == risk_filter]

# KPI METRICS
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Avg Churn Risk", round(df["Churn_Prob"].mean(), 2))
col3.metric("High Risk Customers", (df["Risk_Segment"] == "High Risk").sum())

#CHURN DISTRIBUTION
st.subheader("Churn Probability Distribution")

fig, ax = plt.subplots()
ax.hist(df["Churn_Prob"], bins=30)
st.pyplot(fig)

# DRIFT vs CHURN
st.subheader("Behavioral Drift vs Churn Risk")

fig, ax = plt.subplots()
ax.scatter(df["Transaction_Drift"], df["Churn_Prob"], alpha=0.5)
ax.set_xlabel("Transaction Drift")
ax.set_ylabel("Churn Probability")

st.pyplot(fig)

#SHAP EXPLAINABILITY
st.subheader("Model Explainability (SHAP)")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Handle shape
if isinstance(shap_values, list):
    shap_to_plot = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_to_plot = shap_values[:, :, 1]
else:
    shap_to_plot = shap_values

fig, ax = plt.subplots()
shap.summary_plot(shap_to_plot, X, show=False)
st.pyplot(fig)

# INDIVIDUAL CUSTOMER INSIGHT

st.subheader("Customer-Level Explanation")

idx = st.slider("Select Customer Index", 0, len(df)-1, 0)

st.write(df.iloc[idx])

shap.plots.waterfall(
    shap.Explanation(
        values=shap_to_plot[idx],
        base_values=explainer.expected_value[1],
        data=X.iloc[idx],
        feature_names=X.columns
    )
)