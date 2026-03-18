# 💳 Credit Card Lifecycle Analytics Engine

An end-to-end **FinTech Analytics & Machine Learning project** designed to analyze customer behavior, predict churn risk, and optimize portfolio value using credit card transaction data.

---

# 🚀 Executive Summary

Analyzed **10,000+ credit card customers** to identify spending behavior, churn signals, and portfolio value drivers.

Built a **multi-layer analytics pipeline** integrating:

* Data Governance
* Feature Engineering (Banking KPIs)
* Behavioral Segmentation
* SQL-based Portfolio Analytics
* Machine Learning (Ensemble Models + Explainable AI)

The system simulates **real-world credit card portfolio analytics** used in fintech organizations.

---

# 🎯 Project Objectives

* Analyze customer spending and engagement patterns
* Identify high-value and churn-risk customers
* Perform behavioral segmentation
* Build predictive models for churn detection
* Simulate portfolio-level revenue (CLV & Yield Proxy)

---

# 📊 Dataset

* **Dataset:** Bank Customer Churn Dataset
* ~10,000 customers
* Features include:

  * Demographics
  * Transaction behavior
  * Credit utilization
  * Engagement metrics

---

# 🏗️ Project Architecture

```
Raw Data
   ↓
Data Governance Layer
   ↓
Feature Engineering (Banking KPIs)
   ↓
Customer Segmentation
   ↓
SQL Analytics Layer
   ↓
Model Benchmarking
   ↓
Advanced ML Models (LightGBM, XGBoost)
   ↓
Stacking Ensemble
   ↓
Explainable AI (SHAP)
   ↓
Business Insights & Dashboard
```

---

# 🧠 Key Components

## 1️⃣ Data Governance

* Data cleaning & validation pipeline
* Handling missing/unknown values
* Outlier strategy (VIP flagging)
* Structural and behavioral validation rules

---

## 2️⃣ Feature Engineering (Banking KPIs)

Created domain-specific features:

* **Ticket Size** → Avg transaction value
* **Utilization Ratio** → Credit usage behavior
* **Engagement Score** → Customer activity level
* **Transaction Velocity** → Monthly usage intensity
* **Portfolio Yield Proxy** → Revenue approximation
* **CLV Score** → Customer lifetime value estimation

---

## 3️⃣ Customer Segmentation

Customers classified into:

* **Power Spenders** → High revenue drivers
* **Revolvers** → Interest-generating customers
* **Transactors** → Stable low-risk users
* **Dormant Customers** → High churn risk

---

## 4️⃣ SQL Analytics Layer

Implemented portfolio-level analysis using SQL:

* Segment-wise revenue contribution
* Income vs spending patterns
* Churn indicators
* CLV distribution

---

## 5️⃣ Machine Learning Pipeline

### Model Benchmarking

* Logistic Regression
* Random Forest
* SVM
* XGBoost
* LightGBM

### Results

* Best Models: **XGBoost / LightGBM (~0.98 ROC-AUC)**
* Strong churn prediction capability

---

## 6️⃣ Advanced Modeling (In Progress)

* LightGBM Hyperparameter Optimization
* Class Imbalance Handling (SMOTETomek)
* Cross-validation pipelines
* Feature interaction engineering

---

## 7️⃣ Ensemble Learning (Upcoming)

* Stacking Classifier:

  * Base Models: RF, XGBoost, LightGBM
  * Meta Model: Logistic Regression
* Goal: Improve generalization & robustness

---

## 8️⃣ Explainable AI (XAI)

* SHAP-based model interpretation
* Feature importance analysis
* Customer-level churn explanations

---

## 📈 Business Impact

This project demonstrates:

* Identification of **high-value customers**
* Early detection of **churn risk**
* Optimization of **portfolio yield**
* Data-driven **retention strategies**

---

# 🛠️ Tech Stack

**Languages & Tools**

* Python
* SQL (SQLite)
* Jupyter Notebook

**Libraries**

* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* SHAP
* Imbalanced-learn

---

# 📂 Project Structure

```
project_alpha_bank/

├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_governance.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_customer_segmentation.ipynb
│   ├── 04_sql_analytics.ipynb
│   ├── 05_model_benchmarking.ipynb
│   ├── 06_lightgbm_optimization.ipynb   (in progress)
│   └── 07_stacking_ensemble.ipynb       (planned)
│
├── sql_queries/
│
├── documentation/
│
└── README.md
```

---

# 🔮 Future Enhancements

* 🔹 Stacking Ensemble Implementation
* 🔹 Hyperparameter Optimization (Optuna)
* 🔹 Real-time scoring pipeline simulation
* 🔹 Power BI Dashboard (Executive View)
* 🔹 Deployment-ready ML pipeline
* 🔹 Profit-based evaluation (EMPC approach)

---

# 🧠 Key Learnings

* Importance of **domain-driven feature engineering**
* Difference between **accuracy vs business impact**
* Power of **ensemble models in tabular data**
* Role of **Explainable AI in financial systems**

---

# 👨‍💻 Author

**Om Mishra**
Reliance Foundation Scholar

🔗 LinkedIn: https://www.linkedin.com/in/om-mishra-a62991289

---

# ⭐ Final Note

This project is designed to simulate a **real-world fintech analytics system**, combining:

* Data Analytics
* Machine Learning
* Business Intelligence

It bridges the gap between **Data Analyst → Data Scientist → Analytics Engineer** roles.
