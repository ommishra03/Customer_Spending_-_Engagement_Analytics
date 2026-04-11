# 💳 Credit Card Lifecycle Analytics Engine

> **End-to-End FinTech ML System** — From raw customer data to real-time churn predictions via a production-ready Flask API.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-Backend_API-black?logo=flask)](https://flask.palletsprojects.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML_Model-orange)](https://xgboost.readthedocs.io)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.98-brightgreen)](https://github.com/ommishra03/Customer_Spending_-_Engagement_Analytics)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🚀 Executive Summary

This project is a **full-stack Machine Learning + Backend Engineering system** built to simulate real-world fintech infrastructure. It processes 10,000+ credit card customer records through a complete data pipeline — from raw ingestion to ML-powered predictions — and exposes results through a **modular Flask REST API**.

Unlike typical data science projects that stop at the notebook stage, this system is architected for **production deployment**: feature engineering happens at inference time, models are served via REST endpoints, and the backend is designed for integration with banking dashboards or third-party applications.

**What it delivers:**
- 🔍 Deep behavioral analytics on credit card customers
- 🤖 Churn prediction with ~0.98 ROC-AUC
- ⚡ Real-time predictions via REST API
- 📊 Explainable AI using SHAP for transparent decisions

---

## 🎯 Problem Statement

Financial institutions consistently struggle with three core challenges:

| Challenge | Business Impact |
|-----------|----------------|
| Identifying high-risk customers before they churn | Revenue loss, increased acquisition cost |
| Understanding behavioral shifts in spending | Missed upsell and retention opportunities |
| Translating ML insights into operational decisions | Analytics remains siloed from business systems |

This project directly addresses all three by building a **closed-loop system** that connects data, models, and decision-making infrastructure.

---

## 💡 Solution Overview

The system is built around a single guiding principle: **analytics should be actionable, not just insightful.**

- Raw customer data is cleaned and validated through a governance pipeline
- Domain-specific banking KPIs are engineered as predictive features
- Customers are segmented by behavioral profiles
- An ensemble ML model predicts churn probability with high accuracy
- SHAP values explain *why* a customer is flagged as at-risk
- A Flask API wraps the entire pipeline for real-time, on-demand predictions

---

## 🏗️ System Architecture

```
Raw Customer Data (CSV)
        │
        ▼
┌─────────────────────┐
│  Data Governance    │  ← Validation, cleaning, outlier handling
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Feature Engineering │  ← Banking KPIs, behavioral signals
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│Customer Segmentation│  ← Power Spenders, Revolvers, Dormant, etc.
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   SQL Analytics     │  ← Portfolio-level KPIs via SQLite
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   ML Model Suite    │  ← XGBoost / LightGBM / Random Forest
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Explainable AI     │  ← SHAP feature importance per customer
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Flask REST API    │  ← Real-time inference endpoint
└─────────────────────┘
        │
        ▼
  Dashboards / Apps / Banking Systems
```

---

## 🧩 Key Components

---

### 🔹 1. Data Engineering & Governance

A robust preprocessing pipeline ensures data quality before any modeling occurs.

- **Schema validation** — enforces expected column types and ranges
- **Missing value handling** — context-aware imputation strategies
- **Outlier treatment** — VIP customer segmentation logic prevents high-value customers from being flagged as anomalies
- **Self-healing dataset mechanism** — the API detects and resolves data inconsistencies at runtime without crashing

---

### 🔹 2. Feature Engineering (Banking KPIs)

All features are derived from first principles of credit card behavior — no generic feature extraction. Each KPI maps to a real metric used in banking risk and retention teams.

| Feature | Description |
|---------|-------------|
| `Transaction Velocity` | Monthly activity rate — measures engagement frequency |
| `Engagement Score` | Composite customer interaction level |
| `Utilization Ratio` | Credit usage as a fraction of available limit |
| `Transaction Drift` | Change in transaction count over time — early churn signal |
| `Spend Drift` | Shift in spending patterns across time windows |
| `Behavioral Risk Score` | Composite churn signal aggregated from all behavioral features |

> These features are computed **at inference time inside the API**, meaning the `/predict` endpoint accepts raw customer data and handles all transformation internally.

---

### 🔹 3. Customer Segmentation

Customers are classified into four behavioral segments used for targeted retention strategies:

| Segment | Behavior | Business Action |
|---------|----------|-----------------|
| **Power Spenders** | High transaction volume, high spend | Upsell premium products |
| **Revolvers** | Carry balance month-to-month | Interest revenue drivers |
| **Transactors** | Regular, low-risk usage | Stable portfolio anchor |
| **Dormant Customers** | Low engagement, declining activity | Immediate retention intervention |

---

### 🔹 4. SQL Analytics Layer

Portfolio-level insights are computed using SQL (SQLite) to simulate enterprise reporting:

- Segment-wise revenue and churn contribution
- Spending vs. income correlation analysis
- Customer Lifetime Value (CLV) distribution
- Leading churn indicators by segment

---

### 🔹 5. Machine Learning Pipeline

Five models were trained and evaluated to select the best performer:

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline interpretability benchmark |
| Random Forest | Strong feature importance signal |
| SVM | Effective on high-dimensional feature space |
| XGBoost | Top performer; gradient boosting ensemble |
| LightGBM | Fast training; production efficiency |

**Best performance: ~0.98 ROC-AUC** on the test set, indicating strong separation between churners and non-churners.

---

### 🔹 6. Explainable AI (SHAP)

ML models are only as valuable as the trust they generate. SHAP (SHapley Additive exPlanations) is integrated to:

- Rank features by global importance across the entire customer base
- Generate **per-customer explanations** for individual churn predictions
- Support transparent, auditable decision-making — critical for regulated financial environments

---

## 🔥 Backend System — Flask REST API

> This is what separates this project from a typical ML notebook.

The Flask backend converts the entire analytics + ML pipeline into a **deployable service**. It is structured with a clean separation of concerns across `routes/`, `services/`, and `utils/` — mirroring how real fintech backend teams organize production APIs.

### 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check — confirms API and model are loaded |
| `GET` | `/api/summary` | Portfolio-level KPIs: churn rate, segment distribution, CLV |
| `GET` | `/api/customers` | Paginated customer-level records with behavioral scores |
| `POST` | `/api/predict` | Real-time churn prediction for a single customer |
| `GET` | `/api/meta` | Feature metadata — names, types, and expected ranges |

---

### 🔁 Prediction Flow (`/api/predict`)

```
POST /api/predict
        │
        ▼
  Input: Raw customer fields (JSON)
        │
        ▼
  Data Validation Layer
  (type checks, range enforcement, fallback handling)
        │
        ▼
  Feature Engineering Pipeline
  (same transformations used during training)
        │
        ▼
  ML Model Inference (XGBoost / LightGBM)
        │
        ▼
  Output: {
    "churn_probability": 0.87,
    "risk_segment": "High Risk",
    "top_risk_factors": ["spend_drift", "low_utilization"]
  }
```

### Example Request

```json
POST /api/predict
Content-Type: application/json

{
  "customer_age": 45,
  "credit_limit": 12000,
  "total_trans_amt": 1450,
  "total_trans_ct": 28,
  "months_inactive_12_mon": 4,
  "contacts_count_12_mon": 3
}
```

### Example Response

```json
{
  "churn_probability": 0.84,
  "risk_label": "High Risk",
  "segment": "Dormant",
  "key_drivers": ["months_inactive", "transaction_drift", "spend_drift"]
}
```

---

## 📂 Project Structure

```
project_alpha_bank/
│
├── dataset/
│   ├── raw/                   # Original, unmodified source data
│   └── processed/             # Cleaned, feature-engineered datasets
│
├── api/                       # Flask Backend (Production Core)
│   ├── app.py                 # App factory, config, startup
│   ├── routes/                # Endpoint definitions (predict, summary, customers)
│   ├── services/              # Business logic (feature engineering, model inference)
│   └── utils/                 # Helpers (validation, serialization, error handling)
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  # KPI design and validation
│   ├── 03_segmentation.ipynb  # Customer clustering and profiling
│   ├── 04_ml_models.ipynb     # Model training, evaluation, comparison
│   └── 05_shap_explainability.ipynb  # SHAP analysis and visualization
│
├── sql_queries/               # Portfolio analytics SQL scripts
│   ├── segment_revenue.sql
│   ├── churn_indicators.sql
│   └── clv_distribution.sql
│
├── documentation/             # Architecture diagrams, API specs
└── README.md
```

---

## 📓 Notebooks Breakdown

| Notebook | Purpose |
|----------|---------|
| `01_eda.ipynb` | Distribution analysis, correlation heatmaps, missing value profiling |
| `02_feature_engineering.ipynb` | Design and validation of all 6 banking KPIs |
| `03_segmentation.ipynb` | Behavioral clustering — Power Spenders, Revolvers, Transactors, Dormant |
| `04_ml_models.ipynb` | Training Logistic Regression, Random Forest, SVM, XGBoost, LightGBM; ROC-AUC comparison |
| `05_shap_explainability.ipynb` | Global and per-customer SHAP explanations, waterfall plots |

---

## 🛠️ Tech Stack

### Backend & API
| Tool | Role |
|------|------|
| Python 3.9+ | Core language |
| Flask | REST API framework |
| REST Architecture | Endpoint design and routing |

### Data & Machine Learning
| Tool | Role |
|------|------|
| Pandas, NumPy | Data manipulation and feature computation |
| Scikit-learn | Preprocessing, model evaluation, pipeline utilities |
| XGBoost | Primary churn prediction model |
| LightGBM | High-efficiency ensemble model |
| SHAP | Model explainability layer |

### Data Layer
| Tool | Role |
|------|------|
| SQLite | Portfolio analytics queries |
| CSV (raw/processed) | Input and output data storage |

---

## ✅ What Makes This Project Different

Most ML projects end at the notebook. This one doesn't.

| Typical ML Project | This Project |
|-------------------|--------------|
| ❌ Jupyter notebook only | ✅ Full Flask backend serving predictions |
| ❌ Static analysis output | ✅ Real-time inference via REST API |
| ❌ Model accuracy as the goal | ✅ Business impact as the goal |
| ❌ No production structure | ✅ Modular routes/services/utils architecture |
| ❌ Black-box decisions | ✅ SHAP-powered explainability per customer |
| ❌ Manual feature engineering | ✅ Feature pipeline runs automatically at inference |

---

## 📈 Business Impact

| Outcome | Description |
|---------|-------------|
| 🎯 Early churn detection | Flag at-risk customers weeks before they leave |
| 💰 Revenue protection | Target retention offers to high-value churners first |
| 📊 Portfolio optimization | Identify dormant customers for re-engagement campaigns |
| 🔍 Transparent decisions | SHAP explanations build trust with compliance teams |
| ⚡ Operational speed | REST API enables instant scoring at any point in the customer journey |

---

## 🔮 Future Enhancements

- [ ] **Automated retraining pipeline** — detect drift and retrain on a schedule (Airflow)
- [ ] **Stacking ensemble** — combine XGBoost + LightGBM predictions for improved accuracy
- [ ] **Hyperparameter tuning** — Optuna-based optimization for all models
- [ ] **Frontend dashboard** — React or Power BI visualization layer
- [ ] **Cloud deployment** — AWS EC2 / Render deployment with CI/CD pipeline
- [ ] **Real-time streaming** — Kafka-based event pipeline for live transaction scoring
- [ ] **Model monitoring** — track prediction drift and data quality over time

---

## 👨‍💻 Author

**Om Mishra**  
Reliance Foundation Scholar

Passionate about building systems that sit at the intersection of **data science and backend engineering** — where analytical insight becomes operational impact.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/om-mishra-a62991289)
[![GitHub](https://img.shields.io/badge/GitHub-ommishra03-black?logo=github)](https://github.com/ommishra03)

---

## ⭐ Final Note

> This project represents a deliberate transition from data analyst → ML engineer → backend developer.  
> It is not just a model. It is a **system**.

If you're a recruiter or engineer reviewing this: every design decision — from feature engineering to API architecture — was made with production readiness in mind.

---

*Built with ❤️ to demonstrate what end-to-end ML systems actually look like in fintech.*
