"""
Flask API equivalent of dashboard/app.py (Streamlit churn intelligence).

Environment:
- CHURN_FEATURES_CSV — override path to the engineered feature table (default:
  dataset/processed/credit_card_features.csv). Missing columns trigger an
  automatic rebuild via dataset/feature_engineering.py.
- CHURN_MODEL_PATH — joblib model path. If the file is missing, a small
  RandomForest is trained in-memory from the CSV (needs Attrition_Flag). Set
  CHURN_DISABLE_STUB_MODEL=1 to require a real saved model instead.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import logging

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestClassifier

from dataset.feature_engineering import (
    MODEL_FEATURE_COLUMNS,
    apply_missing_feature_defaults,
    configure_logging,
    ensure_features_dataset,
)

logger = logging.getLogger(__name__)

# Console logging for API debugging (respect LOG_LEVEL; idempotent if Flask already configured).
configure_logging()

# Single source of truth for model inputs (dataset/feature_engineering.py).
FEATURES = list(MODEL_FEATURE_COLUMNS)

RISK_BINS = [0, 0.3, 0.6, 1.0]
RISK_LABELS = ["Low Risk", "Medium Risk", "High Risk"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_model_path() -> str:
    return os.environ.get(
        "CHURN_MODEL_PATH", str(_repo_root() / "dashboard" / "model.pkl")
    )


def _default_features_csv() -> str:
    return os.environ.get(
        "CHURN_FEATURES_CSV",
        str(_repo_root() / "dataset" / "processed" / "credit_card_features.csv"),
    )


@dataclass
class ModelBundle:
    model: Any
    df: pd.DataFrame
    X: pd.DataFrame
    explainer: shap.TreeExplainer
    shap_matrix: np.ndarray


_bundle: ModelBundle | None = None


def _expected_positive_base(explainer: shap.TreeExplainer) -> float:
    ev = explainer.expected_value
    if isinstance(ev, (list, tuple)):
        return float(ev[1])
    arr = np.atleast_1d(np.asarray(ev))
    if arr.size > 1:
        return float(arr[1])
    return float(arr[0])


def _shap_values_for_positive_class(
    explainer: shap.TreeExplainer, X: pd.DataFrame
) -> np.ndarray:
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        return np.asarray(raw[1])
    arr = np.asarray(raw)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def _train_stub_model(df: pd.DataFrame) -> RandomForestClassifier:
    """
    If no model.pkl is present, fit a small tree ensemble on the same CSV.

    This keeps /health green in a fresh checkout. Set CHURN_DISABLE_STUB_MODEL=1
    to require a real joblib model on disk instead.
    """
    if "Attrition_Flag" not in df.columns:
        raise ValueError(
            "Stub training needs Attrition_Flag in the features CSV "
            "(rebuild from dataset/processed/credit_card_clean.csv)."
        )
    X = df[FEATURES].to_numpy(dtype=float)
    y = (df["Attrition_Flag"].astype(str) == "Attrited Customer").astype(int)
    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=10,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(X, y)
    logger.warning(
        "Using in-memory stub RandomForest (no file at CHURN_MODEL_PATH). "
        "Train your real model and save joblib, or set CHURN_DISABLE_STUB_MODEL=1 to fail fast."
    )
    return model


def _load_model(df: pd.DataFrame) -> Any:
    model_path = _default_model_path()
    if os.path.isfile(model_path):
        return joblib.load(model_path)
    disable = os.environ.get("CHURN_DISABLE_STUB_MODEL", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if disable:
        raise FileNotFoundError(
            f"Model not found at {model_path} and stub model is disabled "
            "(CHURN_DISABLE_STUB_MODEL)."
        )
    return _train_stub_model(df)


def load_bundle() -> ModelBundle:
    global _bundle
    if _bundle is not None:
        return _bundle
    csv_path = Path(_default_features_csv())
    logger.info("Loading bundle; ensuring features at %s", csv_path)
    ensure_features_dataset(csv_path, _repo_root())
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        logger.warning(
            "Features still missing after ensure_features_dataset: %s — applying defaults",
            missing,
        )
        df = apply_missing_feature_defaults(df)
        df.to_csv(csv_path, index=False)
        logger.info("Persisted default-filled features to %s", csv_path)
    else:
        df = apply_missing_feature_defaults(df)
    model = _load_model(df)
    X = df[FEATURES].copy()
    df = df.copy()
    df["Churn_Prob"] = model.predict_proba(X)[:, 1]
    df["Risk_Segment"] = pd.cut(
        df["Churn_Prob"], bins=RISK_BINS, labels=RISK_LABELS, include_lowest=True
    )
    explainer = shap.TreeExplainer(model)
    shap_matrix = _shap_values_for_positive_class(explainer, X)
    _bundle = ModelBundle(
        model=model,
        df=df,
        X=X,
        explainer=explainer,
        shap_matrix=shap_matrix,
    )
    return _bundle


def _require_bundle() -> ModelBundle:
    try:
        return load_bundle()
    except (RuntimeError, FileNotFoundError, ValueError, OSError) as e:
        raise ApiNotReady(str(e)) from e


class ApiNotReady(Exception):
    pass


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def home():
        return jsonify({
            "message": "Customer Analytics API is running 🚀",
            "status": "success",
            "available_endpoints": [
                "/health",
                "/api/meta",
                "/api/summary",
                "/api/predict",
                "/api/customers"
            ]
        })

    @app.errorhandler(ApiNotReady)
    def handle_not_ready(err: ApiNotReady):
        return jsonify({"error": str(err)}), 503

    @app.get("/health")
    def health():
        try:
            load_bundle()
            return jsonify({"status": "ok"})
        except (RuntimeError, FileNotFoundError, ValueError, OSError) as e:
            return jsonify({"status": "degraded", "detail": str(e)}), 503

    @app.get("/api/meta")
    def meta():
        b = _require_bundle()
        return jsonify(
            {
                "features": FEATURES,
                "row_count": int(len(b.df)),
                "risk_labels": RISK_LABELS,
            }
        )

    def _filter_df(risk_segment: str | None):
        b = _require_bundle()
        out = b.df
        if risk_segment and risk_segment != "All":
            out = out[out["Risk_Segment"].astype(str) == risk_segment]
        return b, out

    @app.get("/api/summary")
    def summary():
        risk = request.args.get("risk_segment", "All")
        _, d = _filter_df(risk)
        high = int((d["Risk_Segment"].astype(str) == "High Risk").sum())
        return jsonify(
            {
                "risk_segment_filter": risk,
                "total_customers": int(len(d)),
                "avg_churn_risk": float(d["Churn_Prob"].mean())
                if len(d)
                else None,
                "high_risk_customers": high,
            }
        )

    @app.get("/api/distribution")
    def distribution():
        risk = request.args.get("risk_segment", "All")
        bins = int(request.args.get("bins", 30))
        _, d = _filter_df(risk)
        if len(d) == 0:
            return jsonify({"bins": [], "counts": []})
        counts, edges = np.histogram(d["Churn_Prob"], bins=bins)
        return jsonify(
            {
                "bins": [float(x) for x in edges],
                "counts": [int(c) for c in counts],
            }
        )

    @app.get("/api/drift_scatter")
    def drift_scatter():
        risk = request.args.get("risk_segment", "All")
        max_points = int(request.args.get("max_points", 5000))
        _, d = _filter_df(risk)
        if len(d) == 0:
            return jsonify({"points": []})
        sample = d
        if len(sample) > max_points:
            sample = sample.sample(max_points, random_state=42)
        points = [
            {
                "transaction_drift": float(r["Transaction_Drift"]),
                "churn_prob": float(r["Churn_Prob"]),
            }
            for _, r in sample.iterrows()
        ]
        return jsonify({"points": points})

    @app.get("/api/customers")
    def customers():
        risk = request.args.get("risk_segment", "All")
        offset = max(int(request.args.get("offset", 0)), 0)
        limit = min(max(int(request.args.get("limit", 50)), 1), 500)
        _, d = _filter_df(risk)
        page = d.iloc[offset : offset + limit]
        records = _records_from_df(page)
        return jsonify(
            {
                "offset": offset,
                "limit": limit,
                "total": int(len(d)),
                "customers": records,
            }
        )

    @app.get("/api/customers/<int:idx>")
    def customer_one(idx: int):
        b = _require_bundle()
        if idx < 0 or idx >= len(b.df):
            return jsonify({"error": "index out of range"}), 404
        row = b.df.iloc[idx]
        return jsonify(_record_from_series(idx, row))

    @app.post("/api/predict")
    def predict():
        b = _require_bundle()
        payload = request.get_json(silent=True) or {}
        missing = [f for f in FEATURES if f not in payload]
        if missing:
            return jsonify({"error": "missing fields", "fields": missing}), 400
        x = pd.DataFrame([{f: payload[f] for f in FEATURES}])
        prob = float(b.model.predict_proba(x)[0, 1])
        seg = pd.cut([prob], bins=RISK_BINS, labels=RISK_LABELS, include_lowest=True)[
            0
        ]
        return jsonify(
            {
                "churn_probability": prob,
                "risk_segment": str(seg),
            }
        )

    @app.get("/api/customers/<int:idx>/shap")
    def customer_shap(idx: int):
        b = _require_bundle()
        if idx < 0 or idx >= len(b.df):
            return jsonify({"error": "index out of range"}), 404
        base = _expected_positive_base(b.explainer)
        row_x = b.X.iloc[idx]
        vals = b.shap_matrix[idx]
        items = [
            {
                "feature": FEATURES[i],
                "value": float(row_x.iloc[i]),
                "shap_value": float(vals[i]),
            }
            for i in range(len(FEATURES))
        ]
        return jsonify(
            {
                "index": idx,
                "expected_value_positive_class": base,
                "churn_probability": float(b.df.iloc[idx]["Churn_Prob"]),
                "contributions": items,
            }
        )

    @app.get("/api/shap/summary.png")
    def shap_summary_png():
        b = _require_bundle()
        plt.figure(figsize=(10, 6))
        shap.summary_plot(b.shap_matrix, b.X, show=False)
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode("ascii")
        return jsonify({"format": "png", "base64": data})

    return app


def _record_from_series(idx: int, row: pd.Series) -> dict[str, Any]:
    d = row.to_dict()
    out: dict[str, Any] = {"index": idx}
    for k, v in d.items():
        if k == "Risk_Segment":
            out[k] = str(v)
        elif isinstance(v, (np.floating, float)):
            out[k] = float(v) if pd.notna(v) else None
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        elif pd.isna(v):
            out[k] = None
        else:
            out[k] = v
    return out


def _records_from_df(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_record_from_series(int(i), frame.loc[i]) for i in frame.index]


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")