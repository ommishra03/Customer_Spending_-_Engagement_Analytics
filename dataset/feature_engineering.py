"""
Feature engineering pipeline for the churn scoring API.

This module turns bank customer tables (raw Kaggle export or cleaned CSV) into
the exact feature columns expected by api/app.py. Definitions follow the
project notebooks (e.g. Statistical_Analysis/04_behavioral_drift_analysis.ipynb):

- We do not have true time-series, so "recent" behavior is proxied by scaling
  totals by 12 months, and "expected" behavior by spreading totals over
  Months_on_book (lifetime average intensity).
- Drift = recent proxy minus long-run average proxy (same idea as the notebook).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns the Flask model must receive (keep in sync with api/app.py consumers).
MODEL_FEATURE_COLUMNS: list[str] = [
    "Transaction_Velocity",
    "Engagement",
    "Avg_Utilization_Ratio",
    "Months_Inactive_12_mon",
    "Transaction_Drift",
    "Spend_Drift",
    "Engagement_Drift",
    "Behavioral_Risk_Score",
]

# Minimum tenure (months) used in denominators to avoid divide-by-zero.
_MIN_MONTHS_ON_BOOK = 1

# Weights for Behavioral_Risk_Score (from behavioral drift notebook).
_W_TXN = 0.4
_W_SPEND = 0.3
_W_INACTIVITY = 0.3


def _repo_root_from_here() -> Path:
    """This file lives in dataset/; repository root is one level up."""
    return Path(__file__).resolve().parent.parent


def _drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that are not legitimate model inputs (data leakage or IDs).

    The raw Kaggle file includes two Naive Bayes classifier columns that are
    direct churn hints; we drop anything whose name starts with that prefix.
    """
    out = df.copy()
    bad = [c for c in out.columns if str(c).startswith("Naive_Bayes")]
    if bad:
        out = out.drop(columns=bad, errors="ignore")
    return out


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert listed columns to float; non-numeric values become NaN."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _impute_numeric_median(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Fill missing numeric values with the column median (robust default).

    Beginner note: median is less sensitive to extreme values than the mean,
    which is why it is common for banking tabular data.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        med = out[col].median()
        if pd.isna(med):
            med = 0.0
        out[col] = out[col].fillna(med)
    return out


def load_source_dataframe(repo_root: Path | None = None) -> pd.DataFrame:
    """
    Load the best available source table: cleaned CSV first, else raw CSV.

    Returns a DataFrame with standard column names (no quoted headers).
    """
    root = repo_root or _repo_root_from_here()
    processed = root / "dataset" / "processed" / "credit_card_clean.csv"
    raw_path = root / "dataset" / "raw" / "BankChurners_raw.csv"

    if processed.is_file():
        path = processed
        logger.info("Loading cleaned source: %s", path)
    elif raw_path.is_file():
        path = raw_path
        logger.info("Loading raw source: %s", path)
    else:
        raise FileNotFoundError(
            "No source data found. Expected one of:\n"
            f"  - {processed}\n"
            f"  - {raw_path}"
        )

    df = pd.read_csv(path)
    df = _drop_leakage_columns(df)
    return df


def _base_columns_present(df: pd.DataFrame) -> bool:
    """True if we have the raw inputs needed to compute all model features."""
    needed = [
        "Total_Trans_Ct",
        "Total_Trans_Amt",
        "Months_on_book",
        "Months_Inactive_12_mon",
        "Avg_Utilization_Ratio",
    ]
    return all(c in df.columns for c in needed)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add or overwrite MODEL_FEATURE_COLUMNS using notebook-aligned logic.

    Step-by-step (beginner-friendly):
    1) Clean numeric inputs and impute missing values.
    2) Clip Months_on_book so we never divide by zero.
    3) Transaction_Velocity ≈ average transactions per month over last year
       (here: total count / 12, same as "Recent_Transactions" in the notebook).
    4) Engagement ≈ transactions per month over full relationship length
       (total count / Months_on_book).
    5) Transaction_Drift compares "recent monthly rate" to "lifetime monthly rate".
       Negative drift means the customer slowed down recently → higher risk.
    6) Spend_Drift does the same for dollars instead of counts.
    7) Engagement_Drift compares two engagement views (recent vs lifetime); with
       the definitions above it matches Transaction_Drift (as in the notebook).
    8) Inactivity_Risk scales months inactive in the last year to [0, 1] range.
    9) Behavioral_Risk_Score combines drift (negative drift increases risk)
       and inactivity with fixed weights from the notebook.
    """
    if not _base_columns_present(df):
        missing = [
            c
            for c in [
                "Total_Trans_Ct",
                "Total_Trans_Amt",
                "Months_on_book",
                "Months_Inactive_12_mon",
                "Avg_Utilization_Ratio",
            ]
            if c not in df.columns
        ]
        raise ValueError(f"Cannot engineer features; missing columns: {missing}")

    num_cols = [
        "Total_Trans_Ct",
        "Total_Trans_Amt",
        "Months_on_book",
        "Months_Inactive_12_mon",
        "Avg_Utilization_Ratio",
    ]
    out = _coerce_numeric(df, num_cols)
    out = _impute_numeric_median(out, num_cols)

    mob = out["Months_on_book"].clip(lower=_MIN_MONTHS_ON_BOOK).astype(float)

    # Core KPIs (see python_analysis/2_feature_engineering.ipynb).
    out["Transaction_Velocity"] = out["Total_Trans_Ct"] / 12.0
    out["Engagement"] = out["Total_Trans_Ct"] / mob

    # Long-run vs "recent" proxies (Statistical_Analysis/04_behavioral_drift...).
    expected_txn = out["Total_Trans_Ct"] / mob
    recent_txn = out["Total_Trans_Ct"] / 12.0
    out["Transaction_Drift"] = recent_txn - expected_txn

    expected_spend = out["Total_Trans_Amt"] / mob
    recent_spend = out["Total_Trans_Amt"] / 12.0
    out["Spend_Drift"] = recent_spend - expected_spend

    engagement_recent = out["Total_Trans_Ct"] / 12.0
    engagement_long_run = out["Total_Trans_Ct"] / mob
    out["Engagement_Drift"] = engagement_recent - engagement_long_run

    inactivity_risk = out["Months_Inactive_12_mon"].astype(float) / 12.0
    out["Behavioral_Risk_Score"] = (
        -out["Transaction_Drift"] * _W_TXN
        + -out["Spend_Drift"] * _W_SPEND
        + inactivity_risk * _W_INACTIVITY
    )

    # Replace inf/nan from any edge case after math.
    feat_df = out[MODEL_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    feat_df = feat_df.fillna(0.0)
    for col in MODEL_FEATURE_COLUMNS:
        out[col] = feat_df[col]

    return out


def _apply_defaults_for_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Last-resort: if any model column is still missing, fill with 0.0.

    Zero drift / zero risk is a neutral baseline when we cannot compute better.
    """
    out = df.copy()
    for col in MODEL_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def build_features_dataframe(base: pd.DataFrame) -> pd.DataFrame:
    """
    Return a table containing at least MODEL_FEATURE_COLUMNS, ready for CSV export.

    Preserves non-feature columns from `base` (e.g. Attrition_Flag) when present.
    """
    engineered = engineer_features(base)
    # Start from original rows and attach/overwrite engineered columns.
    out = base.copy()
    for col in MODEL_FEATURE_COLUMNS:
        out[col] = engineered[col]
    out = _apply_defaults_for_missing_features(out)
    return out


def run_pipeline(
    output_path: Path | None = None,
    source_path: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    """
    End-to-end: load source → engineer features → write credit_card_features.csv.

    If source_path is provided, that file is used instead of the default discovery.
    """
    root = repo_root or _repo_root_from_here()
    out = output_path or (root / "dataset" / "processed" / "credit_card_features.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    if source_path is not None:
        df = pd.read_csv(source_path)
        df = _drop_leakage_columns(df)
    else:
        df = load_source_dataframe(root)

    final_df = build_features_dataframe(df)
    final_df.to_csv(out, index=False)
    logger.info("Wrote %s rows to %s", len(final_df), out)
    return out


def ensure_features_dataset(
    output_path: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    """
    Ensure output CSV exists and contains every MODEL_FEATURE_COLUMN.

    Used by the API at startup:
    - If the file is missing → build from cleaned/raw source.
    - If the file exists but is missing some features → recompute those rows
      from the same file when possible; otherwise rebuild from source.
    """
    root = repo_root or _repo_root_from_here()
    out = output_path or (root / "dataset" / "processed" / "credit_card_features.csv")

    if not out.is_file():
        logger.info("Features file missing; building: %s", out)
        return run_pipeline(output_path=out, repo_root=root)

    existing = pd.read_csv(out)
    missing = [c for c in MODEL_FEATURE_COLUMNS if c not in existing.columns]

    if not missing:
        # Still normalize NaNs in feature columns if any crept in.
        fixed = _apply_defaults_for_missing_features(existing)
        for col in MODEL_FEATURE_COLUMNS:
            existing[col] = fixed[col]
        existing.to_csv(out, index=False)
        return out

    logger.info("Features file missing columns %s; repairing.", missing)

    if _base_columns_present(existing):
        repaired = build_features_dataframe(existing)
        repaired.to_csv(out, index=False)
        return out

    # Cannot repair in place — rebuild from authoritative source.
    return run_pipeline(output_path=out, repo_root=root)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = run_pipeline()
    print(f"OK: wrote {path}")
