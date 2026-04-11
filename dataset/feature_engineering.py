"""
Feature engineering pipeline for the churn scoring API and model training.

=============================================================================
DATASET LAYOUT (this repo)
=============================================================================
- dataset/raw/BankChurners_raw.csv
    Kaggle export with quoted headers. Core behavioral columns include:
    Total_Trans_Ct, Total_Trans_Amt, Months_on_book, Months_Inactive_12_mon,
    Avg_Utilization_Ratio, Attrition_Flag, plus two Naive Bayes columns (dropped
    here as leakage).

- dataset/processed/credit_card_clean.csv
    Cleaned table: same core columns as above (no leakage columns), often used
    as the preferred input for batch jobs.

- dataset/processed/credit_card_features.csv (this pipeline's OUTPUT)
    All columns from the clean table PLUS the eight model features defined in
    MODEL_FEATURE_COLUMNS — aligned with api/app.py and the analysis notebooks.

=============================================================================
REPRODUCIBILITY
=============================================================================
- All transforms are deterministic (no random sampling inside this module).
- RANDOM_SEED is reserved for future steps (e.g. row sampling) and matches
  training stubs elsewhere in the project (e.g. RandomForest random_state=42).
- Bump PIPELINE_VERSION when you change formulas so you can trace which code
  version produced a CSV.

Beginner note: "drift" here does not use real month-by-month history. Because
we only have snapshot totals, we follow Statistical_Analysis/
04_behavioral_drift_analysis.ipynb: compare a *recent* monthly intensity
(total / 12) with a *lifetime* average intensity (total / Months_on_book).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed name so log lines read the same when imported or run as `python -m dataset.feature_engineering`.
logger = logging.getLogger("dataset.feature_engineering")

# Bump when formulas or imputation strategy change (traceability in logs).
PIPELINE_VERSION = "1.1.0"

# Reserved for any future randomized steps; training code should use the same.
RANDOM_SEED = 42

# Columns the Flask model must receive (keep in sync with api/app.py).
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

# Weights for Behavioral_Risk_Score (from 04_behavioral_drift_analysis.ipynb).
_W_TXN = 0.4
_W_SPEND = 0.3
_W_INACTIVITY = 0.3

# When the API cannot rebuild columns from raw inputs, these neutral defaults
# keep the service up. Zero drift = "no deviation from baseline" in our scaling.
_DEFAULT_NUMERIC_FALLBACK = 0.0


def configure_logging(level: int | None = None) -> None:
    """
    Configure root logging once (safe to call from CLI or API startup).

    If the root logger already has handlers (e.g. Flask), we do not attach
    another stream handler to avoid duplicate lines.
    """
    if level is None:
        level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    else:
        root.setLevel(level)


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
        logger.info("Dropping leakage columns: %s", bad)
        out = out.drop(columns=bad, errors="ignore")
    return out


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip accidental whitespace from headers (common after manual CSV edits)."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
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
        n_missing = int(out[col].isna().sum())
        if n_missing:
            logger.debug("Imputing %s missing values in %s with median %s", n_missing, col, med)
        out[col] = out[col].fillna(med)
    return out


def load_source_dataframe(repo_root: Path | None = None) -> pd.DataFrame:
    """
    Load the best available source table: cleaned CSV first, else raw CSV.

    Returns a DataFrame with standard column names (no quoted-header quirks).
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
    df = _normalize_column_names(df)
    df = _drop_leakage_columns(df)
    logger.info("Loaded %s rows, %s columns from %s", len(df), len(df.columns), path.name)
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


def missing_model_columns(df: pd.DataFrame) -> list[str]:
    """Return which MODEL_FEATURE_COLUMNS are absent from the dataframe."""
    return [c for c in MODEL_FEATURE_COLUMNS if c not in df.columns]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add or overwrite MODEL_FEATURE_COLUMNS using notebook-aligned logic.

    Step-by-step (beginner-friendly):
    1) Clean numeric inputs and impute missing values on base columns.
    2) Clip Months_on_book so we never divide by zero.
    3) Transaction_Velocity = Total_Trans_Ct / 12  (same as "Recent_Transactions"
       in the drift notebook — proxy for recent monthly transaction rate).
    4) Engagement = Total_Trans_Ct / Months_on_book  (lifetime monthly rate).
    5) Transaction_Drift = recent monthly rate minus lifetime monthly rate.
       Negative values mean the customer slowed down recently → higher churn risk
       in the downstream model.
    6) Spend_Drift does the same using Total_Trans_Amt instead of counts.
    7) Engagement_Drift = (Total_Trans_Ct/12) - (Total_Trans_Ct/Months_on_book);
       algebraically this equals Transaction_Drift (see notebook).
    8) Inactivity_Risk = Months_Inactive_12_mon / 12  (scaled to ~[0,1]).
    9) Behavioral_Risk_Score combines negative drift (risk when activity drops)
       and inactivity with fixed weights from the notebook:
       -0.4 * Transaction_Drift - 0.3 * Spend_Drift + 0.3 * Inactivity_Risk
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

    # Core KPIs (python_analysis/2_feature_engineering.ipynb).
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

    logger.info("Engineered %s model feature columns for %s rows", len(MODEL_FEATURE_COLUMNS), len(out))
    return out


def _apply_defaults_for_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure every model column exists and has finite numeric values.

    Missing columns become DEFAULT; NaNs become DEFAULT (neutral baseline).
    """
    out = df.copy()
    for col in MODEL_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = _DEFAULT_NUMERIC_FALLBACK
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(
                _DEFAULT_NUMERIC_FALLBACK
            )
    return out


def apply_missing_feature_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """
    Last-resort path for the API: add any still-missing model columns.

    Use this only when source data cannot be loaded to rebuild features.
    Logs a warning per inserted column so you can spot data problems quickly.
    """
    out = df.copy()
    for col in MODEL_FEATURE_COLUMNS:
        if col not in out.columns:
            logger.warning(
                "Column %s missing after pipeline — filling with default %s",
                col,
                _DEFAULT_NUMERIC_FALLBACK,
            )
            out[col] = _DEFAULT_NUMERIC_FALLBACK
        else:
            before_na = int(out[col].isna().sum())
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(
                _DEFAULT_NUMERIC_FALLBACK
            )
            if before_na:
                logger.warning(
                    "Column %s had %s NaN values — filled with %s",
                    col,
                    before_na,
                    _DEFAULT_NUMERIC_FALLBACK,
                )
    return out


def build_features_dataframe(base: pd.DataFrame) -> pd.DataFrame:
    """
    Return a table containing at least MODEL_FEATURE_COLUMNS, ready for CSV export.

    Preserves non-feature columns from `base` (e.g. Attrition_Flag) when present.
    """
    engineered = engineer_features(base)
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

    logger.info(
        "Starting feature pipeline version=%s (random_seed=%s)",
        PIPELINE_VERSION,
        RANDOM_SEED,
    )

    if source_path is not None:
        df = pd.read_csv(source_path)
        df = _normalize_column_names(df)
        df = _drop_leakage_columns(df)
        logger.info("Loaded explicit source_path=%s (%s rows)", source_path, len(df))
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

    Used by the Flask API on startup (via import, not a subprocess):
    - If the file is missing → build from cleaned/raw source.
    - If the file exists but is missing some features → recompute using base
      columns on that file when possible; otherwise rebuild from source.
    - If the file is complete → impute NaNs only when needed (avoids rewriting
      the CSV on every server start when nothing changed).
    """
    root = repo_root or _repo_root_from_here()
    out = output_path or (root / "dataset" / "processed" / "credit_card_features.csv")

    logger.info(
        "ensure_features_dataset: target=%s pipeline_version=%s",
        out,
        PIPELINE_VERSION,
    )

    if not out.is_file():
        logger.info("Features file missing; building: %s", out)
        return run_pipeline(output_path=out, repo_root=root)

    existing = pd.read_csv(out)
    existing = _normalize_column_names(existing)
    missing = missing_model_columns(existing)

    if not missing:
        subset = existing.reindex(columns=MODEL_FEATURE_COLUMNS)
        if subset.isna().any().any():
            logger.info("Features present but contain NaNs; normalizing in place.")
            fixed = _apply_defaults_for_missing_features(existing)
            for col in MODEL_FEATURE_COLUMNS:
                existing[col] = fixed[col]
            existing.to_csv(out, index=False)
            logger.info("Saved repaired feature columns to %s", out)
        else:
            logger.info("Feature CSV is complete; no rewrite needed.")
        return out

    logger.warning("Features file missing columns %s; attempting repair.", missing)

    if _base_columns_present(existing):
        repaired = build_features_dataframe(existing)
        repaired.to_csv(out, index=False)
        logger.info("Repaired features from existing rows; wrote %s", out)
        return out

    logger.warning("Cannot repair in place (missing base columns); rebuilding from source.")
    return run_pipeline(output_path=out, repo_root=root)


if __name__ == "__main__":
    configure_logging()
    path = run_pipeline()
    print(f"OK: pipeline {PIPELINE_VERSION} wrote {path}")
