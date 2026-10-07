from pathlib import Path  # noqa: I001

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    TARGET_COLUMN,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

from .feature_engineering import add_engineered_features


# ============================================================
# LOAD DATA
# ============================================================

def load_raw_data(path: Path) -> pd.DataFrame:
    """
    Load the raw CSV dataset.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"""
Dataset not found:

{path}

Run:

python -m src.download_data

first.
"""
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    return df


# ============================================================
# TARGET CREATION
# ============================================================

def make_binary_target(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Convert original readmission target:

        <30 -> 1
        >30 -> 0
        NO  -> 0
    """

    target = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .eq("<30")
        .astype(int)
    )

    if target.sum() == 0:

        raise ValueError(
            "No '<30' readmission records found."
        )

    return target


# ============================================================
# MODEL DATA PREPARATION
# ============================================================

def prepare_model_data(
    df: pd.DataFrame,
):
    """
    Clean raw data, engineer features,
    select model features and generate target.
    """

    df = df.copy()

    # Convert ? into missing values
    df = df.replace("?", np.nan)

    # Target
    y = make_binary_target(df)

    # Feature engineering
    df = add_engineered_features(df)

    # Check required columns
    missing_columns = [
        column
        for column in MODEL_FEATURES
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Select final model features
    X = df[MODEL_FEATURES].copy()

    return X, y


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing pipeline.

    Numeric:
        median imputation
        standard scaling

    Categorical:
        most-frequent imputation
        one-hot encoding
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ============================================================
# SINGLE PATIENT RECORD
# ============================================================

def clean_single_record(
    record: dict,
) -> pd.DataFrame:
    """
    Convert a dictionary representing one patient
    into the exact feature format used by the model.
    """

    row = pd.DataFrame([record])

    # Diagnosis columns
    for column in [
        "diag_1",
        "diag_2",
        "diag_3",
    ]:

        if column not in row.columns:
            row[column] = "Unknown"

    # Convert ? values
    row = row.replace("?", np.nan)

    # Feature engineering
    row = add_engineered_features(row)

    # Add missing model fields
    for column in MODEL_FEATURES:

        if column not in row.columns:
            row[column] = np.nan

    return row[MODEL_FEATURES]
