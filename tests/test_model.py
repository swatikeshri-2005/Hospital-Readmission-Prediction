import pandas as pd  # noqa: I001

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import MODEL_FEATURES

from src.data_preprocessing import (
    build_preprocessor,
    clean_single_record,
)

from src.feature_engineering import (
    normalize_icd9,
)


# ============================================================
# SAMPLE PATIENT
# ============================================================

def create_sample_patient():

    patient = {}

    numeric_features = {
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    }

    for column in MODEL_FEATURES:

        if column in numeric_features:

            patient[column] = 1

        else:

            patient[column] = "Unknown"

    patient.update(
        {
            "diag_1": "250.83",
            "diag_2": "401.9",
            "diag_3": "V58.67",
        }
    )

    return patient


# ============================================================
# TEST ICD-9
# ============================================================

def test_icd9_normalization():

    assert (
        normalize_icd9(
            "250.83"
        )
        == "250"
    )

    assert (
        normalize_icd9(
            "401.9"
        )
        == "401"
    )

    assert (
        normalize_icd9(
            "V58.67"
        )
        == "V58"
    )

    assert (
        normalize_icd9("?")
        == "Unknown"
    )


# ============================================================
# TEST SINGLE RECORD
# ============================================================

def test_single_record_shape():

    patient = (
        create_sample_patient()
    )

    X = clean_single_record(
        patient
    )

    assert (
        list(X.columns)
        == MODEL_FEATURES
    )

    assert X.shape == (
        1,
        len(MODEL_FEATURES),
    )


# ============================================================
# TEST PREPROCESSOR
# ============================================================

def test_preprocessor_and_model_fit():

    rows = []

    for i in range(20):

        patient = (
            create_sample_patient()
        )

        patient[
            "time_in_hospital"
        ] = (i % 10) + 1

        patient[
            "diag_1"
        ] = (
            "250.83"
            if i % 2
            else "401.9"
        )

        rows.append(
            clean_single_record(
                patient
            )
        )

    X = pd.concat(
        rows,
        ignore_index=True,
    )

    y = pd.Series(
        [0, 1] * 10
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),

            (
                "model",
                LogisticRegression(
                    max_iter=500,
                    solver="liblinear",
                ),
            ),
        ]
    )

    pipeline.fit(
        X,
        y,
    )

    probabilities = (
        pipeline.predict_proba(
            X
        )[:, 1]
    )

    assert len(probabilities) == len(y)