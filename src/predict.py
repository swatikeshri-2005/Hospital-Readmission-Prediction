import argparse  # noqa: I001
import json

import joblib

from .config import MODEL_DIR
from .data_preprocessing import (
    clean_single_record,
)


# ============================================================
# SAMPLE PATIENT
# ============================================================

SAMPLE_PATIENT = {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[60-70)",

    "admission_type_id": "1",
    "discharge_disposition_id": "1",
    "admission_source_id": "7",

    "time_in_hospital": 7,
    "num_lab_procedures": 45,
    "num_procedures": 2,
    "num_medications": 15,

    "number_outpatient": 1,
    "number_emergency": 1,
    "number_inpatient": 2,
    "number_diagnoses": 8,

    "max_glu_serum": "None",
    "A1Cresult": "None",

    "metformin": "Steady",
    "insulin": "Steady",

    "change": "Ch",
    "diabetesMed": "Yes",

    "diag_1": "250.83",
    "diag_2": "401.9",
    "diag_3": "V58.67",
}


def load_patient_from_json(
    path: str,
):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Hospital Readmission "
            "Prediction"
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Path to patient JSON file."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "readmission_model.pkl"
    )

    bundle = joblib.load(
        model_path
    )

    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]
    model_name = bundle["model_name"]

    # --------------------------------------------------------
    # Patient data
    # --------------------------------------------------------

    if args.input:

        patient = load_patient_from_json(
            args.input
        )

    else:

        patient = SAMPLE_PATIENT

    X = clean_single_record(
        patient
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probability = float(
        pipeline.predict_proba(
            X
        )[0, 1]
    )

    prediction = int(
        probability >= threshold
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("HOSPITAL READMISSION PREDICTION")
    print("=" * 60)

    print(
        f"Model: {model_name}"
    )

    print(
        f"Probability of <30-day readmission: "
        f"{probability:.2%}"
    )

    print(
        f"Decision threshold: "
        f"{threshold:.2f}"
    )

    if prediction == 1:

        print()
        print(
            "MODEL OUTPUT: HIGHER READMISSION RISK"
        )

    else:

        print()
        print(
            "MODEL OUTPUT: LOWER READMISSION RISK"
        )


if __name__ == "__main__":
    main()