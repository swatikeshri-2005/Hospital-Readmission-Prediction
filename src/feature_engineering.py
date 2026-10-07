import re

import pandas as pd


def normalize_icd9(value) -> str:
    """
    Convert an ICD-9 diagnosis code into a broader
    diagnosis group.

    Examples:
        250.83 -> 250
        401.9  -> 401
        V58.67 -> V58
        E888.9 -> E88
    """

    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().upper()

    if value in {"", "?", "NAN"}:
        return "Unknown"

    # ICD-9 V and E codes
    if value.startswith(("V", "E")):
        return value[:3]

    # Standard numeric ICD-9 codes
    match = re.match(r"^(\d{3})", value)

    if match:
        return match.group(1)

    return "Other"


def add_engineered_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add engineered diagnosis features and normalize
    categorical ID columns.
    """

    output = df.copy()

    # --------------------------------------------------------
    # Diagnosis grouping
    # --------------------------------------------------------

    diagnosis_columns = [
        "diag_1",
        "diag_2",
        "diag_3",
    ]

    for column in diagnosis_columns:

        if column in output.columns:

            output[f"{column}_group"] = (
                output[column].apply(
                    normalize_icd9
                )
            )

    # --------------------------------------------------------
    # Treat ID fields as categorical
    # --------------------------------------------------------

    id_feature_columns = [
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
    ]

    for column in id_feature_columns:

        if column in output.columns:

            output[column] = (
                output[column]
                .astype("string")
            )

    return output