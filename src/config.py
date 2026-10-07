from pathlib import Path  # noqa: I001


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT_DIR / "data" / "raw" / "diabetic_data.csv"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"

MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = "readmitted"


# ============================================================
# IDENTIFIER COLUMNS
# ============================================================

ID_COLUMNS = [
    "encounter_id",
    "patient_nbr",
]


# ============================================================
# MODEL FEATURES
# ============================================================

MODEL_FEATURES = [
    "race",
    "gender",
    "age",

    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",

    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",

    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",

    "max_glu_serum",
    "A1Cresult",

    "metformin",
    "insulin",

    "change",
    "diabetesMed",

    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
]


# ============================================================
# NUMERIC FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

CATEGORICAL_FEATURES = [
    column
    for column in MODEL_FEATURES
    if column not in NUMERIC_FEATURES
]


# ============================================================
# RANDOM STATE
# ============================================================

RANDOM_STATE = 42