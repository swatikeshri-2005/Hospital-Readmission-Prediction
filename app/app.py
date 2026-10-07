# app/app.py

import sys
from pathlib import Path

# ============================================================
# 1. PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. IMPORTS
# ============================================================

import joblib  # noqa: I001
import pandas as pd
import streamlit as st

from src.config import MODEL_DIR, REPORT_DIR
from src.data_preprocessing import clean_single_record


# ============================================================
# 3. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 4. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .metric-card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            margin-bottom: 15px;
        }

        .prediction-high {
            padding: 20px;
            border-radius: 12px;
            background-color: #fff1f2;
            border: 1px solid #fecdd3;
        }

        .prediction-low {
            padding: 20px;
            border-radius: 12px;
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
        }

        .info-box {
            padding: 15px;
            border-radius: 10px;
            background-color: #eff6ff;
            border: 1px solid #bfdbfe;
        }

        .small-text {
            color: #6b7280;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 5. PATHS
# ============================================================

MODEL_PATH = MODEL_DIR / "readmission_model.pkl"
METRICS_PATH = REPORT_DIR / "metrics.json"
METADATA_PATH = REPORT_DIR / "model_metadata.json"


# ============================================================
# 6. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    """
    Load the trained model bundle.

    The saved .pkl file contains:
        - pipeline
        - threshold
        - model_name
    """

    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


# ============================================================
# 7. LOAD METADATA
# ============================================================

@st.cache_data
def load_metadata():
    """
    Load optional model metadata.
    """

    if not METADATA_PATH.exists():
        return {}

    try:
        import json

        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception:  # noqa: BLE001
        return {}


# ============================================================
# 8. LOAD METRICS
# ============================================================

@st.cache_data
def load_metrics():
    """
    Load evaluation metrics from reports/metrics.json.
    """

    if not METRICS_PATH.exists():
        return {}

    try:
        import json

        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception:  # noqa: BLE001
        return {}


# ============================================================
# 9. LOAD EVERYTHING
# ============================================================

model_bundle = load_model()
metadata = load_metadata()
metrics = load_metrics()


if model_bundle is None:
    st.error(
        "❌ Model file not found.\n\n"
        "Please train the model first using:\n\n"
        "`python -m src.train`"
    )
    st.stop()


# ============================================================
# 10. EXTRACT PIPELINE FROM MODEL BUNDLE
# ============================================================

if isinstance(model_bundle, dict):

    # IMPORTANT:
    # Your model bundle contains:
    # pipeline
    # threshold
    # model_name

    model = model_bundle.get("pipeline")

    threshold = model_bundle.get(
        "threshold",
        metadata.get("threshold", 0.50)
    )

    model_name = model_bundle.get(
        "model_name",
        metadata.get("model_name", "Unknown")
    )

else:

    # Fallback in case a raw sklearn pipeline was saved
    model = model_bundle

    threshold = metadata.get(
        "threshold",
        0.50
    )

    model_name = metadata.get(
        "model_name",
        "Unknown"
    )


if model is None:
    st.error(
        "❌ The model file does not contain a valid pipeline."
    )
    st.stop()


# ============================================================
# 11. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏥 Readmission Predictor")

    st.markdown(
        """
        Predict whether a diabetic patient is likely to be
        **readmitted within 30 days** after hospital discharge.
        """
    )

    st.divider()

    st.subheader("Model Information")

    st.write(f"**Model:** {model_name}")

    st.write(
        f"**Decision Threshold:** {threshold:.2f}"
    )

    st.divider()

    st.subheader("Prediction Classes")

    st.write("🔴 **High Risk**")
    st.caption("Likely readmission within 30 days.")

    st.write("🟢 **Lower Risk**")
    st.caption("Less likely readmission within 30 days.")

    st.divider()

    st.caption(
        "Machine Learning Project\n"
        "UCI Diabetes 130-US Hospitals Dataset"
    )


# ============================================================
# 12. MAIN HEADER
# ============================================================

st.title("🏥 Hospital Readmission Prediction")

st.markdown(
    """
    ### Predict 30-Day Hospital Readmission Risk

    This application uses a machine learning model trained on
    hospital encounter data to estimate whether a diabetic patient
    is likely to be **readmitted within 30 days**.
    """
)

st.info(
    "⚠️ This application is for educational and research purposes "
    "only. It is not a medical diagnosis or clinical decision-making tool."
)


# ============================================================
# 13. MODEL PERFORMANCE
# ============================================================

st.subheader("📊 Model Performance")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)


# Default values
roc_auc = None
pr_auc = None
recall = None
f1 = None


# ------------------------------------------------------------
# Extract metrics safely
# ------------------------------------------------------------

if isinstance(metrics, dict):

    # Possible structure:
    # {
    #   "test": {
    #       "roc_auc": ...,
    #       ...
    #   }
    # }

    test_metrics = metrics.get("test", metrics)

    if isinstance(test_metrics, dict):

        roc_auc = test_metrics.get(
            "roc_auc",
            test_metrics.get("ROC-AUC")
        )

        pr_auc = test_metrics.get(
            "pr_auc",
            test_metrics.get("PR-AUC")
        )

        recall = test_metrics.get(
            "recall"
        )

        f1 = test_metrics.get(
            "f1",
            test_metrics.get("f1_score")
        )


with metric_col1:

    if roc_auc is not None:
        st.metric(
            "ROC-AUC",
            f"{float(roc_auc):.3f}"
        )
    else:
        st.metric(
            "ROC-AUC",
            "N/A"
        )


with metric_col2:

    if pr_auc is not None:
        st.metric(
            "PR-AUC",
            f"{float(pr_auc):.3f}"
        )
    else:
        st.metric(
            "PR-AUC",
            "N/A"
        )


with metric_col3:

    if recall is not None:
        st.metric(
            "Recall",
            f"{float(recall):.3f}"
        )
    else:
        st.metric(
            "Recall",
            "N/A"
        )


with metric_col4:

    if f1 is not None:
        st.metric(
            "F1 Score",
            f"{float(f1):.3f}"
        )
    else:
        st.metric(
            "F1 Score",
            "N/A"
        )


st.divider()


# ============================================================
# 14. PATIENT INPUT
# ============================================================

st.subheader("👤 Patient Information")

st.markdown(
    """
    Enter the patient's hospital encounter information below.
    Fields are based on the features used during model training.
    """
)


# ============================================================
# 15. BASIC DEMOGRAPHICS
# ============================================================

st.markdown("#### 🧑 Demographics")

col1, col2, col3 = st.columns(3)


with col1:

    race = st.selectbox(
        "Race",
        [
            "Caucasian",
            "AfricanAmerican",
            "Asian",
            "Hispanic",
            "Other",
            "Unknown",
        ],
        index=0,
    )


with col2:

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Unknown/Invalid",
        ],
        index=0,
    )


with col3:

    age = st.selectbox(
        "Age Group",
        [
            "[0-10)",
            "[10-20)",
            "[20-30)",
            "[30-40)",
            "[40-50)",
            "[50-60)",
            "[60-70)",
            "[70-80)",
            "[80-90)",
            "[90-100)",
        ],
        index=6,
    )


# ============================================================
# 16. ADMISSION INFORMATION
# ============================================================

st.markdown("#### 🏥 Admission Information")

col1, col2, col3 = st.columns(3)


with col1:

    admission_type_id = st.selectbox(
        "Admission Type ID",
        [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
        ],
        index=0,
        help="Type of hospital admission.",
    )


with col2:

    discharge_disposition_id = st.selectbox(
        "Discharge Disposition ID",
        [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "27",
            "28",
            "29",
        ],
        index=0,
        help="Disposition of the patient after discharge.",
    )


with col3:

    admission_source_id = st.selectbox(
        "Admission Source ID",
        [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "13",
            "14",
            "17",
            "20",
            "21",
            "22",
            "25",
        ],
        index=0,
        help="Source from which the patient was admitted.",
    )


# ============================================================
# 17. HOSPITAL STAY
# ============================================================

st.markdown("#### 🛏️ Hospital Stay")

col1, col2, col3, col4 = st.columns(4)


with col1:

    time_in_hospital = st.number_input(
        "Days in Hospital",
        min_value=1,
        max_value=30,
        value=4,
        step=1,
    )


with col2:

    num_lab_procedures = st.number_input(
        "Lab Procedures",
        min_value=0,
        max_value=200,
        value=40,
        step=1,
    )


with col3:

    num_procedures = st.number_input(
        "Procedures",
        min_value=0,
        max_value=10,
        value=1,
        step=1,
    )


with col4:

    num_medications = st.number_input(
        "Medications",
        min_value=0,
        max_value=100,
        value=15,
        step=1,
    )


# ============================================================
# 18. PREVIOUS UTILIZATION
# ============================================================

st.markdown("#### 📈 Previous Healthcare Utilization")

col1, col2, col3, col4 = st.columns(4)


with col1:

    number_outpatient = st.number_input(
        "Outpatient Visits",
        min_value=0,
        max_value=50,
        value=0,
        step=1,
    )


with col2:

    number_emergency = st.number_input(
        "Emergency Visits",
        min_value=0,
        max_value=50,
        value=0,
        step=1,
    )


with col3:

    number_inpatient = st.number_input(
        "Previous Inpatient Visits",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
    )


with col4:

    number_diagnoses = st.number_input(
        "Number of Diagnoses",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )


# ============================================================
# 19. LAB / DIABETES INFORMATION
# ============================================================

st.markdown("#### 🧪 Diabetes & Laboratory Information")

col1, col2, col3, col4 = st.columns(4)


with col1:

    max_glu_serum = st.selectbox(
        "Max Glucose Serum",
        [
            "None",
            "Norm",
            ">200",
            ">300",
        ],
        index=0,
    )


with col2:

    A1Cresult = st.selectbox(
        "A1C Result",
        [
            "None",
            "Norm",
            ">7",
            ">8",
        ],
        index=0,
    )


with col3:

    metformin = st.selectbox(
        "Metformin",
        [
            "No",
            "Steady",
            "Up",
            "Down",
        ],
        index=0,
    )


with col4:

    insulin = st.selectbox(
        "Insulin",
        [
            "No",
            "Steady",
            "Up",
            "Down",
        ],
        index=0,
    )


# ============================================================
# 20. MEDICATION CHANGE
# ============================================================

st.markdown("#### 💊 Medication Information")

col1, col2 = st.columns(2)


with col1:

    change = st.selectbox(
        "Medication Change",
        [
            "No",
            "Ch",
        ],
        index=0,
    )


with col2:

    diabetesMed = st.selectbox(
        "Diabetes Medication",
        [
            "No",
            "Yes",
        ],
        index=0,
    )


# ============================================================
# 21. DIAGNOSES
# ============================================================

st.markdown("#### 🩺 Diagnoses")

st.caption(
    "Enter ICD-9 diagnosis codes. The application automatically "
    "groups the codes for the machine learning model."
)

col1, col2, col3 = st.columns(3)


with col1:

    diag_1 = st.text_input(
        "Primary Diagnosis (ICD-9)",
        value="250.00",
        help="Example: 250.00",
    )


with col2:

    diag_2 = st.text_input(
        "Secondary Diagnosis (ICD-9)",
        value="401.9",
        help="Example: 401.9",
    )


with col3:

    diag_3 = st.text_input(
        "Tertiary Diagnosis (ICD-9)",
        value="272.4",
        help="Example: 272.4",
    )


# ============================================================
# 22. BUILD PATIENT RECORD
# ============================================================

patient_record = {
    "race": race,
    "gender": gender,
    "age": age,
    "admission_type_id": admission_type_id,
    "discharge_disposition_id": discharge_disposition_id,
    "admission_source_id": admission_source_id,
    "time_in_hospital": time_in_hospital,
    "num_lab_procedures": num_lab_procedures,
    "num_procedures": num_procedures,
    "num_medications": num_medications,
    "number_outpatient": number_outpatient,
    "number_emergency": number_emergency,
    "number_inpatient": number_inpatient,
    "number_diagnoses": number_diagnoses,
    "max_glu_serum": max_glu_serum,
    "A1Cresult": A1Cresult,
    "metformin": metformin,
    "insulin": insulin,
    "change": change,
    "diabetesMed": diabetesMed,
    "diag_1": diag_1,
    "diag_2": diag_2,
    "diag_3": diag_3,
}


# ============================================================
# 23. PREDICTION BUTTON
# ============================================================

st.divider()

predict_col1, predict_col2, predict_col3 = st.columns(
    [1, 2, 1]
)

with predict_col2:

    predict_button = st.button(
        "🔍 Predict Readmission Risk",
        use_container_width=True,
        type="primary",
    )


# ============================================================
# 24. MAKE PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # Prepare input
        # ----------------------------------------------------

        X_input = clean_single_record(patient_record)

        # ----------------------------------------------------
        # Probability prediction
        # ----------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(X_input)

            if probabilities.shape[1] >= 2:
                probability = float(
                    probabilities[0, 1]
                )
            else:
                probability = float(
                    probabilities[0, 0]
                )

        else:

            prediction = model.predict(X_input)

            probability = float(prediction[0])


        # ----------------------------------------------------
        # Apply saved threshold
        # ----------------------------------------------------

        prediction = int(
            probability >= float(threshold)
        )


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.subheader("📋 Prediction Result")


        result_col1, result_col2 = st.columns(2)


        # ----------------------------------------------------
        # Risk classification
        # ----------------------------------------------------

        with result_col1:

            if prediction == 1:

                st.markdown(
                    f"""
                    <div class="prediction-high">

                    <h2>🔴 High Readmission Risk</h2>

                    <p>
                    The model predicts that this patient has a
                    higher likelihood of being readmitted within
                    30 days.
                    </p>

                    <h3>
                    Probability: {probability:.1%}
                    </h3>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    f"""
                    <div class="prediction-low">

                    <h2>🟢 Lower Readmission Risk</h2>

                    <p>
                    The model predicts that this patient has a
                    lower likelihood of being readmitted within
                    30 days.
                    </p>

                    <h3>
                    Probability: {probability:.1%}
                    </h3>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


        # ----------------------------------------------------
        # Probability metrics
        # ----------------------------------------------------

        with result_col2:

            st.metric(
                "Readmission Probability",
                f"{probability:.1%}",
            )

            st.metric(
                "Decision Threshold",
                f"{float(threshold):.1%}",
            )

            st.metric(
                "Model",
                model_name,
            )


        # ====================================================
        # INTERPRETATION
        # ====================================================

        st.markdown("### 📌 Interpretation")

        if prediction == 1:

            st.warning(
                f"""
                The predicted probability ({probability:.1%}) is
                above the model's decision threshold
                ({float(threshold):.1%}).

                The patient is therefore classified as
                **higher risk for readmission within 30 days**.
                """
            )

        else:

            st.success(
                f"""
                The predicted probability ({probability:.1%}) is
                below the model's decision threshold
                ({float(threshold):.1%}).

                The patient is therefore classified as
                **lower risk for readmission within 30 days**.
                """
            )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        with st.expander("🔎 View Patient Input"):

            display_record = pd.DataFrame(
                [patient_record]
            ).T

            display_record.columns = ["Value"]

            st.dataframe(
                display_record,
                use_container_width=True,
            )


    except Exception as error:  # noqa: BLE001

        st.error(
            "❌ Prediction failed."
        )

        st.exception(error)


# ============================================================
# 25. ABOUT THE PROJECT
# ============================================================

st.divider()

st.subheader("📚 About This Project")

about_col1, about_col2 = st.columns(2)


with about_col1:

    st.markdown(
        """
        **Dataset**

        Diabetes 130-US Hospitals for Years 1999-2008

        **Source**

        UCI Machine Learning Repository

        **Target**

        Whether the patient is readmitted within 30 days.
        """
    )


with about_col2:

    st.markdown(
        """
        **Machine Learning Pipeline**

        1. Data preprocessing
        2. Missing-value handling
        3. Feature engineering
        4. One-hot encoding
        5. Feature scaling
        6. XGBoost classification
        7. Probability threshold optimization
        8. Patient-level train/validation/test splitting
        """
    )


# ============================================================
# 26. FOOTER
# ============================================================

st.divider()

st.caption(
    "Hospital Readmission Prediction using Machine Learning • "
    "Built with Python, Scikit-learn, XGBoost and Streamlit"
)

st.caption(
    "⚠️ For educational purposes only. "
    "Not intended for clinical diagnosis or medical decision-making."
)