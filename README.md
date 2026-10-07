# 🏥 Hospital Readmission Prediction Using Machine Learning

An end-to-end machine learning project that predicts whether a diabetic patient is likely to be **readmitted to the hospital within 30 days** after discharge.

The project uses the **Diabetes 130-US Hospitals for Years 1999-2008** dataset from the UCI Machine Learning Repository and implements data preprocessing, feature engineering, patient-level data splitting, multiple machine learning models, threshold optimization, evaluation, and an interactive Streamlit application.

---

## 📌 Project Overview

Hospital readmissions are an important healthcare challenge because frequent readmissions can increase healthcare costs, reduce hospital capacity, and negatively affect patient outcomes.

This project builds a machine learning system that estimates the probability of **30-day hospital readmission** using patient demographics, hospital utilization, laboratory information, medications, admission details, and diagnosis information.

### 🎯 Objective

Predict:

- `1` → Patient is likely to be readmitted within 30 days
- `0` → Patient is not expected to be readmitted within 30 days

The model is specifically trained to identify the minority class:

```text
readmitted = <30
🚀 Key Features
📥 Automatic UCI dataset download
🧹 Data cleaning and missing-value handling
🔧 Feature engineering
🩺 ICD-9 diagnosis grouping
👥 Patient-level train/validation/test splitting
⚖️ Class imbalance handling
🤖 Multiple machine learning models
🌳 XGBoost-based final model
🎯 Decision-threshold optimization
📊 ROC-AUC, PR-AUC, Recall and F1 evaluation
💾 Saved production-ready preprocessing + model pipeline
🖥️ Interactive Streamlit web application
🧪 Automated testing
📁 Organized modular project structure
📊 Dataset

This project uses:

Diabetes 130-US Hospitals for Years 1999-2008

Dataset source:

UCI Machine Learning Repository

The dataset contains approximately:

101,766 patient encounters
50 original columns
Patient demographic information
Admission information
Diagnosis information
Medication information
Laboratory information
Hospital utilization information
Readmission outcome
Target Distribution

The original target contains three categories:

Readmission Status	Description
<30	Readmitted within 30 days
>30	Readmitted after 30 days
NO	Not readmitted

For this project, the problem is converted into a binary classification task:

Original Value	Model Target
<30	1
>30	0
NO	0

This creates an imbalanced classification problem where the positive class represents approximately 11% of encounters.

🧠 Machine Learning Approach
1. Data Loading

The dataset is downloaded automatically from the UCI Machine Learning Repository using ucimlrepo.

The original dataset is preserved so that important identifiers such as:

encounter_id
patient_nbr

are available for patient-level splitting.

2. Data Cleaning

The dataset contains missing values represented by:

?

These values are converted into:

NaN

Missing values are then handled inside the preprocessing pipeline.

Numeric Features

Median imputation is used.

Missing Numeric Value
        ↓
Median Imputation
        ↓
StandardScaler
Categorical Features

Most-frequent imputation is used.

Missing Categorical Value
        ↓
Most-Frequent Imputation
        ↓
One-Hot Encoding
🔧 Feature Engineering

Diagnosis codes are based on ICD-9 codes.

The project transforms the three diagnosis columns:

diag_1
diag_2
diag_3

into grouped diagnosis features:

diag_1_group
diag_2_group
diag_3_group

For example:

250.00
   ↓
250

Special ICD-9 codes beginning with V or E are handled separately.

This reduces the extremely high cardinality of raw diagnosis codes while retaining useful clinical information.

📋 Model Features

The final model uses 23 features.

Demographic Features
race
gender
age
Admission Features
admission_type_id
discharge_disposition_id
admission_source_id
Hospital Stay Features
time_in_hospital
num_lab_procedures
num_procedures
num_medications
Previous Healthcare Utilization
number_outpatient
number_emergency
number_inpatient
number_diagnoses
Laboratory Features
max_glu_serum
A1Cresult
Medication Features
metformin
insulin
change
diabetesMed
Diagnosis Features
diag_1_group
diag_2_group
diag_3_group
👥 Patient-Level Data Splitting

A major consideration in healthcare datasets is preventing the same patient from appearing in both training and testing data.

Instead of performing a simple random row split, this project performs a patient-level group split using patient_nbr.

This helps reduce data leakage and provides a more realistic evaluation.

Dataset Split
Total Encounters
       │
       ├── Training:   65,133
       │
       ├── Validation: 16,145
       │
       └── Test:       20,488
⚖️ Class Imbalance

The target is highly imbalanced.

Training distribution:

Negative Class: 57,874
Positive Class:  7,259

To help the XGBoost model focus on the minority class, the project uses:

scale_pos_weight ≈ 7.97

This improves the model's ability to identify patients who are actually readmitted within 30 days.

🤖 Models Evaluated

Three machine learning algorithms were evaluated.

Logistic Regression

Validation:

ROC-AUC: 0.6496
PR-AUC:  0.1916
F1:      0.2497
Random Forest

Validation:

ROC-AUC: 0.6669
PR-AUC:  0.1939
F1:      0.2615
XGBoost

Validation:

ROC-AUC: 0.6639
PR-AUC:  0.2044
F1:      0.2657

XGBoost was selected as the final model because it provided the strongest overall minority-class performance.

🏆 Final Model Performance

The final XGBoost model was evaluated on a held-out test set.

Metric	Score
ROC-AUC	0.6779
PR-AUC	0.2370
Recall	0.5099
F1 Score	0.2870
Decision Threshold	0.52
Interpretation

The model achieves approximately:

67.8% ROC-AUC
50.99% Recall
28.70% F1

for the 30-day readmission classification task.

Because identifying the minority class is important in this problem, recall, PR-AUC and F1-score are considered alongside ROC-AUC rather than relying on accuracy alone.

🎯 Threshold Optimization

Instead of automatically using:

threshold = 0.50

the project evaluates different probability thresholds on the validation set.

The selected threshold is:

0.52

Therefore:

Probability >= 0.52
        ↓
High Readmission Risk

while:

Probability < 0.52
        ↓
Lower Readmission Risk

This allows the classification decision to better reflect the imbalanced nature of the problem.

🖥️ Streamlit Application

The project includes an interactive Streamlit dashboard.

Run:

streamlit run app/app.py

The application allows users to enter:

Patient Information
Race
Gender
Age
Admission Information
Admission type
Discharge disposition
Admission source
Hospital Information
Days in hospital
Laboratory procedures
Procedures
Number of medications
Previous Healthcare Utilization
Outpatient visits
Emergency visits
Previous inpatient visits
Number of diagnoses
Diabetes Information
Glucose result
A1C result
Metformin
Insulin
Medication changes
Diabetes medication
Diagnosis Information
Primary diagnosis
Secondary diagnosis
Tertiary diagnosis

The application then returns:

Readmission Probability
        ↓
Threshold Comparison
        ↓
Risk Classification

Example:

Readmission Probability: 64.2%
Threshold:               52.0%

Prediction:
🔴 High Readmission Risk
🏗️ Project Structure
Hospital-Readmission-Prediction/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── diabetic_data.csv
│   │
│   └── processed/
│       └── .gitkeep
│
├── models/
│   └── readmission_model.pkl
│
├── reports/
│   ├── metrics.json
│   └── feature_importance.csv
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── download_data.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
│   └── test_model.py
│
├── .gitignore
├── requirements.txt
├── README.md
└── run_project.ps1
⚙️ Technologies Used
Programming Language
Python 3.12
Machine Learning
Scikit-learn
XGBoost
NumPy
Pandas
Data Visualization / Application
Streamlit
Dataset
UCI Machine Learning Repository
Model Persistence
Joblib
Testing
Pytest
Development
VS Code
PowerShell
Git
GitHub
🔧 Installation
1. Clone the Repository
git clone https://github.com/your-username/Hospital-Readmission-Prediction.git

Move into the project:

cd Hospital-Readmission-Prediction
2. Create a Virtual Environment

Windows PowerShell:

python -m venv .venv

Activate it:

.\.venv\Scripts\Activate.ps1

If PowerShell blocks script execution, run:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Then:

.\.venv\Scripts\Activate.ps1
3. Install Dependencies
pip install -r requirements.txt
▶️ Run the Project
Option 1 — Run the Complete Pipeline

From the project root:

.\run_project.ps1

The script performs:

Download Dataset
      ↓
Train Models
      ↓
Evaluate Model
      ↓
Run Sample Prediction
Option 2 — Run Individual Steps
Download Dataset
python -m src.download_data
Train Models
python -m src.train
Evaluate Model
python -m src.evaluate
Run Sample Prediction
python -m src.predict
🌐 Launch the Web Application

Run:

streamlit run app/app.py

Then open:

http://localhost:8501
🧪 Run Tests

Run:

pytest

For more detailed output:

pytest -v
💾 Saved Model

The trained model is saved as:

models/readmission_model.pkl

The model bundle contains:

{
    "pipeline": trained_pipeline,
    "threshold": 0.52,
    "model_name": "xgboost"
}

The pipeline includes both preprocessing and the trained XGBoost model.

Therefore, the Streamlit application can directly perform:

Raw Patient Input
       ↓
Preprocessing
       ↓
Feature Transformation
       ↓
XGBoost
       ↓
Probability
       ↓
Risk Classification
📈 Generated Reports

The training pipeline generates:

reports/
├── metrics.json
└── feature_importance.csv

metrics.json contains model evaluation results.

feature_importance.csv contains feature importance information from the trained model.

🔍 Example Prediction Flow

Example patient:

Age: 60-70
Gender: Female
Race: Caucasian

Hospital Stay: 6 days

Lab Procedures: 55
Procedures: 2
Medications: 20

Previous Outpatient Visits: 1
Emergency Visits: 1
Previous Inpatient Visits: 2

Diagnoses: 8

A1C: >8
Insulin: Steady
Diabetes Medication: Yes

The model generates a probability:

P(Readmission < 30 days) = 0.64

Since:

0.64 > 0.52

the prediction becomes:

🔴 High Readmission Risk
🧩 Important Machine Learning Design Decisions
Patient-Level Splitting

The project uses patient_nbr for grouped splitting to reduce patient-level leakage.

Pipeline-Based Preprocessing

Preprocessing is included inside the saved Scikit-learn pipeline.

This ensures that training and inference use the same transformations.

Class Imbalance Handling

XGBoost uses class weighting through:

scale_pos_weight
Probability Threshold

The final prediction threshold is optimized instead of blindly using 0.50.

PR-AUC

Because the positive class is relatively rare, PR-AUC is reported alongside ROC-AUC.

🔐 Data Leakage Prevention

The following identifiers are not used as predictive features:

encounter_id
patient_nbr

However, patient_nbr is retained during dataset preparation so it can be used for patient-level grouping.

This prevents the model from learning directly from patient identifiers.

⚠️ Limitations

This project has several limitations.

1. Historical Dataset

The data comes from hospital encounters between 1999 and 2008 and may not represent modern clinical practice.

2. Imbalanced Target

30-day readmission is a minority class, making prediction difficult.

3. Moderate Predictive Performance

The final ROC-AUC of approximately 0.68 indicates that the model has useful predictive signal but is not suitable as a standalone clinical decision system.

4. Missing Information

Important clinical variables may not be available in the dataset.

5. No Clinical Validation

The model has not been validated prospectively in a real hospital environment.

🚀 Future Improvements

Possible future improvements include:

Hyperparameter optimization with Optuna
SHAP-based model explainability
Calibration curves
Probability calibration
Cost-sensitive learning
More advanced ensemble models
Model monitoring
Data drift detection
Explainable AI dashboard
Fairness analysis across demographic groups
REST API deployment with FastAPI
Docker containerization
Cloud deployment
CI/CD pipeline
MLflow experiment tracking
Automated model retraining
Clinical validation with modern datasets
🧠 Future Architecture

The project can be extended into a production-style ML system:

                    ┌──────────────────┐
                    │ Hospital Dataset │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Data Validation  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Feature Engineer │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ ML Preprocessor  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    XGBoost       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Risk Probability │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Streamlit / API  │
                    └──────────────────┘
📚 Dataset Citation

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014).

Diabetes 130-US Hospitals for Years 1999-2008.

UCI Machine Learning Repository.

DOI:

https://doi.org/10.24432/C5230J

⚖️ License

The dataset is available under the CC BY 4.0 license.

Please refer to the UCI Machine Learning Repository for the dataset's licensing and attribution requirements.

⚠️ Disclaimer

This project is intended strictly for educational, research, and portfolio purposes.

It is not a medical device, does not provide medical advice, and should not be used to diagnose patients or make clinical decisions.

Predictions should not replace evaluation by qualified healthcare professionals.

👩‍💻 Author

Swati Keshri

B.Tech Computer Science Engineering

Interests:

Machine Learning
Artificial Intelligence
Data Science
Generative AI
MLOps
⭐ If You Find This Project Useful

Consider giving the repository a ⭐ on GitHub.

Hospital Readmission Prediction
├── Data Engineering
├── Feature Engineering
├── Machine Learning
├── Model Evaluation
├── Explainability
└── Streamlit Deployment

### Recommended GitHub description

Use this as the repository's short GitHub description:

> **End-to-end XGBoost ML system for predicting 30-day hospital readmission using the UCI Diabetes 130-US Hospitals dataset, with patient-level splitting, feature engineering, threshold optimization, and Streamlit deployment.**

### Recommended topics

```text
machine-learning
healthcare
healthcare-ai
hospital-readmission
xgboost
scikit-learn
python
streamlit
data-science
classification
predictive-analytics
feature-engineering
uci-dataset