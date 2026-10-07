import json  # noqa: I001

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from .config import (
    DATA_RAW,
    DATA_PROCESSED,
    MODEL_DIR,
    REPORT_DIR,
    RANDOM_STATE,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

from .data_preprocessing import (
    load_raw_data,
    prepare_model_data,
    build_preprocessor,
)


# ============================================================
# METRICS
# ============================================================

def metric_summary(
    y_true,
    probabilities,
    threshold=0.5,
):
    """
    Calculate classification metrics.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "threshold": float(threshold),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "report": classification_report(
            y_true,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def find_best_threshold(
    y_true,
    probabilities,
):
    """
    Search for the threshold that maximizes F1.
    """

    thresholds = np.arange(
        0.20,
        0.81,
        0.01,
    )

    best_score = -1.0
    best_threshold = 0.50

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_score:

            best_score = score
            best_threshold = threshold

    return float(best_threshold)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HOSPITAL READMISSION PREDICTION - MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    raw = load_raw_data(
        DATA_RAW
    )

    print()
    print(
        f"Raw rows: {len(raw):,}"
    )

    X, y = prepare_model_data(
        raw
    )

    # Patient IDs used only for grouping
    groups = (
        raw["patient_nbr"]
        .astype(str)
    )

    print(
        f"Model features: {X.shape[1]}"
    )

    print()
    print("Target distribution:")
    print(y.value_counts())

    # --------------------------------------------------------
    # Patient-level train/test split
    # --------------------------------------------------------

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    trainval_index, test_index = next(
        first_split.split(
            X,
            y,
            groups=groups,
        )
    )

    # --------------------------------------------------------
    # Patient-level train/validation split
    # --------------------------------------------------------

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_relative, validation_relative = next(
        second_split.split(
            X.iloc[trainval_index],
            y.iloc[trainval_index],
            groups=groups.iloc[trainval_index],
        )
    )

    train_index = trainval_index[
        train_relative
    ]

    validation_index = trainval_index[
        validation_relative
    ]

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]

    X_validation = X.iloc[
        validation_index
    ]
    y_validation = y.iloc[
        validation_index
    ]

    X_train_validation = X.iloc[
        trainval_index
    ]
    y_train_validation = y.iloc[
        trainval_index
    ]

    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]

    print()
    print(
        f"Train rows:      {len(X_train):,}"
    )
    print(
        f"Validation rows: {len(X_validation):,}"
    )
    print(
        f"Test rows:       {len(X_test):,}"
    )

    # --------------------------------------------------------
    # Class imbalance
    # --------------------------------------------------------

    class_counts = np.bincount(
        y_train
    )

    negative_count = class_counts[0]
    positive_count = class_counts[1]

    scale_pos_weight = (
        negative_count
        / max(positive_count, 1)
    )

    print()
    print(
        f"Negative class: {negative_count:,}"
    )

    print(
        f"Positive class: {positive_count:,}"
    )

    print(
        f"XGBoost scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = {

        "logistic_regression":
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="saga",
                random_state=RANDOM_STATE,
            ),

        "random_forest":
            RandomForestClassifier(
                n_estimators=120,
                class_weight="balanced_subsample",
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),

        "xgboost":
            XGBClassifier(
                n_estimators=250,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
    }

    # --------------------------------------------------------
    # Train models
    # --------------------------------------------------------

    validation_results = {}
    fitted_models = {}

    for model_name, estimator in models.items():

        print()
        print("-" * 70)
        print(
            f"Training: {model_name}"
        )
        print("-" * 70)

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "model",
                    estimator,
                ),
            ]
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        validation_probabilities = (
            pipeline.predict_proba(
                X_validation
            )[:, 1]
        )

        metrics = metric_summary(
            y_validation,
            validation_probabilities,
            threshold=0.50,
        )

        best_threshold = (
            find_best_threshold(
                y_validation,
                validation_probabilities,
            )
        )

        best_predictions = (
            validation_probabilities
            >= best_threshold
        ).astype(int)

        metrics[
            "best_f1_threshold"
        ] = best_threshold

        metrics[
            "validation_recall_at_best_threshold"
        ] = float(
            recall_score(
                y_validation,
                best_predictions,
                zero_division=0,
            )
        )

        validation_results[
            model_name
        ] = metrics

        fitted_models[
            model_name
        ] = pipeline

        print(
            f"Validation ROC-AUC: "
            f"{metrics['roc_auc']:.4f}"
        )

        print(
            f"Validation PR-AUC:  "
            f"{metrics['pr_auc']:.4f}"
        )

        print(
            f"Validation F1:      "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"Best threshold:     "
            f"{best_threshold:.2f}"
        )

    # --------------------------------------------------------
    # Model selection
    # --------------------------------------------------------

    best_model_name = max(
        validation_results,
        key=lambda name:
        validation_results[name][
            "pr_auc"
        ],
    )

    threshold = (
        validation_results[
            best_model_name
        ][
            "best_f1_threshold"
        ]
    )

    print()
    print("=" * 70)
    print(
        f"BEST MODEL: {best_model_name}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Refit selected model on train + validation
    # --------------------------------------------------------

    best_pipeline = fitted_models[
        best_model_name
    ]

    print()
    print(
        "Refitting best model on "
        "train + validation data..."
    )

    best_pipeline.fit(
        X_train_validation,
        y_train_validation,
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    test_probabilities = (
        best_pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    test_metrics = metric_summary(
        y_test,
        test_probabilities,
        threshold=threshold,
    )

    # --------------------------------------------------------
    # Create folders
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATA_PROCESSED.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save model bundle
    # --------------------------------------------------------

    model_bundle = {
        "pipeline": best_pipeline,
        "threshold": threshold,
        "model_name": best_model_name,
    }

    joblib.dump(
        model_bundle,
        MODEL_DIR / "readmission_model.pkl",
    )

    # --------------------------------------------------------
    # Save test dataset
    # --------------------------------------------------------

    test_data = X_test.copy()

    test_data["target"] = (
        y_test.values
    )

    test_data.to_csv(
        DATA_PROCESSED / "test_data.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    preprocessor = (
        best_pipeline
        .named_steps["preprocessor"]
    )

    model = (
        best_pipeline
        .named_steps["model"]
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    if hasattr(
        model,
        "feature_importances_",
    ):

        importance = (
            model.feature_importances_
        )

    else:

        importance = np.abs(
            model.coef_[0]
        )

    feature_importance = (
        pd.DataFrame(
            {
                "feature":
                    feature_names,

                "importance":
                    importance,
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
    )

    feature_importance.to_csv(
        REPORT_DIR
        / "feature_importance.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Category metadata for Streamlit
    # --------------------------------------------------------

    categorical_values = {}

    for column in CATEGORICAL_FEATURES:

        values = (
            X_train_validation[column]
            .dropna()
            .astype(str)
            .value_counts()
            .head(100)
            .index
            .tolist()
        )

        categorical_values[
            column
        ] = values

    metadata = {
        "model_name":
            best_model_name,

        "threshold":
            threshold,

        "features":
            X.columns.tolist(),

        "numeric_features":
            NUMERIC_FEATURES,

        "categorical_features":
            CATEGORICAL_FEATURES,

        "categorical_values":
            categorical_values,
    }

    with open(
        MODEL_DIR
        / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    all_metrics = {
        "best_model":
            best_model_name,

        "validation":
            validation_results,

        "test":
            test_metrics,
    }

    with open(
        REPORT_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            all_metrics,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Test ROC-AUC : "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"Test PR-AUC  : "
        f"{test_metrics['pr_auc']:.4f}"
    )

    print(
        f"Test Recall  : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"Test F1      : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"Threshold    : "
        f"{threshold:.2f}"
    )

    print()
    print("Saved:")
    print(
        MODEL_DIR
        / "readmission_model.pkl"
    )

    print(
        REPORT_DIR
        / "metrics.json"
    )

    print(
        REPORT_DIR
        / "feature_importance.csv"
    )


if __name__ == "__main__":
    main()