import json  # noqa: I001

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import (
    DATA_PROCESSED,
    MODEL_DIR,
    REPORT_DIR,
)


def main():

    model_path = (
        MODEL_DIR
        / "readmission_model.pkl"
    )

    test_path = (
        DATA_PROCESSED
        / "test_data.csv"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            "Model not found. "
            "Run `python -m src.train` first."
        )

    if not test_path.exists():

        raise FileNotFoundError(
            "Test data not found. "
            "Run `python -m src.train` first."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    bundle = joblib.load(
        model_path
    )

    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    test_data = pd.read_csv(
        test_path
    )

    X_test = test_data.drop(
        columns=["target"]
    )

    y_test = test_data["target"]

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    probabilities = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = {

        "accuracy":
            float(
                accuracy_score(
                    y_test,
                    predictions,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    y_test,
                    probabilities,
                )
            ),

        "pr_auc":
            float(
                average_precision_score(
                    y_test,
                    probabilities,
                )
            ),

        "threshold":
            float(threshold),

        "confusion_matrix":
            confusion_matrix(
                y_test,
                predictions,
            ).tolist(),

        "classification_report":
            classification_report(
                y_test,
                predictions,
                zero_division=0,
            ),
    }

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    print(
        f"Accuracy : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1       : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC   : "
        f"{metrics['pr_auc']:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print()
    print("Classification Report:")
    print(
        metrics[
            "classification_report"
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_DIR / "evaluation.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )


if __name__ == "__main__":
    main()