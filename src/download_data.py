import pandas as pd  # noqa: F401
from ucimlrepo import fetch_ucirepo

from .config import DATA_RAW


def download_dataset() -> None:
    print("Downloading UCI dataset...")

    dataset = fetch_ucirepo(id=296)

    # ---------------------------------------------------------
    # IMPORTANT:
    # dataset.data.original contains the original UCI dataset.
    # It preserves encounter_id and patient_nbr.
    # ---------------------------------------------------------
    df = dataset.data.original.copy()

    print(f"Original dataset shape: {df.shape}")

    # Make sure target has the expected name
    if "readmitted" not in df.columns:
        raise ValueError(
            "Target column 'readmitted' was not found in the "
            "original UCI dataset."
        )

    print()
    print("Checking required columns...")

    required_columns = [
        "encounter_id",
        "patient_nbr",
        "readmitted",
    ]

    for column in required_columns:
        if column in df.columns:
            print(f"[OK] {column}")
        else:
            print(f"[MISSING] {column}")

    # Stop immediately if patient-level split columns are missing.
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing from original dataset: "
            f"{missing_columns}"
        )

    # ---------------------------------------------------------
    # Save dataset
    # ---------------------------------------------------------
    DATA_RAW.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(DATA_RAW, index=False)

    print()
    print("Dataset downloaded successfully.")
    print(f"Location: {DATA_RAW}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print()
    print("First 10 columns:")
    print(df.columns[:10].tolist())

    print()
    print("Required ID columns:")
    print(f"encounter_id: {df['encounter_id'].notna().sum():,}")
    print(f"patient_nbr:  {df['patient_nbr'].notna().sum():,}")

    print()
    print("Target distribution:")
    print(df["readmitted"].value_counts())


if __name__ == "__main__":
    download_dataset()