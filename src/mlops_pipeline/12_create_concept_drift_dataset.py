import numpy as np
import pandas as pd
from pathlib import Path


FEATURES_PATH = Path("data/monitoring/drift/evolved_data.csv")
LABELS_PATH = Path("data/processed/y_test.npy")
OUTPUT_PATH = Path("data/monitoring/drift/evolved_labeled_data.csv")

RANDOM_SEED = 42


def create_concept_drift_dataset():
    rng = np.random.default_rng(RANDOM_SEED)

    X = pd.read_csv(FEATURES_PATH)
    y = np.load(LABELS_PATH).astype(int)

    if len(X) != len(y):
        raise ValueError(
            f"Feature/label mismatch: X={len(X)}, y={len(y)}"
        )

    y_evolved = y.copy()

    # ---------------------------------------------------------
    # Simulate a changed customer-churn relationship.
    #
    # Customers with month-to-month contracts become more
    # likely to churn in the evolved environment.
    # ---------------------------------------------------------
    if "Contract" in X.columns:
        month_to_month = X["Contract"].eq("Month-to-month")

        affected_indices = np.where(month_to_month)[0]

        flip_count = int(len(affected_indices) * 0.25)

        selected_indices = rng.choice(
            affected_indices,
            size=flip_count,
            replace=False,
        )

        y_evolved[selected_indices] = 1

    # ---------------------------------------------------------
    # Simulate another relationship change:
    # high MonthlyCharges become more strongly associated
    # with churn.
    # ---------------------------------------------------------
    if "MonthlyCharges" in X.columns:
        high_charge_indices = np.where(
            X["MonthlyCharges"] > X["MonthlyCharges"].quantile(0.75)
        )[0]

        flip_count = int(len(high_charge_indices) * 0.15)

        selected_indices = rng.choice(
            high_charge_indices,
            size=flip_count,
            replace=False,
        )

        y_evolved[selected_indices] = 1

    result = X.copy()
    result["Churn"] = y_evolved

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("=" * 70)
    print("CONCEPT DRIFT DATASET CREATED")
    print("=" * 70)

    print(f"Feature source : {FEATURES_PATH}")
    print(f"Label source   : {LABELS_PATH}")
    print(f"Output         : {OUTPUT_PATH}")
    print(f"Rows           : {len(result)}")
    print(f"Columns        : {len(result.columns)}")

    print()
    print("Baseline label distribution:")
    print(pd.Series(y).value_counts().sort_index())

    print()
    print("Evolved label distribution:")
    print(pd.Series(y_evolved).value_counts().sort_index())

    print()
    print("Label changes:")
    print(f"Changed labels : {(y != y_evolved).sum()}")


if __name__ == "__main__":
    create_concept_drift_dataset()