import pandas as pd
import numpy as np
from pathlib import Path


SOURCE_PATH = Path("data/processed/X_test_raw.csv")
OUTPUT_PATH = Path("data/monitoring/drift/evolved_data.csv")

RANDOM_SEED = 42


def create_evolved_dataset():
    rng = np.random.default_rng(RANDOM_SEED)

    df = pd.read_csv(SOURCE_PATH).copy()

    # ---------------------------------------------------------
    # 1. Numerical drift: MonthlyCharges
    # ---------------------------------------------------------
    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = (
            df["MonthlyCharges"] * rng.normal(1.15, 0.08, len(df))
        ).clip(lower=18.0, upper=120.0)

    # ---------------------------------------------------------
    # 2. Numerical drift: tenure
    # ---------------------------------------------------------
    if "tenure" in df.columns:
        df["tenure"] = (
            df["tenure"] + rng.normal(4, 6, len(df))
        ).clip(lower=0, upper=72).round().astype(int)

    # ---------------------------------------------------------
    # 3. Recalculate TotalCharges consistently
    # ---------------------------------------------------------
    if {"TotalCharges", "MonthlyCharges", "tenure"}.issubset(df.columns):
        df["TotalCharges"] = (
            df["MonthlyCharges"] * df["tenure"]
        ).round(2)

    # ---------------------------------------------------------
    # 4. Categorical drift: Contract
    # ---------------------------------------------------------
    if "Contract" in df.columns:
        df["Contract"] = rng.choice(
            ["Month-to-month", "One year", "Two year"],
            size=len(df),
            p=[0.75, 0.15, 0.10],
        )

    # ---------------------------------------------------------
    # 5. Categorical drift: InternetService
    # ---------------------------------------------------------
    if "InternetService" in df.columns:
        df["InternetService"] = rng.choice(
            ["DSL", "Fiber optic", "No"],
            size=len(df),
            p=[0.20, 0.65, 0.15],
        )

    # ---------------------------------------------------------
    # 6. Categorical drift: PaymentMethod
    # ---------------------------------------------------------
    if "PaymentMethod" in df.columns:
        df["PaymentMethod"] = rng.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            size=len(df),
            p=[0.55, 0.10, 0.20, 0.15],
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("EVOLVED DATASET CREATED")
    print("=" * 60)

    print(f"\nSource      : {SOURCE_PATH}")
    print(f"Output      : {OUTPUT_PATH}")
    print(f"Rows        : {len(df)}")
    print(f"Columns     : {len(df.columns)}")

    print("\nModified features:")
    print("- MonthlyCharges")
    print("- tenure")
    print("- TotalCharges")
    print("- Contract")
    print("- InternetService")
    print("- PaymentMethod")


if __name__ == "__main__":
    create_evolved_dataset()