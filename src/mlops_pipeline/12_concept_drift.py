import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from api.model_loader import ModelLoader
from api.preprocessing import clean_prediction_data


BASELINE_X_PATH = Path("data/processed/X_test_raw.csv")
BASELINE_Y_PATH = Path("data/processed/y_test.npy")
EVOLVED_PATH = Path("data/monitoring/drift/evolved_labeled_data.csv")
REPORT_PATH = Path("outputs/drift/concept_drift_report.csv")


def evaluate(model_loader, X, y):
    X_clean = clean_prediction_data(X)
    X_transformed = model_loader.preprocessor.transform(X_clean)

    predictions = model_loader.model.predict(X_transformed)
    probabilities = model_loader.model.predict_proba(X_transformed)[:, 1]

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
    }


def main():
    print("=" * 70)
    print("CONCEPT DRIFT ANALYSIS")
    print("=" * 70)

    baseline_X = pd.read_csv(BASELINE_X_PATH)
    baseline_y = np.load(BASELINE_Y_PATH)

    evolved = pd.read_csv(EVOLVED_PATH)

    evolved_X = evolved.drop(columns=["Churn"])
    evolved_y = evolved["Churn"].to_numpy()

    if len(baseline_X) != len(baseline_y):
        raise ValueError("Baseline feature/label count mismatch")

    if len(evolved_X) != len(evolved_y):
        raise ValueError("Evolved feature/label count mismatch")

    model_loader = ModelLoader()
    model_loader.load()

    print()
    print("Model information")
    print("-" * 60)
    print(f"Model name : {model_loader.model_name}")
    print(f"Version    : {model_loader.model_version}")
    print(f"Run ID     : {model_loader.run_id}")

    baseline_metrics = evaluate(
        model_loader,
        baseline_X,
        baseline_y,
    )

    evolved_metrics = evaluate(
        model_loader,
        evolved_X,
        evolved_y,
    )

    print()
    print("BASELINE PERFORMANCE")
    print("-" * 60)

    for metric, value in baseline_metrics.items():
        print(f"{metric.upper():10s}: {value:.4f}")

    print()
    print("EVOLVED PERFORMANCE")
    print("-" * 60)

    for metric, value in evolved_metrics.items():
        print(f"{metric.upper():10s}: {value:.4f}")

    results = []

    for metric in baseline_metrics:
        baseline_value = baseline_metrics[metric]
        evolved_value = evolved_metrics[metric]
        change = evolved_value - baseline_value

        results.append({
            "metric": metric,
            "baseline": baseline_value,
            "evolved": evolved_value,
            "change": change,
        })

    report = pd.DataFrame(results)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    print()
    print("=" * 70)
    print("PERFORMANCE CHANGE")
    print("=" * 70)

    print(report.to_string(index=False))

    print()
    print(f"Report saved : {REPORT_PATH}")


if __name__ == "__main__":
    main()