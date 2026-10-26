import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from api.model_loader import ModelLoader
from api.preprocessing import clean_prediction_data


X_TEST_PATH = Path("data/processed/X_test_raw.csv")
Y_TEST_PATH = Path("data/processed/y_test.npy")


def load_test_data():
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    if len(X_test) != len(y_test):
        raise ValueError(
            f"X_test and y_test size mismatch: {len(X_test)} != {len(y_test)}"
        )

    return X_test, y_test


def evaluate_model(model_loader, X_test, y_test):
    X_clean = clean_prediction_data(X_test)
    X_transformed = model_loader.preprocessor.transform(X_clean)

    predictions = model_loader.model.predict(X_transformed)
    probabilities = model_loader.model.predict_proba(X_transformed)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    return metrics


def print_results(metrics, model_loader, X_test, y_test):
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE MONITORING")
    print("=" * 60)

    print(f"\nEvaluation dataset : X_test_raw.csv")
    print(f"Test samples       : {len(X_test)}")
    print(f"Actual churn       : {int(y_test.sum())}")
    print(f"Actual non-churn   : {int((y_test == 0).sum())}")

    print(f"\nModel name    : {model_loader.model_name}")
    print(f"Model version : {model_loader.model_version}")
    print(f"Run ID        : {model_loader.run_id}")

    print("\nPerformance metrics:")

    for metric, value in metrics.items():
        print(f"{metric.capitalize():<12}: {value:.4f}")

    print("\nQuality gate:")
    print(f"Recall >= 0.70 : {metrics['recall'] >= 0.70}")


def main():
    X_test, y_test = load_test_data()

    model_loader = ModelLoader()
    model_loader.load()

    if not model_loader.is_ready():
        raise RuntimeError("Production model or preprocessor is not available")

    metrics = evaluate_model(model_loader, X_test, y_test)

    print_results(metrics, model_loader, X_test, y_test)


if __name__ == "__main__":
    main()