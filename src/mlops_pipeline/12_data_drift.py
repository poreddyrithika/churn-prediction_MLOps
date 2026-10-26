import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp

BASELINE_PATH = Path("data/processed/X_test_raw.csv")
EVOLVED_PATH = Path("data/monitoring/drift/evolved_data.csv")
REPORT_PATH = Path("outputs/drift/data_drift_report.csv")

NUMERICAL_COLUMNS = ["MonthlyCharges", "tenure", "TotalCharges"]
CATEGORICAL_COLUMNS = ["Contract", "InternetService", "PaymentMethod"]

KS_THRESHOLD = 0.10
JS_THRESHOLD = 0.10


def load_data():
    baseline = pd.read_csv(BASELINE_PATH)
    evolved = pd.read_csv(EVOLVED_PATH)

    if baseline.shape != evolved.shape:
        raise ValueError(
            f"Dataset shape mismatch: baseline={baseline.shape}, evolved={evolved.shape}"
        )

    return baseline, evolved


def calculate_js_divergence(baseline, evolved):
    categories = sorted(set(baseline.dropna().unique()) | set(evolved.dropna().unique()))

    baseline_counts = baseline.value_counts(normalize=True).reindex(categories, fill_value=0).values
    evolved_counts = evolved.value_counts(normalize=True).reindex(categories, fill_value=0).values

    baseline_counts = np.asarray(baseline_counts, dtype=float)
    evolved_counts = np.asarray(evolved_counts, dtype=float)

    baseline_counts /= baseline_counts.sum()
    evolved_counts /= evolved_counts.sum()

    midpoint = (baseline_counts + evolved_counts) / 2

    baseline_mask = baseline_counts > 0
    evolved_mask = evolved_counts > 0
    midpoint_mask = midpoint > 0

    kl_baseline = np.sum(
        baseline_counts[baseline_mask]
        * np.log2(baseline_counts[baseline_mask] / midpoint[baseline_mask])
    )

    kl_evolved = np.sum(
        evolved_counts[evolved_mask]
        * np.log2(evolved_counts[evolved_mask] / midpoint[evolved_mask])
    )

    return float((kl_baseline + kl_evolved) / 2)


def analyze_numerical_drift(baseline, evolved):
    results = []

    for column in NUMERICAL_COLUMNS:
        statistic, p_value = ks_2samp(
            baseline[column].dropna(),
            evolved[column].dropna(),
        )

        status = "DRIFT" if statistic >= KS_THRESHOLD else "OK"

        results.append({
            "feature": column,
            "type": "numerical",
            "metric": "KS",
            "score": round(float(statistic), 6),
            "p_value": round(float(p_value), 6),
            "threshold": KS_THRESHOLD,
            "status": status,
        })

    return results


def analyze_categorical_drift(baseline, evolved):
    results = []

    for column in CATEGORICAL_COLUMNS:
        score = calculate_js_divergence(
            baseline[column],
            evolved[column],
        )

        status = "DRIFT" if score >= JS_THRESHOLD else "OK"

        results.append({
            "feature": column,
            "type": "categorical",
            "metric": "JS",
            "score": round(score, 6),
            "p_value": None,
            "threshold": JS_THRESHOLD,
            "status": status,
        })

    return results


def create_drift_report(baseline, evolved):
    results = []

    results.extend(analyze_numerical_drift(baseline, evolved))
    results.extend(analyze_categorical_drift(baseline, evolved))

    report = pd.DataFrame(results)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    return report


def print_report(report):
    print("=" * 70)
    print("DATA DRIFT ANALYSIS")
    print("=" * 70)

    print(f"Baseline dataset : {BASELINE_PATH}")
    print(f"Evolved dataset  : {EVOLVED_PATH}")
    print()

    print(report.to_string(index=False))

    drift_count = (report["status"] == "DRIFT").sum()
    total_features = len(report)

    print()
    print("=" * 70)
    print(f"Features analyzed : {total_features}")
    print(f"Drift detected    : {drift_count}")
    print(f"Report saved      : {REPORT_PATH}")
    print("=" * 70)


def main():
    baseline, evolved = load_data()

    print(f"Baseline shape : {baseline.shape}")
    print(f"Evolved shape  : {evolved.shape}")
    print()

    report = create_drift_report(baseline, evolved)
    print_report(report)


if __name__ == "__main__":
    main()