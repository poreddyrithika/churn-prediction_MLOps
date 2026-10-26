import pandas as pd
from pathlib import Path


DATA_DRIFT_PATH = Path("outputs/drift/data_drift_report.csv")
PREDICTION_DRIFT_PATH = Path("outputs/drift/prediction_drift_report.csv")
CONCEPT_DRIFT_PATH = Path("outputs/drift/concept_drift_report.csv")

OUTPUT_PATH = Path("outputs/drift/drift_analysis_report.csv")


def load_reports():
    data_drift = pd.read_csv(DATA_DRIFT_PATH)
    prediction_drift = pd.read_csv(PREDICTION_DRIFT_PATH)
    concept_drift = pd.read_csv(CONCEPT_DRIFT_PATH)

    return data_drift, prediction_drift, concept_drift


def analyze_data_drift(data_drift):
    drifted = data_drift[data_drift["status"] == "DRIFT"]

    return {
        "features_analyzed": len(data_drift),
        "features_with_drift": len(drifted),
        "drifted_features": ", ".join(drifted["feature"].tolist()),
    }


def analyze_prediction_drift(prediction_drift):
    row = prediction_drift.iloc[0]

    prediction_status = row["prediction_distribution_status"]
    probability_status = row["probability_distribution_status"]

    return {
        "prediction_distribution_status": prediction_status,
        "prediction_distribution_ks": row["prediction_distribution_ks"],
        "probability_distribution_status": probability_status,
        "probability_distribution_ks": row["probability_distribution_ks"],
    }


def analyze_concept_drift(concept_drift):
    report = concept_drift.copy()

    degraded = report[report["change"] < 0]

    return {
        "metrics_analyzed": len(report),
        "metrics_degraded": len(degraded),
        "degraded_metrics": ", ".join(degraded["metric"].tolist()),
    }


def create_summary(data_summary, prediction_summary, concept_summary):
    return pd.DataFrame([
        {
            "category": "Data Drift",
            "metric": "Features with drift",
            "value": data_summary["features_with_drift"],
            "status": "DRIFT" if data_summary["features_with_drift"] > 0 else "OK",
            "details": data_summary["drifted_features"],
        },
        {
            "category": "Prediction Drift",
            "metric": "Prediction distribution",
            "value": prediction_summary["prediction_distribution_ks"],
            "status": prediction_summary["prediction_distribution_status"],
            "details": "KS statistic",
        },
        {
            "category": "Prediction Drift",
            "metric": "Probability distribution",
            "value": prediction_summary["probability_distribution_ks"],
            "status": prediction_summary["probability_distribution_status"],
            "details": "KS statistic",
        },
        {
            "category": "Concept Drift",
            "metric": "Degraded performance metrics",
            "value": concept_summary["metrics_degraded"],
            "status": "DRIFT" if concept_summary["metrics_degraded"] > 0 else "OK",
            "details": concept_summary["degraded_metrics"],
        },
    ])


def print_analysis(data_summary, prediction_summary, concept_summary):
    print("=" * 70)
    print("COMBINED DRIFT ANALYSIS")
    print("=" * 70)

    print()
    print("1. DATA DRIFT")
    print("-" * 70)
    print(f"Features analyzed : {data_summary['features_analyzed']}")
    print(f"Features drifted  : {data_summary['features_with_drift']}")
    print(f"Drifted features  : {data_summary['drifted_features'] or 'None'}")

    print()
    print("2. PREDICTION DRIFT")
    print("-" * 70)
    print(
        f"Prediction distribution : "
        f"{prediction_summary['prediction_distribution_status']}"
    )
    print(
        f"Prediction KS           : "
        f"{prediction_summary['prediction_distribution_ks']:.6f}"
    )
    print(
        f"Probability distribution : "
        f"{prediction_summary['probability_distribution_status']}"
    )
    print(
        f"Probability KS          : "
        f"{prediction_summary['probability_distribution_ks']:.6f}"
    )

    print()
    print("3. CONCEPT DRIFT")
    print("-" * 70)
    print(f"Metrics analyzed : {concept_summary['metrics_analyzed']}")
    print(f"Metrics degraded : {concept_summary['metrics_degraded']}")
    print(f"Degraded metrics : {concept_summary['degraded_metrics'] or 'None'}")

    print()
    print("=" * 70)

    if (
        data_summary["features_with_drift"] > 0
        or prediction_summary["prediction_distribution_status"] == "DRIFT"
        or prediction_summary["probability_distribution_status"] == "DRIFT"
        or concept_summary["metrics_degraded"] > 0
    ):
        print("OVERALL STATUS : DRIFT / MODEL BEHAVIOR CHANGE OBSERVED")
    else:
        print("OVERALL STATUS : NO SIGNIFICANT DRIFT OBSERVED")

    print("=" * 70)


def main():
    data_drift, prediction_drift, concept_drift = load_reports()

    data_summary = analyze_data_drift(data_drift)
    prediction_summary = analyze_prediction_drift(prediction_drift)
    concept_summary = analyze_concept_drift(concept_drift)

    print_analysis(
        data_summary,
        prediction_summary,
        concept_summary,
    )

    summary = create_summary(
        data_summary,
        prediction_summary,
        concept_summary,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print()
    print(f"Combined report saved : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()