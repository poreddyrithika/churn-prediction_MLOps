import json
from pathlib import Path

import pandas as pd


LOG_PATH = Path("logs/inference/inference_log.jsonl")


def load_logs():
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Log file not found: {LOG_PATH}")

    with LOG_PATH.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        raise ValueError("Inference log is empty")

    return pd.DataFrame(records)


def analyze_predictions(df):
    total = len(df)

    prediction_counts = df["prediction"].value_counts().sort_index()
    prediction_percentages = (prediction_counts / total * 100).round(2)

    probability_stats = df["churn_probability"].describe()

    print("\n" + "=" * 60)
    print("PREDICTION DISTRIBUTION MONITORING")
    print("=" * 60)

    print(f"\nTotal predictions : {total}")

    print("\nPrediction counts:")
    print(prediction_counts)

    print("\nPrediction percentages:")
    print(prediction_percentages)

    print("\nChurn probability statistics:")
    print(probability_stats)

    print("\nModel information:")
    print(f"Model name    : {df['model_name'].iloc[-1]}")
    print(f"Model version : {df['model_version'].iloc[-1]}")
    print(f"Run ID        : {df['run_id'].iloc[-1]}")


def main():
    df = load_logs()
    analyze_predictions(df)


if __name__ == "__main__":
    main()