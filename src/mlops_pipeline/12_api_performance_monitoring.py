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


def analyze_performance(df):
    total_requests = len(df)

    success_count = (df["status"] == "success").sum()
    failure_count = (df["status"] == "failure").sum()

    success_rate = success_count / total_requests * 100
    failure_rate = failure_count / total_requests * 100

    latency = df["latency_ms"].dropna()

    print("\n" + "=" * 60)
    print("API PERFORMANCE MONITORING")
    print("=" * 60)

    print(f"\nTotal requests : {total_requests}")
    print(f"Successful     : {success_count}")
    print(f"Failed         : {failure_count}")
    print(f"Success rate   : {success_rate:.2f}%")
    print(f"Failure rate   : {failure_rate:.2f}%")

    if not latency.empty:
        print("\nLatency:")
        print(f"Average        : {latency.mean():.3f} ms")
        print(f"Median         : {latency.median():.3f} ms")
        print(f"P95            : {latency.quantile(0.95):.3f} ms")
        print(f"Minimum        : {latency.min():.3f} ms")
        print(f"Maximum        : {latency.max():.3f} ms")

    timestamps = pd.to_datetime(df["timestamp"], utc=True)

    elapsed_seconds = (
        timestamps.max() - timestamps.min()
    ).total_seconds()

    if elapsed_seconds > 0:
        throughput = total_requests / elapsed_seconds
        print(f"\nThroughput     : {throughput:.3f} requests/sec")
    else:
        print("\nThroughput     : Cannot calculate from one timestamp")

    print("\nStatus distribution:")
    print(df["status"].value_counts())


def main():
    df = load_logs()
    analyze_performance(df)


if __name__ == "__main__":
    main()