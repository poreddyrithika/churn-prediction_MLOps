import json
from pathlib import Path

import pandas as pd


LOG_PATH = Path("logs/inference/inference_log.jsonl")


def load_history():
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Log file not found: {LOG_PATH}")

    with LOG_PATH.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        raise ValueError("Inference history is empty")

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    return df


def show_summary(df):
    print("\n" + "=" * 60)
    print("REQUEST-RESPONSE HISTORY")
    print("=" * 60)

    print(f"\nTotal records : {len(df)}")

    print("\nRecent requests:")

    columns = [
        "request_id",
        "timestamp",
        "prediction",
        "churn_probability",
        "latency_ms",
        "status",
        "model_version",
    ]

    print(df[columns].tail(10).to_string(index=False))


def find_request(df, request_id):
    result = df[df["request_id"] == request_id]

    if result.empty:
        print(f"\nRequest not found: {request_id}")
        return

    print("\n" + "=" * 60)
    print("REQUEST DETAILS")
    print("=" * 60)

    print(result.to_string(index=False))


def show_failures(df):
    failures = df[df["status"] == "failure"]

    print("\n" + "=" * 60)
    print("FAILED REQUESTS")
    print("=" * 60)

    if failures.empty:
        print("\nNo failed requests.")
    else:
        print(f"\nFailed requests: {len(failures)}")
        print(
            failures[
                [
                    "request_id",
                    "timestamp",
                    "status",
                    "error",
                    "model_version",
                ]
            ].to_string(index=False)
        )


def main():
    df = load_history()

    show_summary(df)
    show_failures(df)

    request_id = df["request_id"].iloc[0]
    find_request(df, request_id)


if __name__ == "__main__":
    main()