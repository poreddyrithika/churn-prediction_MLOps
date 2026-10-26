from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil

import pandas as pd
import yaml

from validation import validate_schema

CONFIG_PATH = Path("config.yaml")


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_lineage(lineage_path):
    if not lineage_path.exists():
        return []

    try:
        with lineage_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_lineage(lineage_path, lineage):
    lineage_path.parent.mkdir(parents=True, exist_ok=True)

    with lineage_path.open("w", encoding="utf-8") as f:
        json.dump(lineage, f, indent=2)


def get_next_archive_version(archive_dir):
    versions = []

    for file_path in archive_dir.glob("churn_v*.csv"):
        try:
            version = int(file_path.stem.replace("churn_v", ""))
            versions.append(version)
        except ValueError:
            continue

    return max(versions, default=0) + 1


def validate_new_batch_identity(new_df, current_df):
    if new_df.empty:
        raise ValueError("new_churn.csv is empty.")

    if current_df.empty:
        raise ValueError("Current churn.csv is empty.")

    if not new_df["customerID"].is_unique:
        raise ValueError("new_churn.csv contains duplicate customerID values.")

    if current_df["customerID"].duplicated().any():
        raise ValueError("Current churn.csv contains duplicate customerID values.")

    current_ids = set(current_df["customerID"].astype(str))
    new_ids = set(new_df["customerID"].astype(str))
    overlapping_ids = current_ids.intersection(new_ids)

    if overlapping_ids:
        raise ValueError(f"new_churn.csv contains {len(overlapping_ids)} customer IDs already present in churn.csv.")

    print("[SUCCESS] New batch identity validation passed.")


def create_data_load_report(report_path, report):
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)


def ingest_new_data(raw_path, new_data_path, archive_dir, lineage_path, report_path):
    current_df = pd.read_csv(raw_path)
    new_df = pd.read_csv(new_data_path)

    print(f"[INFO] Current dataset rows: {len(current_df)}")
    print(f"[INFO] New dataset rows: {len(new_df)}")

    print("[INFO] Validating new dataset schema...")
    schema_valid = validate_schema(new_df, output_report_name="new_data_validation.csv")

    if not schema_valid:
        raise ValueError("New dataset failed schema validation. Ingestion stopped.")

    new_data_hash = calculate_file_hash(new_data_path)
    print(f"[INFO] New dataset SHA256: {new_data_hash}")

    lineage = load_lineage(lineage_path)
    already_ingested = any(record.get("new_data_hash") == new_data_hash for record in lineage)

    if already_ingested:
        print("[INFO] This new dataset batch has already been ingested.")
        print("[INFO] No archive or retraining action will be performed.")

        report = {
            "status": "already_ingested",
            "ingestion_performed": False,
            "new_data_hash": new_data_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_rows": len(current_df),
            "new_rows": len(new_df),
        }

        create_data_load_report(report_path, report)
        return False

    validate_new_batch_identity(new_df, current_df)

    archive_dir.mkdir(parents=True, exist_ok=True)

    archive_version = get_next_archive_version(archive_dir)
    archive_filename = f"churn_v{archive_version:02d}.csv"
    archive_path = archive_dir / archive_filename

    shutil.copy2(raw_path, archive_path)
    print(f"[SUCCESS] Current dataset archived as: {archive_filename}")

    previous_rows = len(current_df)
    new_rows = len(new_df)
    updated_df = pd.concat([current_df, new_df], ignore_index=True)

    temp_path = raw_path.with_suffix(".tmp.csv")
    updated_df.to_csv(temp_path, index=False)
    os.replace(temp_path, raw_path)

    print(f"[SUCCESS] New data appended to {raw_path}")
    print(f"[INFO] Previous rows: {previous_rows}")
    print(f"[INFO] Added rows: {new_rows}")
    print(f"[INFO] Current rows: {len(updated_df)}")

    lineage_record = {
        "dataset_version": f"v{archive_version + 1:02d}",
        "archive_file": str(archive_path).replace("\\", "/"),
        "source_file": str(new_data_path).replace("\\", "/"),
        "new_data_hash": new_data_hash,
        "previous_rows": previous_rows,
        "new_rows": new_rows,
        "current_rows": len(updated_df),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    lineage.append(lineage_record)
    save_lineage(lineage_path, lineage)

    report = {
        "status": "ingested",
        "ingestion_performed": True,
        "dataset_version": lineage_record["dataset_version"],
        "archive_file": lineage_record["archive_file"],
        "source_file": lineage_record["source_file"],
        "new_data_hash": new_data_hash,
        "previous_rows": previous_rows,
        "new_rows": new_rows,
        "current_rows": len(updated_df),
        "timestamp": lineage_record["timestamp"],
    }

    create_data_load_report(report_path, report)
    print("[SUCCESS] Dataset ingestion completed.")
    return True

def load_current_data(raw_path):
    if not raw_path.exists():
        raise FileNotFoundError(f"Current dataset not found: {raw_path}")

    df = pd.read_csv(raw_path)

    if df.empty:
        raise ValueError(f"Current dataset is empty: {raw_path}")

    print(f"[INFO] Loaded dataset: {raw_path}")
    print(f"[INFO] Dataset shape: {df.shape}")

    return df


def main():
    config = load_config()

    raw_path = Path(config["data"]["raw_path"])
    new_data_path = raw_path.parent / "new_churn.csv"
    archive_dir = raw_path.parent / "archive"
    lineage_path = archive_dir / "dataset_lineage.json"
    report_path = Path("artifacts/data_load_report.json")

    print("=" * 70)
    print("DATA LOADING")
    print("=" * 70)

    if not raw_path.exists():
        raise FileNotFoundError(f"Current dataset not found: {raw_path}")

    if new_data_path.exists():
        print("[INFO] New dataset detected.")
        ingest_new_data(raw_path, new_data_path, archive_dir, lineage_path, report_path)
    else:
        print("[INFO] No new dataset detected.")
        print("[INFO] Continuing with current dataset.")

        current_df = load_current_data(raw_path)

        report = {
            "status": "no_new_data",
            "ingestion_performed": False,
            "current_rows": len(current_df),
            "current_columns": len(current_df.columns),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        create_data_load_report(report_path, report)

    final_df = load_current_data(raw_path)

    print(f"[SUCCESS] Final dataset shape: {final_df.shape}")
    print("[SUCCESS] Data loading stage completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()