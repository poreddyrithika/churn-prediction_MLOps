import os

import mlflow
import yaml
from mlflow.tracking import MlflowClient


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    model_name = config["registry"]["model_name"]

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    effective_uri = mlflow.get_tracking_uri()

    print(f"Model: {model_name}")
    print(f"Tracking URI: {effective_uri}")
    print()
    print("VERSION | STAGE | RUN_ID")

    versions = client.search_model_versions(f"name='{model_name}'")
    versions = sorted(versions, key=lambda v: int(v.version), reverse=True)

    for version in versions:
        print(f"{version.version} | {version.current_stage} | {version.run_id}")


if __name__ == "__main__":
    main()