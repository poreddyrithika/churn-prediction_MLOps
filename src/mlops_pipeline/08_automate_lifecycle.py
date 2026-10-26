import json
import os
import sys

import mlflow
import yaml
from mlflow.tracking import MlflowClient


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def save_lifecycle_report(report):
    os.makedirs("reports", exist_ok=True)
    with open("reports/lifecycle_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)


def automate_model_lifecycle():
    print("[INFO] Starting Automated Model Lifecycle Manager...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)
    client = MlflowClient()
    model_name = config["registry"]["model_name"]
    metric_to_optimize = config["registry"]["decision_metric"]

    report = {
        "status": "success",
        "model_name": model_name,
        "decision_metric": metric_to_optimize,
        "candidate_version": None,
        "candidate_run_id": None,
        "candidate_metric": None,
        "production_version_before": None,
        "production_metric_before": None,
        "promoted": False,
        "decision": None,
        "production_version_after": None
    }

    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")
        new_models = [mv for mv in all_versions if mv.current_stage == "None"]

        for mv in new_models:
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Staging",
                archive_existing_versions=False
            )
            print(f"[INFO] Version {mv.version} moved to Staging.")

        all_versions = client.search_model_versions(f"name='{model_name}'")
        models_staging = [mv for mv in all_versions if mv.current_stage == "Staging"]
        production_models = [mv for mv in all_versions if mv.current_stage == "Production"]
        production_model = production_models[0] if production_models else None

        if production_model:
            production_run = client.get_run(production_model.run_id)
            production_score = production_run.data.metrics.get(metric_to_optimize, 0.0)
            report["production_version_before"] = int(production_model.version)
            report["production_metric_before"] = production_score
        else:
            production_score = None

        if not models_staging:
            report["decision"] = "NO_CANDIDATE"
            report["production_version_after"] = report["production_version_before"]
            save_lifecycle_report(report)
            print("[INFO] No models in Staging. Nothing to promote.")
            print("[INFO] Lifecycle report saved to reports/lifecycle_report.json")
            return True

        best_model = None
        best_score = -1.0

        for mv in models_staging:
            run = client.get_run(mv.run_id)
            score = run.data.metrics.get(metric_to_optimize, 0.0)
            print(f"[INFO] Staging Version {mv.version} | {metric_to_optimize}={score:.4f}")

            if score > best_score:
                best_score = score
                best_model = mv

        report["candidate_version"] = int(best_model.version)
        report["candidate_run_id"] = best_model.run_id
        report["candidate_metric"] = best_score

        if production_model is None:
            move_to_production = True
            print("[INFO] No Production model exists. Staging model will become Production.")
        else:
            move_to_production = best_score > production_score

            if move_to_production:
                print("[SUCCESS] Staging model is better than Production.")
            else:
                print("[INFO] Production model remains unchanged.")

        if move_to_production:
            client.transition_model_version_stage(
                name=model_name,
                version=best_model.version,
                stage="Production",
                archive_existing_versions=False
            )

            if production_model:
                client.transition_model_version_stage(
                    name=model_name,
                    version=production_model.version,
                    stage="Archived",
                    archive_existing_versions=False
                )

            report["promoted"] = True
            report["decision"] = "PROMOTED"
            report["production_version_after"] = int(best_model.version)

            print(f"[SUCCESS] Version {best_model.version} is now Production.")
        else:
            client.transition_model_version_stage(
                name=model_name,
                version=best_model.version,
                stage="Archived",
                archive_existing_versions=False
            )

            report["promoted"] = False
            report["decision"] = "ARCHIVED"
            report["production_version_after"] = report["production_version_before"]

            print(f"[INFO] Version {best_model.version} archived because it did not improve.")

        final_versions = client.search_model_versions(f"name='{model_name}'")

        for mv in final_versions:
            if mv.current_stage == "Staging":
                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Archived",
                    archive_existing_versions=False
                )

        save_lifecycle_report(report)
        print("[SUCCESS] Automated Model Lifecycle execution complete!")
        print("[INFO] Lifecycle report saved to reports/lifecycle_report.json")
        return True

    except Exception as exc:
        report["status"] = "failed"
        report["decision"] = "ERROR"
        report["error"] = str(exc)
        save_lifecycle_report(report)
        print(f"[ERROR] Failed to automate model lifecycle: {exc}")
        return False


if __name__ == "__main__":
    passed = automate_model_lifecycle()
    sys.exit(0 if passed else 1)
