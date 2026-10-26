import json
import os
import sys

import yaml


def run_quality_gate():
    print("[INFO] Starting Model Quality Gate...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    report_path = config["paths"]["evaluation_report"]
    metric_name = config["quality_gate"]["metric"]
    minimum_value = float(config["quality_gate"]["minimum_value"])

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Evaluation report not found: {report_path}")
        return False

    actual_value = float(report["metrics"].get(metric_name, 0.0))
    passed = actual_value >= minimum_value

    print(
        f"[INFO] Quality Gate: {metric_name}={actual_value:.4f} | "
        f"required>={minimum_value:.4f}"
    )

    os.makedirs(os.path.dirname(config["paths"]["quality_gate_report"]), exist_ok=True)

    gate_report = {
        "stage": "Lab 8 - Model Quality Gate",
        "metric": metric_name,
        "minimum_value": minimum_value,
        "actual_value": actual_value,
        "status": "PASSED" if passed else "FAILED",
    }

    with open(config["paths"]["quality_gate_report"], "w", encoding="utf-8") as f:
        json.dump(gate_report, f, indent=4)

    if passed:
        print("[SUCCESS] Model Quality Gate PASSED.")
    else:
        print("[ERROR] Model Quality Gate FAILED. Pipeline will stop before promotion.")

    return passed


if __name__ == "__main__":
    passed = run_quality_gate()
    sys.exit(0 if passed else 1)
