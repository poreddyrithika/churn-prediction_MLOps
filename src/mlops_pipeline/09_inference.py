import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def clean_dataframe(df, config):
    df = df.copy()
    target_column = config["data"]["target_column"]
    id_column = config["data"]["id_column"]

    if id_column in df.columns:
        df = df.drop(columns=[id_column])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if target_column in df.columns:
        df = df.drop(columns=[target_column])

    return df


def run_inference_demo():
    print("[INFO] Starting ML Pipeline Production Inference...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)
    model_name = config["registry"]["model_name"]
    model_uri = f"models:/{model_name}/Production"

    try:
        print(f"[INFO] Loading Production model from MLflow: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        print(f"[ERROR] Could not load Production model from MLflow: {exc}")
        print("[ERROR] Run 07_automate_lifecycle_mlops.py first.")
        return False

    preprocessor_path = config["paths"]["preprocessor"]
    raw_data_path = config["data"]["raw_path"]

    if not os.path.exists(preprocessor_path):
        print(f"[ERROR] Preprocessor not found at {preprocessor_path}")
        return False

    if not os.path.exists(raw_data_path):
        print(f"[ERROR] Raw dataset not found at {raw_data_path}")
        return False

    preprocessor = joblib.load(preprocessor_path)
    df = pd.read_csv(raw_data_path)

    sample = df.sample(n=5, random_state=7)
    X_sample = clean_dataframe(sample, config)
    X_sample = preprocessor.transform(X_sample)

    predictions = model.predict(X_sample)
    probabilities = model.predict_proba(X_sample)[:, 1]

    result = sample.copy()
    result["Predicted_Churn"] = predictions
    result["Churn_Probability"] = np.round(probabilities, 4)

    print("\n--- ML Pipeline Production Inference Results ---")
    print(result[["Predicted_Churn", "Churn_Probability"]].to_string(index=False))
    print("-----------------------------------------\n")

    os.makedirs("outputs", exist_ok=True)
    output_path = config["paths"]["inference_output"]
    result.to_csv(output_path, index=False)

    print(f"[SUCCESS] Predictions saved to {output_path}")
    return True


if __name__ == "__main__":
    passed = run_inference_demo()
    sys.exit(0 if passed else 1)
