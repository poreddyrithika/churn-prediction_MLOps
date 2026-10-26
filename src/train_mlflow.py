import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

def train_and_track(run_name="RandomForest_Baseline", params=None):
    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "class_weight": "balanced"
        }

    print(f"\n--- Starting MLflow Run: {run_name} ---")

    # 1. Load Data
    X_train = np.load('data/processed/X_train_final.npy')
    X_test = np.load('data/processed/X_test_final.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')

    # Set Experiment
    mlflow.set_experiment("Telco_Churn_Prediction")

    with mlflow.start_run(run_name=run_name):
        # 2. Log Parameters
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForest")

        # 3. Train Model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # 4. Evaluate Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

        # 5. Log Metrics to MLflow
        mlflow.log_metrics(metrics)
        print(f"Metrics logged: F1 = {metrics['f1_score']:.4f} | ROC-AUC = {metrics['roc_auc']:.4f}")

        # 6. Generate and Log Diagnostic Plots
        os.makedirs("artifacts", exist_ok=True)

        # Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm, cmap="Blues")
        ax_cm.set_title(f"Confusion Matrix - {run_name}")
        cm_path = "artifacts/confusion_matrix.png"
        fig_cm.savefig(cm_path, bbox_inches="tight")
        plt.close(fig_cm)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # ROC Curve
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax_roc)
        ax_roc.set_title(f"ROC Curve - {run_name}")
        roc_path = "artifacts/roc_curve.png"
        fig_roc.savefig(roc_path, bbox_inches="tight")
        plt.close(fig_roc)
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # 7. Log Preprocessing Metadata for Lineage
        if os.path.exists('data/processed/dataset_metadata.json'):
            mlflow.log_artifact('data/processed/dataset_metadata.json', artifact_path="metadata")

        # 8. Log Model Artifact
        mlflow.sklearn.log_model(sk_model=model, name="model")

        # Save local backup
        joblib.dump(model, 'models/random_forest_model.pkl')
        print(f"Run '{run_name}' successfully tracked!")

# run3
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    # Baseline Run1
    train_and_track(run_name="RandomForest_KNN_Imputation")

    # run2
    # tuned_params = {
    #     "n_estimators": 200,
    #     "max_depth": 5, 
    #     "random_state": 42,
    #     "class_weight": "balanced"
    # }
    
    # train_and_track(run_name="RandomForest_Shallow", params=tuned_params)

    # run3
    # print("\n--- Starting MLflow Run: LogisticRegression_Baseline ---")
    
    # X_train = np.load('data/processed/X_train_final.npy')
    # X_test = np.load('data/processed/X_test_final.npy')
    # y_train = np.load('data/processed/y_train.npy')
    # y_test = np.load('data/processed/y_test.npy')

    # mlflow.set_experiment("Telco_Churn_Prediction")

    # with mlflow.start_run(run_name="LogisticRegression_Baseline"):
        
    #     # 1. Log Parameters for Logistic Regression
    #     params = {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"}
    #     mlflow.log_params(params)
    #     mlflow.log_param("model_family", "LogisticRegression")
        
    #     # 2. Train Model
    #     model = LogisticRegression(**params)
    #     model.fit(X_train, y_train)
        
    #     # 3. Evaluate Predictions
    #     y_pred = model.predict(X_test)
    #     y_prob = model.predict_proba(X_test)[:, 1]
        
    #     metrics = {
    #         "accuracy": accuracy_score(y_test, y_pred),
    #         "precision": precision_score(y_test, y_pred),
    #         "recall": recall_score(y_test, y_pred),
    #         "f1_score": f1_score(y_test, y_pred),
    #         "roc_auc": roc_auc_score(y_test, y_prob)
    #     }
        
    #     mlflow.log_metrics(metrics)
    #     mlflow.sklearn.log_model(sk_model=model, name="model")
        
    #     print(f"Metrics logged: F1 = {metrics['f1_score']:.4f} | ROC-AUC = {metrics['roc_auc']:.4f}")