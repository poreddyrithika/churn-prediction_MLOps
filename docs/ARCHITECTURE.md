# MLOps Churn Prediction — Architecture

## Purpose

This repository implements a Telco Customer Churn prediction workflow with data validation, preprocessing, MLflow experiment/model tracking, a recall quality gate, model lifecycle decisions, monitoring-triggered retraining, and Docker/Kubernetes model serving.

## End-to-end flow

```text
GitHub Actions (manual or scheduled)
              |
              v
Self-hosted Windows runner
              |
              v
12_monitor_trigger.py
   |          |
   |          +--> Read existing drift reports
   |               - data drift
   |               - prediction drift
   |               - concept/performance drift
   |
   +--> Check for data/raw/new_churn.csv
              |
              v
churn_prediction_pipeline.py (when drift and a batch are present)
              |
              v
01 Data Loading + batch hash/identity checks
              |
              v
02 Raw validation --> 03 Preprocessing --> 04 Output validation
              |
              v
05 Training + MLflow/model registry
              |
              v
06 Evaluation --> 07 Recall quality gate (minimum 0.70)
              |
              v
08 Automated lifecycle decision
              |
        +-----+----------------+
        |                      |
   Not promoted             Promoted
        |                      |
        v                      v
Skip deployment        Prepare Production model bundle
                               |
                               v
                       Docker image + Kubernetes rollout
              |
              v
09 Production inference
```

## Main components

| Component | Responsibility |
|---|---|
| `src/mlops_pipeline/12_monitor_trigger.py` | Runs drift analysis, checks for an incoming batch, calls the main pipeline when appropriate, and writes a monitoring-trigger report. |
| `src/mlops_pipeline/12_drift_analysis.py` | Combines existing data-, prediction-, and concept-drift reports into an overall summary. It reads reports already produced by monitoring scripts. |
| `src/mlops_pipeline/01_load_data.py` | Validates incoming schema, hashes the batch, prevents re-ingestion of the same file, rejects overlapping customer IDs, archives the prior dataset, appends new rows, and records lineage. |
| `src/mlops_pipeline/churn_prediction_pipeline.py` | Runs ordered ML stages and stops safely if the incoming batch was already ingested. |
| `src/mlops_pipeline/05_train_registry.py` | Trains a candidate model, tracks the run in MLflow, and registers a model version. |
| `src/mlops_pipeline/07_quality_gate.py` | Requires recall to meet the configured minimum of 0.70. Passing this gate alone does not promote a model. |
| `src/mlops_pipeline/08_automate_lifecycle.py` | Compares the candidate with Production and records promotion/archive decisions in `reports/lifecycle_report.json`. |
| `src/mlops_pipeline/deploy_if_promoted.py` | Invokes deployment only when the lifecycle report says the candidate was promoted. |
| `src/mlops_pipeline/11_prepare_deployment.py` | Packages the registered Production model, its matching preprocessor, and metadata. |
| `deployment/deploy_production.py` | Builds or reuses a model-version Docker image, updates Kubernetes, waits for rollout, and writes a deployment report. |
| `.github/workflows/lab13_monitoring.yml` | Starts the monitoring trigger through the self-hosted Windows runner. |

## Data ingestion and idempotency

`data/raw/new_churn.csv` being present does not prove that a batch is new. The loader computes a SHA-256 hash and compares it with `data/raw/archive/dataset_lineage.json`. If the hash already exists, it reports `already_ingested` and does not append rows or retrain. For a new hash, it validates schema and customer IDs, archives the previous current dataset, appends the batch to `data/raw/churn.csv`, and updates lineage.

The Lab 13 demonstration used a simulated 100-row batch with IDs `NEW000201` through `NEW000300`. This verifies ingestion mechanics; it is not an external live-data connector.

## GitHub Actions execution

The workflow uses labels `[self-hosted, Windows, X64]` and invokes the project virtual environment at `C:\Users\jnane\Favorites\churn-prediction\venv\Scripts\python.exe`. It supports `workflow_dispatch` and a scheduled trigger configured for minute 17 of each hour in `Asia/Kolkata`.

The runner is configured in **interactive mode** (`.\run.cmd`), not as a Windows service. The PowerShell runner window must remain open, the computer must be awake and online, and local dependencies must be available at job time. This is not unattended-after-reboot operation.

## Verified lifecycle example

The end-to-end GitHub Actions test ingested 100 rows, increasing the raw dataset from 7,244 to 7,344 rows. Candidate model version 17 achieved accuracy 0.7672, recall 0.7219, F1 0.6221, and ROC-AUC 0.8452. The recall quality gate passed (`0.7219 >= 0.70`), but the lifecycle manager did not promote v17 because Production v1 had recall approximately 0.78075. Version 17 was archived, deployment was skipped, and Production inference continued using the Production model.

A separate deployment test verified Docker image reuse and a successful Kubernetes rollout for `churn-prediction-api:model-v1`. The candidate-v17 promotion-positive path was not exercised because v17 was correctly rejected.

## Runtime dependencies and limits

- MLflow, data, and deployment artifacts are available on the Windows project machine used by the self-hosted runner.
- Drift analysis combines existing drift reports; it does not fetch fresh external data.
- Docker Desktop and its Kubernetes cluster must be running and accessible through `kubectl` for deployment or cluster checks.
- Deployment occurs only after promotion. Passing training or the recall quality gate alone is insufficient.
