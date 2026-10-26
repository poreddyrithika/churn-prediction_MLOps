# Lab 13 — End-to-End MLOps Integration Report

## Objective

Integrate data versioning and validation, preprocessing, training, MLflow tracking/model registry, evaluation and quality gates, lifecycle decisions, deployment, monitoring, retraining triggers, reproducibility, and engineering documentation into one traceable system.

## Integrated execution

GitHub Actions runs on a self-hosted Windows runner and executes `src/mlops_pipeline/12_monitor_trigger.py` from the project virtual environment. The trigger runs drift analysis, checks for `data/raw/new_churn.csv`, and invokes `src/mlops_pipeline/churn_prediction_pipeline.py` when drift and a batch are present.

The main pipeline runs these stages:

1. Data Loading
2. Raw Data Validation
3. Preprocessing
4. Processed Output Validation
5. Training + MLflow + Model Registry
6. Model Evaluation
7. Model Quality Gate
8. Automated Model Lifecycle
9. Deployment Update (promotion-gated)
10. Production Inference

If the batch was already ingested, the pipeline stops safely rather than training again on the same data.

## Validation evidence

### Duplicate-batch behavior

A monitoring run detected drift and found `new_churn.csv`, but the loader reported `already_ingested`. The monitoring report showed `pipeline_status: passed`, `new_data_ingested: false`, and `retraining_executed: false`. This demonstrates idempotent handling of a previously consumed batch.

### New-batch ingestion and retraining

A simulated 100-row batch with distinct IDs `NEW000201`–`NEW000300` passed schema and identity validation. The loader archived the prior dataset as `churn_v03.csv`, appended the new rows, and increased the current raw dataset from 7,244 to 7,344 rows.

### Drift observations from the tested run

- Data drift was reported for `MonthlyCharges` and `TotalCharges`.
- Prediction-distribution KS statistic: 0.101490 (status `DRIFT`).
- Probability-distribution KS statistic: 0.229950 (status `DRIFT`).
- Four monitored concept/performance metrics were reported as degraded: accuracy, recall, F1, and ROC-AUC.

These are recorded demonstration results, not current live measurements.

### Candidate training and quality gate

Candidate model version 17 was trained and registered in MLflow:

| Metric | Candidate v17 |
|---|---:|
| Accuracy | 0.7672 |
| Recall | 0.7219 |
| F1 | 0.6221 |
| ROC-AUC | 0.8452 |

The recall quality gate required at least 0.70, so the candidate passed the gate.

### Lifecycle and deployment decision

Production version 1 had recall approximately 0.78075, higher than candidate v17's 0.7219. The lifecycle manager retained Production v1 and archived v17. The promotion-gated deployment step saw `promoted: false` and skipped deployment. Production inference then ran using the registered Production model.

This is an intentionally correct lifecycle outcome: passing the minimum quality gate does not automatically promote a candidate that fails to improve on Production.

### Deployment and Kubernetes

The standalone production-deployment automation was tested with Production model v1. It reused `churn-prediction-api:model-v1`, updated the Kubernetes Deployment, and reported a successful rollout. Kubernetes was observed with 3/3 replicas ready and three pods running during the recorded verification.

The candidate-v17 `promoted=true` path has not been demonstrated because the real candidate was correctly rejected. The non-promotion branch was verified in the integrated run, and the deployment mechanism was independently tested with Production v1.

## Important artifacts

| Artifact | Purpose |
|---|---|
| `artifacts/data_load_report.json` | Ingestion outcome, hash, and row counts. |
| `data/raw/archive/dataset_lineage.json` | Batch hashes and dataset lineage. |
| `outputs/drift/drift_analysis_report.csv` | Combined drift summary. |
| `outputs/monitoring/monitor_trigger_report.json` | Monitoring/retraining trigger decision. |
| `reports/evaluation_report.json` | Candidate model evaluation metrics. |
| `reports/lifecycle_report.json` | Candidate-versus-Production decision. |
| `reports/deployment_automation_report.json` | Deployment image, model version, Kubernetes target, and status. |
| `deployment/model_metadata.json` | Identity and lineage of the packaged Production model. |
| `logs/churn_prediction_pipeline_run_log.json` | Stage-by-stage pipeline status. |

Some runtime reports live in ignored local output directories. Their existence on the runner does not mean every report is committed to Git.

## GitHub Actions validation

The self-hosted runner used labels `self-hosted`, `Windows`, and `X64`. A diagnostic workflow successfully checked Python, Git, Docker, `kubectl`, and the project path. The end-to-end workflow then successfully ran monitoring and retraining through GitHub Actions.

The monitoring workflow is merged into `main` and is configured with `workflow_dispatch` and an hourly schedule at minute 17 in `Asia/Kolkata`. A successful scheduled run has not yet been confirmed in the recorded evidence; manual and push-triggered runs were used for integration testing.

## Limitations and follow-up

- The incoming batch was simulated data, not a feed from an external production source.
- The runner currently runs interactively via `C:\actions-runner\run.cmd`; the terminal must remain open and the computer must remain awake and online.
- Scheduled execution therefore depends on the runner being online at the scheduled time.
- The promotion-positive deployment path remains unverified with a naturally better candidate.
- Deployment/Kubernetes checks require Docker Desktop and Kubernetes to be running.
