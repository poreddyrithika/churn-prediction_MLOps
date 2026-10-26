# Runbook — Churn Prediction MLOps

## Before running

1. Start Docker Desktop and wait until the Docker engine and Kubernetes cluster are ready if the run needs Docker/Kubernetes operations.
2. Ensure the local project checkout is on `main` and up to date.
3. Confirm the project virtual environment exists at `C:\Users\jnane\Favorites\churn-prediction\venv`.
4. Start the interactive GitHub Actions runner in a separate PowerShell window:

   ```powershell
   cd C:\actions-runner
   .\run.cmd
   ```

   Keep this window open. GitHub should show `jnaneswar-mlops-runner` as `Idle` when no job is running.

5. For Docker/Kubernetes operations, verify:

   ```powershell
   docker info
   kubectl get nodes
   kubectl get deployment churn-api
   ```

## Start monitoring manually

1. Open the repository on GitHub.
2. Select **Actions → Lab 13 - End-to-End MLOps**.
3. Choose **Run workflow** and branch `main`.
4. Open the run and inspect each step and log.

The workflow is also configured to run at minute 17 of every hour in `Asia/Kolkata`. Because the runner is interactive, scheduled jobs can run only while the runner process is online and Windows is available.

## Interpret the monitoring report

Inspect `outputs/monitoring/monitor_trigger_report.json` from the project root.

| Status | Meaning |
|---|---|
| `no_drift` | The combined drift report did not identify drift; the pipeline is not called. |
| `drift_waiting_for_data` | Drift was detected, but the incoming batch file is missing. |
| `no_new_data_retraining_skipped` | The file exists, but its hash was already recorded as ingested. |
| `retraining_completed` | New data was ingested and the main pipeline returned success. Check the lifecycle report for the promotion decision. |
| `drift_analysis_failed` | Drift analysis failed; inspect the workflow log and the required source reports. |

A file named `new_churn.csv` is not automatically new data. The loader compares the file's SHA-256 hash with `data/raw/archive/dataset_lineage.json`. A new batch must pass schema validation and use customer IDs that do not overlap with the current dataset.

## Important reports

- `artifacts/data_load_report.json` — ingestion status and batch hash.
- `data/raw/archive/dataset_lineage.json` — ingested batches and dataset versions.
- `outputs/drift/drift_analysis_report.csv` — combined drift categories.
- `outputs/monitoring/monitor_trigger_report.json` — trigger decision.
- `reports/evaluation_report.json` — model evaluation metrics.
- `reports/lifecycle_report.json` — promotion decision.
- `reports/deployment_automation_report.json` — deployment outcome.
- `logs/churn_prediction_pipeline_run_log.json` — stage-by-stage status.

## Lifecycle behavior

The recall quality gate is configured at 0.70. A candidate passing the minimum is not necessarily promoted: the lifecycle manager compares it with Production. `deploy_if_promoted.py` skips deployment when `reports/lifecycle_report.json` says `promoted: false`.

## Troubleshooting

### Workflow stays queued

- In **Settings → Actions → Runners**, confirm the runner is online.
- Confirm the PowerShell window running `C:\actions-runner\run.cmd` remains open.
- Confirm the workflow requests labels `self-hosted`, `Windows`, and `X64`.

### Kubernetes commands fail

Start Docker Desktop, wait for Kubernetes to become ready, then run `kubectl get nodes`. Retry after the cluster is available. A Kubernetes check failure can be caused by the cluster being unavailable rather than by the monitoring code.

### Retraining is skipped

Check `artifacts/data_load_report.json` and `data/raw/archive/dataset_lineage.json`. `already_ingested` means the same file hash was consumed earlier. Do not delete or rewrite lineage just to force retraining. Obtain or generate a genuinely new, schema-valid batch with unique customer IDs.

### Model was not deployed

Inspect `reports/lifecycle_report.json`. If `promoted` is false, deployment was correctly skipped. Passing the quality gate alone does not mean the candidate should replace Production.

### Docker access denied from Python

Docker access has previously depended on the Windows integrity level of the virtual-environment Python executable. If direct Docker commands work but Python-launched Docker commands fail with a named-pipe access error, compare the integrity levels of the shell and `venv\Scripts\python.exe` before changing Docker permissions.

## Operational security

This runner executes workflow code on a personal Windows machine and can access local project resources. Avoid running untrusted pull-request code on it. Restrict repository permissions and workflow triggers to trusted contributors.
