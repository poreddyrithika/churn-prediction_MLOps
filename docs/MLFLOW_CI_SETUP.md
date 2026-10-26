# MLflow Tracking for Labs 7–9

The project reads the MLflow Tracking URI from:

1. `MLFLOW_TRACKING_URI` environment variable, or
2. `mlflow.tracking_uri` in `config.yaml`.

For local development, leave the URI empty and MLflow can use local tracking storage.

For persistent CI experiment history, point `MLFLOW_TRACKING_URI` to a persistent MLflow Tracking Server and store that value as the GitHub Actions secret:

```text
MLFLOW_TRACKING_URI
```

Example local server:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

PowerShell:

```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

Linux/macOS:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

A GitHub-hosted runner cannot reach a server running only on your laptop. For CI, use an externally reachable Tracking Server or a suitable self-hosted runner.
