# Lab 8 — Pipeline Automation and CI Workflows

## Goal

Automate the normal Churn Prediction MLOps pipeline with GitHub Actions. Lab 8 does not create a second ML pipeline; it automatically executes the Lab 7 pipeline when relevant repository changes occur.

## CI flow

```text
Git push / pull request / manual trigger
                ↓
        install dependencies
                ↓
churn_prediction_pipeline.py
                ↓
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09
                ↓
        upload reports/artifacts
```

## Triggered changes

- `config.yaml`
- `src/mlops_pipeline/**`
- `requirements.txt`
- `data/raw/churn.csv`
- the CI workflow itself

## Repeated run demonstration

Change a model parameter in `config.yaml`, push the change, and observe a new MLflow run in:

```text
churn_prediction_model_training_lab7
```

Example:

```text
Run 1 → n_estimators=150
Run 2 → n_estimators=200
Run 3 → max_depth=8
```

## Quality gate

```yaml
quality_gate:
  metric: recall
  minimum_value: 0.70
```

If the metric is below the threshold, the pipeline exits before lifecycle promotion.

## Jenkins comparison

| Feature | GitHub Actions | Jenkins |
|---|---|---|
| Repository integration | Native to GitHub | Requires integration |
| Workflow definition | YAML | Jenkinsfile / UI |
| Hosted runners | Available | Normally self-managed |
| Setup for this lab | Simple | More infrastructure |
| Use in this project | Implemented | Compared conceptually |

## MLflow note

For CI runs to share one experiment history across temporary GitHub runners, set the GitHub Actions secret `MLFLOW_TRACKING_URI` to a persistent MLflow Tracking Server. See `MLFLOW_CI_SETUP.md`.
