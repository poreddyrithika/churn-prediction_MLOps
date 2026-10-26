# Lab 9 — Pipeline Failure Handling and Recovery

## Goal

Intentionally break important parts of the existing MLOps pipeline and verify that the system fails safely or recovers through the Model Registry.

## Run locally

```bash
python src/mlops_pipeline/10_failure_injection.py
```

## Scenarios

| Scenario | Expected behavior |
|---|---|
| Missing raw dataset | HALT |
| Invalid datatype | HALT |
| Schema mismatch | HALT |
| Corrupted processed output | HALT |
| Missing local model artifact | Detect failure, recover the Production model from MLflow, then continue |

## Repeated execution

`config.yaml` controls the number of repetitions:

```yaml
lab9:
  repetitions: 3
```

The failure test creates:

```text
reports/failure_report.json
```

## Recovery path

```text
models/churn_model.pkl missing
            ↓
model evaluation detects the missing artifact
            ↓
load MLflow Production model
            ↓
restore local model artifact
            ↓
run evaluation again
            ↓
success
```

## Safety behavior

Corrupted input data is not silently repaired by the pipeline. Validation failures stop downstream stages so that bad data does not reach training.
