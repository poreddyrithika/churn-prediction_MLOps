# Lab 7 — ML Pipeline Engineering

Lab 7 builds the core churn prediction ML pipeline.

Normal stages introduced in Lab 7:

```text
01 Load Data
02 Raw Data Validation
03 Preprocessing
04 Processed Output Validation
05 Training + MLflow + Model Registry
06 Model Evaluation
08 Automated Model Lifecycle
09 Production Inference
```

Lab 8 later inserts `07_quality_gate.py` between evaluation and lifecycle.

Run the pipeline after the Lab 7 stages are implemented:

```bash
python src/mlops_pipeline/churn_prediction_pipeline.py
```

Training uses one Random Forest trained on the full training split. Evaluation is done on the held-out test split using accuracy, precision, recall, F1 and ROC-AUC. MLflow records the parameters and metrics and registers each trained model under:

```text
Telco_Churn_Production_Model
```

The lifecycle uses the course-friendly stages:

```text
None → Staging → Production / Archived
```
