import mlflow
from mlflow import MlflowClient

client = MlflowClient()

model_name = "Telco_Churn_Production_Model"

model = client.get_registered_model(model_name)

print("MODEL:", model.name)
print("VERSIONS:")

versions = client.search_model_versions(f"name='{model_name}'")

for version in versions:
    print(
        f"version={version.version} "
        f"stage={version.current_stage} "
        f"run_id={version.run_id}"
    )