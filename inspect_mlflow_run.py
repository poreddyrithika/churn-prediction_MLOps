import mlflow
from mlflow import MlflowClient

client = MlflowClient()

run_id = "e6b24fcde8574a95a7bc1ce6da74b0dc"

run = client.get_run(run_id)

print("RUN ID:", run.info.run_id)
print("STATUS:", run.info.status)
print("ARTIFACT URI:", run.info.artifact_uri)

print("\nARTIFACTS:")

artifacts = client.list_artifacts(run_id)

for artifact in artifacts:
    print(f"{artifact.path} | is_dir={artifact.is_dir}")

    if artifact.is_dir:
        children = client.list_artifacts(run_id, artifact.path)

        for child in children:
            print(f"  └── {child.path} | is_dir={child.is_dir}")