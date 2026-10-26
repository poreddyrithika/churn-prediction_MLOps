# Lab 11 — Containerization and Kubernetes Deployment Report

## 1. Objective

The objective of Lab 11 was to containerize the Churn Prediction
ML application, validate its portability and prediction consistency,
deploy the containerized API using Kubernetes, demonstrate scaling,
simulate deployment failures, and validate recovery.

---

## 2. Deployment Architecture

The deployment architecture follows this flow:

MLflow Production Model
        ↓
Production Model Version
        ↓
Matching Model + Preprocessor
        ↓
Deployment Bundle
        ↓
Docker Image
        ↓
Kubernetes Deployment
        ↓
Kubernetes Pods
        ↓
Kubernetes Service
        ↓
FastAPI API

MLflow is used as the model lifecycle/source-of-truth system.

The deployment bundle contains the model and preprocessing artifact
associated with the same MLflow Production run.

---

## 3. MLflow Production Model

Model Name:

Telco_Churn_Production_Model

Production Version:

1

Stage:

Production

MLflow Run ID:

e6b24fcde8574a95a7bc1ce6da74b0dc

Model ID:

m-0b44c639cae04d2da060f7b7f01e34d1

The deployment preparation process extracted the Production model
and the matching preprocessor from the same MLflow run.

---

## 4. Deployment Bundle

The deployment bundle contains:

deployment/
├── model.pkl
├── preprocessor.pkl
└── model_metadata.json

The metadata records the MLflow model name, version, stage,
Run ID, model ID, and source information.

This allows the deployed model to retain its provenance.

---

## 5. Docker Containerization

Docker was used to package the FastAPI application together with
its runtime dependencies and verified ML artifacts.

Docker image:

churn-prediction-api:1.0

The API container contains:

- FastAPI
- Uvicorn
- Pydantic
- NumPy
- Pandas
- scikit-learn
- joblib
- PyYAML
- FastAPI application code
- Deployment bundle
- Configuration file

MLflow is not required during API runtime because the verified
deployment bundle is included directly in the image.

---

## 6. Docker Image Creation

The image was built using:

docker build -t churn-prediction-api:1.0 .

The image was successfully created.

The container was started using:

docker run --name churn-api -p 8000:8000 churn-prediction-api:1.0

FastAPI successfully started on port 8000.

---

## 7. Containerized API Validation

The containerized API was tested using:

GET /health

GET /metadata

POST /predict

The health endpoint returned:

status = healthy

The metadata endpoint confirmed:

service = Churn Prediction API
version = 1.0.0
model_name = Telco_Churn_Production_Model
model_uri = /app/deployment/model.pkl
model_loaded = True
preprocessor_loaded = True

The prediction endpoint also successfully returned predictions.

---

## 8. Local vs Container Prediction Consistency

The same customer input was sent to the local API and the
containerized API.

Local prediction:

0

Container prediction:

0

Local churn probability:

0.41734259770833143

Container churn probability:

0.41734259770833143

Prediction match:

True

Probability match:

True

Therefore, the containerized deployment produced the same
prediction and probability as the local deployment.

This demonstrates prediction consistency across environments.

---

## 9. Deployment Failure Simulation

A missing model artifact failure was intentionally simulated.

The model file was renamed inside the running container:

docker exec churn-api mv /app/deployment/model.pkl /app/deployment/model_backup.pkl

The container was then restarted.

During application startup, ModelLoader attempted to load:

/app/deployment/model.pkl

The application failed with:

FileNotFoundError: Model not found: /app/deployment/model.pkl

The container exited with status code 3.

This successfully demonstrated a deployment failure caused by
a missing model artifact.

---

## 10. Failure Debugging

The failed container was inspected using Docker commands.

The container status was:

exited

Exit code:

3

Docker logs showed:

FileNotFoundError: Model not found: /app/deployment/model.pkl

Root cause:

The required model artifact was missing from the deployment bundle
inside the running container.

---

## 11. Deployment Recovery

The corrupted container was not manually repaired.

Instead, the failed container was removed:

docker rm churn-api

The container was recreated from the known-good immutable image:

docker run --name churn-api -p 8000:8000 churn-prediction-api:1.0

The recovery was successful.

The application again loaded:

- Model
- Preprocessor
- Model metadata

FastAPI successfully started after recovery.

---

# 12. Kubernetes Deployment

Docker Desktop Kubernetes was enabled using Kind.

Kubernetes version:

v1.36.1

Node:

desktop-control-plane

Node status:

Ready

The Kubernetes Deployment was created using:

k8s/deployment.yaml

Deployment name:

churn-api

Docker image:

churn-prediction-api:1.0

Initial replicas:

1

The deployment successfully created a running Pod.

---

## 13. Kubernetes Service

A Kubernetes Service was created using:

k8s/service.yaml

Service name:

churn-api-service

Service type:

NodePort

Service port:

8000

Target port:

8000

NodePort:

30383

Cluster IP:

10.96.188.64

The Service selected Pods using:

app=churn-api

The Service successfully connected to the FastAPI Pods.

---

## 14. Kubernetes Scaling

The Deployment was scaled from one replica to three replicas:

kubectl scale deployment churn-api --replicas=3

The resulting Deployment status was:

3/3 READY
3/3 UP-TO-DATE
3/3 AVAILABLE

Three Pods were running simultaneously.

This demonstrated Kubernetes horizontal scaling.

---

## 15. Kubernetes Service Interruption

One running Pod was intentionally deleted:

kubectl delete pod <pod-name>

The deleted Pod was:

churn-api-84f5fcd955-cc6gt

Kubernetes detected that the actual number of Pods was below
the desired replica count.

A replacement Pod was automatically created:

churn-api-84f5fcd955-llpfs

The final state returned to three running Pods.

---

## 16. Kubernetes Recovery Validation

After recovery, the Deployment showed:

3/3 READY
3/3 UP-TO-DATE
3/3 AVAILABLE

The Service had three active endpoints:

10.244.0.5:8000
10.244.0.6:8000
10.244.0.8:8000

The application was then validated through the Kubernetes Service.

Health endpoint:

/health

Result:

healthy

Metadata endpoint confirmed:

service             : Churn Prediction API
version             : 1.0.0
model_name          : Telco_Churn_Production_Model
model_uri           : /app/deployment/model.pkl
model_loaded        : True
preprocessor_loaded : True

Therefore, Kubernetes successfully recovered the failed Pod and
the application remained operational.

---

# 17. Lab 11 Results

The following Lab 11 requirements were demonstrated:

- ML deployment containerization
- Docker image creation
- Dependency packaging
- FastAPI container execution
- Containerized API validation
- Local/container prediction consistency
- Deployment failure simulation
- Failure diagnosis
- Deployment recovery
- Kubernetes deployment
- Kubernetes service exposure
- Kubernetes scaling
- Kubernetes service interruption
- Kubernetes automatic recovery
- Post-recovery API validation

---

# 18. Final Architecture

MLflow
  ↓
Production Model
  ↓
Deployment Bundle
  ↓
Docker Image
  ↓
Kubernetes Deployment
  ↓
3 FastAPI Pods
  ↓
Kubernetes NodePort Service
  ↓
Client

The system demonstrates an end-to-end ML deployment workflow
from model lifecycle management to containerized and
Kubernetes-managed inference.

---

# 19. Next Step

The remaining Lab 11 activities are:

11.19 — Deployment and Debugging Report

11.20 — Final Lab 11 Validation

After Lab 11, Lab 12 will focus on:

- Monitoring
- Inference logging
- Prediction distributions
- Latency and throughput
- Failure monitoring
- Data drift
- Prediction drift
- Drift analysis
- Retraining
- Model comparison
- Updated model validation
- Deployment consistency