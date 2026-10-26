# Lab 11 - Containerization and Deployment of ML Systems

## Purpose

This file is the fixed roadmap and progress tracker for Lab 11.

The objective is to take the completed Lab 10 FastAPI ML service, containerize it with Docker, validate portability and reproducibility, simulate and recover from deployment failures, deploy it with Kubernetes, expose and scale the service, and document the results.

**Lab 13 is not treated as a separate implementation phase.** Its relevant end-to-end integration requirements will be incorporated progressively during Labs 11 and 12 so that the complete MLOps project is finished by the end of Lab 12.

---

# 1. Starting Point

Lab 10 is complete.

Baseline:

- FastAPI application
- `/` endpoint
- `/health`
- `/metadata`
- `/predict`
- `/predict/batch`
- Pydantic schemas
- Validation and standardized errors
- Robustness testing
- Performance and reliability testing
- API testing
- 29/29 automated tests passed

Final Lab 10 integration test:

```powershell
pytest tests/ -v
```

Result:

```text
29 passed in 12.72s
```

Lab 10 performance evidence:

```text
Average latency: 19.38 ms
Median latency: 18.79 ms
Maximum latency: 23.89 ms
Reliability: 100%
Batch latency: 21.34 ms
Throughput: 51.68 requests/second
```

These are local `TestClient` measurements, not production network benchmarks.

---

# 2. Lab 11 Objectives

Lab 11 covers:

1. Docker containerization
2. Dependency isolation
3. Portability
4. Reproducibility
5. Containerized API validation
6. Prediction consistency
7. Deployment failure simulation
8. Failure debugging
9. Deployment recovery
10. Kubernetes deployment
11. Kubernetes service exposure
12. Kubernetes scaling
13. Service interruption
14. Recovery validation
15. Deployment documentation

---

# 3. Target Architecture

```text
Existing ML Pipeline
        |
        v
Production Model
        |
        v
FastAPI
        |
   +----+----+
   |         |
/predict  /predict/batch
   |         |
   +----+----+
        |
        v
   Docker Image
        |
        v
 Docker Container
        |
        v
   Kubernetes
        |
   +----+----+
   |         |
 Pod 1     Pod 2
   |         |
   +----+----+
        |
        v
Kubernetes Service
        |
        v
      Client
```

The containerized and Kubernetes-deployed service should preserve the inference behavior of the Lab 10 system.

---

# 4. Fixed Stage Order

Follow these stages in exactly this order:

```text
11.1  Containerization Architecture
11.2  Prepare Project for Docker
11.3  Create Dockerfile
11.4  Create .dockerignore
11.5  Build Docker Image
11.6  Run FastAPI Inside Docker
11.7  Containerized API Validation
11.8  Local vs Container Prediction Consistency
11.9  Dependency Isolation and Reproducibility
11.10 Deployment Failure Simulation
11.11 Failure Debugging
11.12 Deployment Recovery
11.13 Kubernetes Deployment
11.14 Kubernetes Service and Exposure
11.15 Kubernetes Scaling
11.16 Kubernetes Service Interruption
11.17 Kubernetes Recovery Validation
11.18 Containerization Report
11.19 Deployment and Debugging Report
11.20 Final Lab 11 Validation
```

---

# 5. Stage 11.1 - Containerization Architecture

## Objective

Understand how the Lab 10 FastAPI ML system will run inside Docker.

## Tasks

- Identify runtime files.
- Identify the production model artifact.
- Identify the preprocessor artifact.
- Identify API source files.
- Identify configuration files.
- Identify Python dependencies.
- Separate training-only files from inference-time files.
- Understand host environment versus container environment.

## Runtime components

At a high level:

```text
api/
models/
config.yaml
requirements.txt
Dockerfile
.dockerignore
```

## Evidence

- [ ] Runtime file list
- [ ] Final architecture
- [ ] Explanation of what the container needs

## Status

- [ ] Completed
- [ ] Verified

---

# 6. Stage 11.2 - Prepare Project for Docker

## Objective

Make the current project Docker-ready without breaking Lab 10.

## Tasks

- Verify FastAPI imports.
- Verify model paths.
- Verify preprocessor paths.
- Verify configuration paths.
- Check Linux-compatible path handling.
- Verify runtime dependencies.
- Confirm all required runtime files exist.

## Important

Do not unnecessarily include training-only files in the runtime image.

Keep the distinction clear:

```text
Training environment
        !=
Inference/deployment environment
```

## Evidence

- [ ] Runtime file list
- [ ] Project structure
- [ ] Path/configuration verification

## Status

- [ ] Completed
- [ ] Verified

---

# 7. Stage 11.3 - Create Dockerfile

## Objective

Create a reproducible Docker image definition for the FastAPI ML service.

## Dockerfile responsibilities

- Base Python image
- Working directory
- Dependency installation
- Application files
- Runtime configuration
- Exposed port
- API startup command

## Conceptual flow

```text
Python base image
      |
      v
Working directory
      |
      v
Install dependencies
      |
      v
Copy application
      |
      v
Expose API port
      |
      v
Start FastAPI
```

## Evidence

- [ ] Final Dockerfile
- [ ] Dockerfile build succeeds

## Status

- [ ] Completed
- [ ] Verified

---

# 8. Stage 11.4 - Create .dockerignore

## Objective

Prevent unnecessary development files from entering the Docker image.

Typical exclusions:

```text
venv/
.git/
.pytest_cache/
__pycache__/
*.pyc
.ipynb_checkpoints/
```

Add other unnecessary files after checking the project.

## Goal

Keep the image:

- Smaller
- Cleaner
- Faster to build
- Focused on deployment

## Evidence

- [ ] Final `.dockerignore`

## Status

- [ ] Completed
- [ ] Verified

---

# 9. Stage 11.5 - Build Docker Image

## Objective

Build a runnable image for the ML API.

## Workflow

```text
Dockerfile
   |
   v
docker build
   |
   v
Docker image
```

## Example command

```powershell
docker build -t churn-prediction-api .
```

## Verify

- Build succeeds.
- Dependencies install.
- Application files are present.
- Model/preprocessor artifacts are available.
- No required runtime file is missing.
- Image exists with the expected tag.

## Evidence

- [ ] Build command
- [ ] Build output
- [ ] Image name/tag
- [ ] Important warnings/errors and resolutions

## Status

- [ ] Completed
- [ ] Verified

---

# 10. Stage 11.6 - Run FastAPI Inside Docker

## Objective

Run the FastAPI service completely inside the Docker container.

## Concept

```text
Host
  |
  v
Docker container
  |
  +--> FastAPI
  +--> Model
  +--> Preprocessor
```

## Example command pattern

```powershell
docker run -p 8000:8000 churn-prediction-api
```

## Verify

- Container starts.
- API process starts.
- Model loads.
- Preprocessor loads.
- Port mapping works.
- No startup exception occurs.

## Evidence

- [ ] `docker run` command
- [ ] Startup logs
- [ ] Container status

## Status

- [ ] Completed
- [ ] Verified

---

# 11. Stage 11.7 - Containerized API Validation

## Objective

Verify that the Dockerized API behaves correctly.

## Endpoints

```text
GET  /
GET  /health
GET  /metadata
POST /predict
POST /predict/batch
```

## Check

- HTTP status
- Model loading
- Preprocessor loading
- Metadata
- Single prediction
- Batch prediction
- Validation errors
- Standardized errors

## Evidence

- [ ] Endpoint test results
- [ ] Prediction output
- [ ] Validation/error output
- [ ] Logs/screenshots where required

## Status

- [ ] Completed
- [ ] Verified

---

# 12. Stage 11.8 - Local vs Container Prediction Consistency

## Objective

Prove that containerization has not changed inference behavior.

## Test design

Use exactly the same customer input:

```text
Same input
   |
   +----> Local FastAPI
   |
   +----> Docker FastAPI
```

Compare:

- Prediction
- Churn probability
- Response structure

## Evidence

Record:

```text
Input: TEST001

Local:
Prediction = ...
Probability = ...

Docker:
Prediction = ...
Probability = ...
```

## Status

- [ ] Completed
- [ ] Verified

---

# 13. Stage 11.9 - Dependency Isolation and Reproducibility

## Objective

Demonstrate that the container uses its own controlled environment and does not rely on the host virtual environment.

## Concept

```text
Host Python environment
        !=
Container Python environment
```

## Verify

- Python version inside container
- Dependency availability
- Required packages
- API startup
- Model inference
- Fresh-container behavior

## Reproducibility check

A fresh container created from the same image should produce consistent behavior.

## Evidence

- [ ] Dependency/version information
- [ ] Container environment information
- [ ] Fresh-container test
- [ ] Prediction consistency

## Status

- [ ] Completed
- [ ] Verified

---

# 14. Stage 11.10 - Deployment Failure Simulation

## Objective

Deliberately introduce controlled deployment failures.

## Failure scenarios

### A. Dependency mismatch

Simulate a missing or incompatible runtime dependency.

Expected type of result:

```text
Container startup/runtime failure
```

### B. Missing model artifact

Temporarily make the model or preprocessor unavailable.

Expected type of result:

```text
ModelLoader failure
API/container startup failure
```

### C. Incorrect configuration

Use an invalid configuration/path.

Expected type of result:

```text
Configuration/runtime failure
```

### D. Incorrect runtime/service configuration

Use an incorrect startup or runtime configuration.

Expected type of result:

```text
Service failure or unavailability
```

## Important

All failure simulations must be reversible. Do not permanently damage the working project.

## Evidence

For each failure:

```text
Failure setup
Command
Observed error
Logs
Root cause
```

## Status

- [ ] Completed
- [ ] Verified

---

# 15. Stage 11.11 - Failure Debugging

## Objective

Identify the cause of deployment failures and demonstrate how they are diagnosed.

## Debugging workflow

```text
Failure
  |
  v
Check container/service status
  |
  v
Inspect logs
  |
  v
Identify root cause
  |
  v
Fix
```

## Inspect where appropriate

- Docker logs
- Container status
- Image contents
- Paths
- Environment variables
- Dependencies
- Configuration
- Model/preprocessor artifacts

## Evidence

For each failure:

```text
Problem
Cause
Evidence
Fix
Verification
```

## Status

- [ ] Completed
- [ ] Verified

---

# 16. Stage 11.12 - Deployment Recovery

## Objective

Restore the system after simulated failures.

## Recovery workflow

```text
Broken deployment
       |
       v
Identify cause
       |
       v
Repair
       |
       v
Rebuild/restart
       |
       v
Health check
       |
       v
Prediction test
```

## Verify

- Container starts.
- `/health` works.
- Model loads.
- Prediction works.
- Batch prediction works.

## Evidence

- [ ] Recovery command
- [ ] Startup logs
- [ ] Health response
- [ ] Prediction response

## Status

- [ ] Completed
- [ ] Verified

---

# 17. Stage 11.13 - Kubernetes Deployment

## Objective

Deploy the Dockerized ML service using Kubernetes.

## Concept

```text
Docker image
      |
      v
Kubernetes Deployment
      |
      v
Pod
      |
      v
FastAPI container
```

## Tasks

- Create Kubernetes Deployment manifest.
- Configure image.
- Configure container port.
- Apply deployment.
- Verify pod creation.
- Verify pod status.
- Inspect logs.

## Evidence

- [ ] Deployment YAML
- [ ] Apply command
- [ ] Pod status
- [ ] Deployment status
- [ ] Logs

## Status

- [ ] Completed
- [ ] Verified

---

# 18. Stage 11.14 - Kubernetes Service and Exposure

## Objective

Expose the FastAPI service through Kubernetes.

## Concept

```text
Client
  |
  v
Kubernetes Service
  |
  v
Pod
  |
  v
FastAPI
```

## Tasks

- Create Kubernetes Service.
- Connect it to the deployment.
- Expose the API.
- Verify service endpoint.
- Test `/health`.
- Test `/predict`.
- Test `/predict/batch`.

## Evidence

- [ ] Service YAML
- [ ] Service status
- [ ] Service endpoint
- [ ] API response through Kubernetes

## Status

- [ ] Completed
- [ ] Verified

---

# 19. Stage 11.15 - Kubernetes Scaling

## Objective

Demonstrate scaling of the ML service.

## Example

```text
1 replica
   |
   v
2 replicas
   |
   v
3 replicas
```

## Tasks

- Observe initial replica count.
- Scale deployment.
- Verify multiple pods.
- Verify service remains functional.
- Observe deployment state.

## Evidence

- [ ] Scaling command
- [ ] Replica counts
- [ ] Pod status
- [ ] Successful API access after scaling

## Status

- [ ] Completed
- [ ] Verified

---

# 20. Stage 11.16 - Kubernetes Service Interruption

## Objective

Simulate a service interruption.

## Example

```text
Running pod
    |
    v
Pod stopped/deleted
    |
    v
Observe system behavior
```

## Observe

- Pod state
- Deployment state
- Service availability
- Whether Kubernetes recreates the pod
- API availability during/after interruption

## Evidence

- [ ] Interruption action
- [ ] Status before
- [ ] Status after
- [ ] Logs
- [ ] Observed behavior

## Status

- [ ] Completed
- [ ] Verified

---

# 21. Stage 11.17 - Kubernetes Recovery Validation

## Objective

Verify that the service returns to a working state after interruption.

## Recovery validation

```text
Deployment
   |
   v
Pods Running
   |
   v
Service Available
   |
   v
/health
   |
   v
/predict
   |
   v
/predict/batch
```

## Evidence

- [ ] Recovery status
- [ ] Pod status
- [ ] Service status
- [ ] Health response
- [ ] Prediction response

## Status

- [ ] Completed
- [ ] Verified

---

# 22. Stage 11.18 - Containerization Report

## Objective

Document the Docker portion of Lab 11.

## Include

- Objective
- Containerization architecture
- Runtime project structure
- Dockerfile
- `.dockerignore`
- Image build
- Image information
- Container startup
- API validation
- Local vs container comparison
- Dependency isolation
- Reproducibility
- Problems and resolutions

## Suggested output

```text
reports/containerization_report.md
```

## Status

- [ ] Completed
- [ ] Verified

---

# 23. Stage 11.19 - Deployment and Debugging Report

## Objective

Document deployment, failures, debugging, recovery, and Kubernetes work.

## Include

### Docker deployment

- Container startup
- API validation

### Failure testing

- Failure scenario
- Error
- Root cause
- Fix
- Recovery

### Kubernetes

- Deployment
- Pods
- Service
- Exposure
- Scaling
- Interruption
- Recovery

### Evidence

Include important commands, logs, screenshots, and outputs.

## Suggested outputs

```text
reports/deployment_report.md
reports/debugging_report.md
```

## Status

- [ ] Completed
- [ ] Verified

---

# 24. Stage 11.20 - Final Lab 11 Validation

## Docker

- [ ] Dockerfile works
- [ ] Image builds
- [ ] Container starts
- [ ] API works
- [ ] Model loads
- [ ] Preprocessor loads

## API

- [ ] `/` works
- [ ] `/health` works
- [ ] `/metadata` works
- [ ] `/predict` works
- [ ] `/predict/batch` works
- [ ] Validation works
- [ ] Error handling works

## Portability and reproducibility

- [ ] Local prediction verified
- [ ] Container prediction verified
- [ ] Predictions consistent
- [ ] Container independent of host venv
- [ ] Dependencies isolated
- [ ] Fresh container reproducibility verified

## Failure handling

- [ ] Dependency failure simulated
- [ ] Missing artifact failure simulated
- [ ] Configuration failure simulated
- [ ] Runtime/service failure simulated
- [ ] Failures debugged
- [ ] Deployment recovered

## Kubernetes

- [ ] Deployment created
- [ ] Pods running
- [ ] Service created
- [ ] Service exposed
- [ ] API tested through Kubernetes
- [ ] Scaling demonstrated
- [ ] Service interruption simulated
- [ ] Recovery validated

## Documentation

- [ ] Containerization report
- [ ] Deployment report
- [ ] Debugging report
- [ ] Final validation evidence

## Final status

- [ ] Lab 11 fully completed

---

# 25. Quick Progress Tracker

Use this section for day-to-day tracking.

```text
[ ] 11.1 Containerization Architecture
[ ] 11.2 Prepare Project for Docker
[ ] 11.3 Create Dockerfile
[ ] 11.4 Create .dockerignore
[ ] 11.5 Build Docker Image
[ ] 11.6 Run FastAPI Inside Docker
[ ] 11.7 Containerized API Validation
[ ] 11.8 Local vs Container Prediction Consistency
[ ] 11.9 Dependency Isolation and Reproducibility
[ ] 11.10 Deployment Failure Simulation
[ ] 11.11 Failure Debugging
[ ] 11.12 Deployment Recovery
[ ] 11.13 Kubernetes Deployment
[ ] 11.14 Kubernetes Service and Exposure
[ ] 11.15 Kubernetes Scaling
[ ] 11.16 Kubernetes Service Interruption
[ ] 11.17 Kubernetes Recovery Validation
[ ] 11.18 Containerization Report
[ ] 11.19 Deployment and Debugging Report
[ ] 11.20 Final Lab 11 Validation
```

---

# 26. Recommended Working Method

For every stage, follow:

```text
UNDERSTAND
    ↓
IMPLEMENT
    ↓
RUN
    ↓
VERIFY
    ↓
DOCUMENT
    ↓
MARK COMPLETE
```

A stage should only be marked complete after its behavior has been tested and evidence has been recorded.

---

# 27. Lab 11 Completion Definition

Lab 11 is complete only when:

```text
Docker containerization        ✅
Dependency isolation          ✅
Portability                   ✅
Reproducibility               ✅
Prediction consistency        ✅
Failure simulation            ✅
Failure debugging             ✅
Deployment recovery           ✅
Kubernetes deployment         ✅
Service exposure              ✅
Scaling                       ✅
Service interruption          ✅
Kubernetes recovery           ✅
Documentation                ✅
```

Final target:

```text
Lab 10 - FastAPI
       |
       v
Docker
       |
       v
Container Validation
       |
       v
Failure Simulation
       |
       v
Debugging
       |
       v
Recovery
       |
       v
Kubernetes
       |
       v
Exposure
       |
       v
Scaling
       |
       v
Interruption
       |
       v
Recovery
       |
       v
LAB 11 COMPLETE
       |
       v
Lab 12 - Monitoring + Drift + Retraining
       |
       v
COMPLETE MLOPS PROJECT
```

---

# 28. Relationship to Lab 12

Lab 11 ends with a working deployed ML service.

Lab 12 will then add:

- Monitoring
- Logging
- Prediction tracking
- Latency tracking
- Throughput tracking
- Failure tracking
- Drift detection
- Performance degradation analysis
- Retraining
- Model comparison
- Updated model validation
- Redeployment

The complete MLOps project should be finished by the end of Lab 12.

---

# 29. Final End-to-End Project Goal

By the end of Lab 12, the complete lifecycle should be:

```text
Data
  ↓
Preprocessing
  ↓
Training
  ↓
Experiment Tracking
  ↓
Validation
  ↓
Quality Gate
  ↓
Model Registry
  ↓
Inference
  ↓
FastAPI
  ↓
Docker
  ↓
Kubernetes
  ↓
Monitoring
  ↓
Drift Detection
  ↓
Performance Monitoring
  ↓
Retraining
  ↓
Model Validation
  ↓
Redeployment
```

This document is the fixed Lab 11 roadmap and can be used as the reference checklist throughout implementation.
