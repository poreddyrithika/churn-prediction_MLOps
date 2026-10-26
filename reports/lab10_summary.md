# Lab 10 - ML Model Serving and API Engineering using FastAPI

## 1. Objective

The objective of Lab 10 is to deploy the trained machine learning model as a FastAPI inference service and evaluate its API functionality, validation behavior, robustness, performance, and reliability.

---

## 2. Implementation Completed

The following components were implemented:

- FastAPI application
- Health endpoint
- ModelLoader
- MLflow production model loading
- Preprocessor loading
- Metadata endpoint
- Single prediction endpoint
- Pydantic input/output schemas
- Request validation
- Standardized error responses
- Batch prediction endpoint
- API testing
- Invalid and corrupted input testing
- Performance testing
- Reliability testing
- Throughput measurement

---

## 3. API Endpoints

### GET `/`

Returns basic API service information.

### GET `/health`

Checks API health and verifies whether the model and preprocessor are loaded.

### GET `/metadata`

Returns model and service metadata.

### POST `/predict`

Accepts one customer record and returns:

- Prediction
- Churn probability

### POST `/predict/batch`

Accepts multiple customer records and returns predictions for all customers.

---

## 4. Input Validation

Pydantic schemas are used to validate API requests.

Validation includes:

- Required fields
- Categorical values
- Numeric datatypes
- Non-negative tenure
- Non-negative monthly charges
- Batch size validation
- Null-value rejection
- Invalid datatype detection

---

## 5. Standardized Error Handling

Validation failures return HTTP 422 responses.

The standardized response contains:

- `status`
- `message`
- `details`

Example:

```json
{
    "status": "error",
    "message": "Request validation failed",
    "details": [
        {
            "field": "gender",
            "message": "Input should be 'Male' or 'Female'"
        }
    ]
}