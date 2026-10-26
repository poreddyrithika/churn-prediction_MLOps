import pytest

from starlette.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


VALID_CUSTOMER = {
    "customerID": "TEST001",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 10,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 50.0,
    "TotalCharges": "500.0",
}


SECOND_CUSTOMER = {
    "customerID": "TEST002",
    "gender": "Female",
    "SeniorCitizen": 1,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 35,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "Yes",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 80.0,
    "TotalCharges": "2800.0",
}


# ---------------------------------------------------------
# 9.1 - Health Endpoint
# ---------------------------------------------------------

def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


# ---------------------------------------------------------
# 9.2 - Metadata Endpoint
# ---------------------------------------------------------

def test_metadata(client):
    response = client.get("/metadata")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "Churn Prediction API"
    assert data["version"] == "1.0.0"

    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True

    assert data["model_name"] is not None
    assert data["model_uri"] is not None


# ---------------------------------------------------------
# 9.3 - Valid Single Prediction
# ---------------------------------------------------------

def test_valid_prediction(client):
    response = client.post(
        "/predict",
        json=VALID_CUSTOMER,
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "churn_probability" in data

    assert data["prediction"] in [0, 1]

    assert 0.0 <= data["churn_probability"] <= 1.0


# ---------------------------------------------------------
# 9.4 - Invalid Single Prediction: Gender
# ---------------------------------------------------------

def test_invalid_prediction_gender(client):
    payload = VALID_CUSTOMER.copy()

    payload["gender"] = "ABC"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "gender"


# ---------------------------------------------------------
# 9.4 - Invalid Single Prediction: Tenure
# ---------------------------------------------------------

def test_invalid_prediction_tenure(client):
    payload = VALID_CUSTOMER.copy()

    payload["tenure"] = -5

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "tenure"


# ---------------------------------------------------------
# 9.4 - Missing Required Field
# ---------------------------------------------------------

def test_missing_prediction_field(client):
    payload = VALID_CUSTOMER.copy()

    del payload["MonthlyCharges"]

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "MonthlyCharges"


# ---------------------------------------------------------
# 9.5 - Valid Batch Prediction
# ---------------------------------------------------------

def test_valid_batch_prediction(client):
    payload = {
        "customers": [
            VALID_CUSTOMER,
            SECOND_CUSTOMER,
        ]
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 2

    assert len(data["predictions"]) == 2

    for prediction in data["predictions"]:
        assert "prediction" in prediction
        assert "churn_probability" in prediction

        assert prediction["prediction"] in [0, 1]

        assert 0.0 <= prediction["churn_probability"] <= 1.0


# ---------------------------------------------------------
# 9.6 - Invalid Batch Prediction
# ---------------------------------------------------------

def test_invalid_batch_prediction(client):
    invalid_customer = SECOND_CUSTOMER.copy()

    invalid_customer["gender"] = "ABC"

    payload = {
        "customers": [
            VALID_CUSTOMER,
            invalid_customer,
        ]
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "customers.1.gender"


# ---------------------------------------------------------
# 9.7 - Multiple Batch Validation Errors
# ---------------------------------------------------------

def test_multiple_batch_validation_errors(client):
    first_customer = VALID_CUSTOMER.copy()
    second_customer = SECOND_CUSTOMER.copy()

    first_customer["gender"] = "ABC"
    second_customer["tenure"] = -5

    payload = {
        "customers": [
            first_customer,
            second_customer,
        ]
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 2

    fields = [
        error["field"]
        for error in data["details"]
    ]

    assert "customers.0.gender" in fields
    assert "customers.1.tenure" in fields


# ---------------------------------------------------------
# 9.8 - Empty Batch
# ---------------------------------------------------------

def test_empty_batch(client):
    payload = {
        "customers": []
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "customers"


# ---------------------------------------------------------
# 9.9 - Invalid Datatype
# ---------------------------------------------------------

def test_invalid_datatype(client):
    payload = VALID_CUSTOMER.copy()

    payload["tenure"] = "ten"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "tenure"


# ---------------------------------------------------------
# 9.10 - Extreme Valid Numeric Values
# ---------------------------------------------------------

def test_extreme_numeric_values(client):
    payload = VALID_CUSTOMER.copy()

    payload["tenure"] = 100
    payload["MonthlyCharges"] = 10000.0
    payload["TotalCharges"] = "1000000.0"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]

    assert 0.0 <= data["churn_probability"] <= 1.0

# ---------------------------------------------------------
# Stage 10.1 - Null / Corrupted Input
# ---------------------------------------------------------

def test_null_value(client):
    payload = VALID_CUSTOMER.copy()

    payload["gender"] = None

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "gender"

# =========================================================
# STAGE 10 - INVALID / EXTREME / CORRUPTED INPUT TESTING
# =========================================================


# ---------------------------------------------------------
# 10.2 - Malformed JSON
# ---------------------------------------------------------

def test_malformed_json(client):
    malformed_json = """
    {
        "customerID": "TEST001",
        "gender": "Male"
    """

    response = client.post(
        "/predict",
        content=malformed_json,
        headers={
            "Content-Type": "application/json"
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) >= 1


# ---------------------------------------------------------
# 10.3 - Completely Wrong JSON Structure
# ---------------------------------------------------------

def test_wrong_json_structure(client):
    payload = [
        "this",
        "should",
        "be",
        "an",
        "object"
    ]

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) >= 1


# ---------------------------------------------------------
# 10.4 - Null Numeric Value
# ---------------------------------------------------------

def test_null_numeric_value(client):
    payload = VALID_CUSTOMER.copy()

    payload["MonthlyCharges"] = None

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "MonthlyCharges"


# ---------------------------------------------------------
# 10.5 - Null Required String Value
# ---------------------------------------------------------

def test_null_total_charges(client):
    payload = VALID_CUSTOMER.copy()

    payload["TotalCharges"] = None

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "TotalCharges"


# ---------------------------------------------------------
# 10.6 - Wrong Datatype for Multiple Fields
# ---------------------------------------------------------

def test_multiple_invalid_datatypes(client):
    payload = VALID_CUSTOMER.copy()

    payload["tenure"] = "ten years"
    payload["MonthlyCharges"] = "fifty"
    payload["SeniorCitizen"] = "old"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) >= 2

    fields = [
        error["field"]
        for error in data["details"]
    ]

    assert "tenure" in fields
    assert "MonthlyCharges" in fields


# ---------------------------------------------------------
# 10.7 - Invalid Batch Structure
# ---------------------------------------------------------

def test_invalid_batch_structure(client):
    payload = {
        "customers": VALID_CUSTOMER
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) >= 1


# ---------------------------------------------------------
# 10.8 - Null Batch Customers
# ---------------------------------------------------------

def test_null_batch_customers(client):
    payload = {
        "customers": None
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) >= 1


# ---------------------------------------------------------
# 10.9 - Batch Containing Null Customer
# ---------------------------------------------------------

def test_batch_with_null_customer(client):
    payload = {
        "customers": [
            VALID_CUSTOMER,
            None,
        ]
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) >= 1


# ---------------------------------------------------------
# 10.10 - Extreme Negative Numeric Values
# ---------------------------------------------------------

def test_extreme_negative_values(client):
    payload = VALID_CUSTOMER.copy()

    payload["tenure"] = -1000000
    payload["MonthlyCharges"] = -999999.0

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    fields = [
        error["field"]
        for error in data["details"]
    ]

    assert "tenure" in fields
    assert "MonthlyCharges" in fields


# ---------------------------------------------------------
# 10.11 - Very Large Valid Batch
# ---------------------------------------------------------

def test_large_batch(client):
    customers = []

    for i in range(20):
        customer = VALID_CUSTOMER.copy()

        customer["customerID"] = f"TEST{i:03d}"

        customers.append(customer)

    payload = {
        "customers": customers
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 20

    assert len(data["predictions"]) == 20

    for prediction in data["predictions"]:
        assert prediction["prediction"] in [0, 1]

        assert (
            0.0
            <= prediction["churn_probability"]
            <= 1.0
        )