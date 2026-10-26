import json
from urllib.request import Request, urlopen

from fastapi.testclient import TestClient

from api.main import app


PAYLOAD = {
    "customerID": "SOMEONE18",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 50.0,
    "TotalCharges": "600.0",
}


def call_container():
    data = json.dumps(PAYLOAD).encode("utf-8")

    request = Request(
        "http://localhost:8000/predict",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    print("=" * 60)
    print("LOCAL VS CONTAINER PREDICTION CONSISTENCY")
    print("=" * 60)

    with TestClient(app) as client:
        local_response = client.post("/predict", json=PAYLOAD)

    container_response = call_container()

    print("\nLOCAL RESPONSE:")
    print(json.dumps(local_response.json(), indent=2))

    print("\nCONTAINER RESPONSE:")
    print(json.dumps(container_response, indent=2))

    local_data = local_response.json()

    local_prediction = local_data["prediction"]
    container_prediction = container_response["prediction"]

    local_probability = local_data.get("probability")
    container_probability = container_response.get("probability")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    print(f"Local prediction      : {local_prediction}")
    print(f"Container prediction : {container_prediction}")

    if local_probability is not None:
        print(f"Local probability     : {local_probability}")

    if container_probability is not None:
        print(f"Container probability: {container_probability}")

    prediction_match = local_prediction == container_prediction

    probability_match = (
        local_probability is None
        or container_probability is None
        or abs(local_probability - container_probability) < 1e-10
    )

    print(f"\nPrediction match      : {prediction_match}")
    print(f"Probability match     : {probability_match}")

    if prediction_match and probability_match:
        print("\n[SUCCESS] Local and container predictions are consistent.")
    else:
        print("\n[FAILURE] Local and container predictions differ.")
        raise AssertionError("Prediction consistency check failed.")


if __name__ == "__main__":
    main()