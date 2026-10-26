import time
import statistics

from starlette.testclient import TestClient

from api.main import app


# ---------------------------------------------------------
# Test Client
# ---------------------------------------------------------

def create_client():
    return TestClient(app)


# ---------------------------------------------------------
# Valid Customer
# ---------------------------------------------------------

VALID_CUSTOMER = {
    "customerID": "PERF001",
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


# ---------------------------------------------------------
# 11.1 - Single Request Latency
# ---------------------------------------------------------

def test_single_request_latency():
    with create_client() as client:

        start_time = time.perf_counter()

        response = client.post(
            "/predict",
            json=VALID_CUSTOMER,
        )

        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        assert response.status_code == 200

        print(f"\nSingle request latency: {latency_ms:.2f} ms")


# ---------------------------------------------------------
# 11.2 - Repeated Request Latency
# ---------------------------------------------------------

def test_repeated_request_latency():
    repetitions = 20

    latencies = []

    with create_client() as client:

        for _ in range(repetitions):

            start_time = time.perf_counter()

            response = client.post(
                "/predict",
                json=VALID_CUSTOMER,
            )

            end_time = time.perf_counter()

            assert response.status_code == 200

            latency_ms = (end_time - start_time) * 1000

            latencies.append(latency_ms)

    average_latency = statistics.mean(latencies)
    minimum_latency = min(latencies)
    maximum_latency = max(latencies)
    median_latency = statistics.median(latencies)

    print("\nRepeated request performance:")
    print(f"Requests: {repetitions}")
    print(f"Average latency: {average_latency:.2f} ms")
    print(f"Median latency: {median_latency:.2f} ms")
    print(f"Minimum latency: {minimum_latency:.2f} ms")
    print(f"Maximum latency: {maximum_latency:.2f} ms")

    assert len(latencies) == repetitions


# ---------------------------------------------------------
# 11.3 - Response Consistency
# ---------------------------------------------------------

def test_prediction_consistency():
    repetitions = 10

    predictions = []
    probabilities = []

    with create_client() as client:

        for _ in range(repetitions):

            response = client.post(
                "/predict",
                json=VALID_CUSTOMER,
            )

            assert response.status_code == 200

            data = response.json()

            predictions.append(data["prediction"])
            probabilities.append(data["churn_probability"])

    # Same input should produce the same prediction
    assert len(set(predictions)) == 1

    # Same input should produce the same probability
    assert len(set(probabilities)) == 1

    print("\nPrediction consistency:")
    print(f"Prediction: {predictions[0]}")
    print(f"Probability: {probabilities[0]:.6f}")


# ---------------------------------------------------------
# 11.4 - Reliability / Success Rate
# ---------------------------------------------------------

def test_api_reliability():
    repetitions = 50

    successful_requests = 0

    with create_client() as client:

        for _ in range(repetitions):

            response = client.post(
                "/predict",
                json=VALID_CUSTOMER,
            )

            if response.status_code == 200:
                successful_requests += 1

    success_rate = (
        successful_requests / repetitions
    ) * 100

    print("\nAPI reliability:")
    print(f"Total requests: {repetitions}")
    print(f"Successful requests: {successful_requests}")
    print(f"Success rate: {success_rate:.2f}%")

    assert successful_requests == repetitions


# ---------------------------------------------------------
# 11.5 - Batch Prediction Performance
# ---------------------------------------------------------

def test_batch_prediction_performance():

    customers = []

    for i in range(20):

        customer = VALID_CUSTOMER.copy()

        customer["customerID"] = f"PERF{i:03d}"

        customers.append(customer)

    payload = {
        "customers": customers
    }

    with create_client() as client:

        start_time = time.perf_counter()

        response = client.post(
            "/predict/batch",
            json=payload,
        )

        end_time = time.perf_counter()

    latency_ms = (
        end_time - start_time
    ) * 1000

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 20

    assert len(data["predictions"]) == 20

    print("\nBatch performance:")
    print("Batch size: 20")
    print(f"Batch latency: {latency_ms:.2f} ms")


# ---------------------------------------------------------
# 11.6 - Throughput
# ---------------------------------------------------------

def test_throughput():

    repetitions = 50

    with create_client() as client:

        start_time = time.perf_counter()

        successful_requests = 0

        for _ in range(repetitions):

            response = client.post(
                "/predict",
                json=VALID_CUSTOMER,
            )

            if response.status_code == 200:
                successful_requests += 1

        end_time = time.perf_counter()

    total_time = end_time - start_time

    throughput = (
        successful_requests / total_time
    )

    print("\nThroughput:")
    print(f"Requests: {successful_requests}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Throughput: {throughput:.2f} requests/second")

    assert successful_requests == repetitions
    assert throughput > 0