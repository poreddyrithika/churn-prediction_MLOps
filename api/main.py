import time
import uuid

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.model_loader import ModelLoader
from api.preprocessing import clean_prediction_data
from api.schemas import CustomerInput, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
from api.monitoring_logger import log_inference

model_loader = ModelLoader()


# @app.on_event("startup")
# def startup_event():
#     model_loader.load()

@asynccontextmanager
async def lifespan(app: FastAPI):
    #startup
    model_loader.load()

    yield

    #shutdowon
    print("[INFO] Churn Prediction API shutting down...")


app = FastAPI(
    title="Churn Prediction API",
    description="Lab 10 - ML Model Serving and API Engineering",
    version="1.0.0",
    lifespan=lifespan
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = uuid.uuid4().hex[:12]
    errors = []

    for error in exc.errors():
        locations = [str(location) for location in error["loc"] if location != "body"]
        field = ".".join(locations)

        errors.append({
            "field": field,
            "message": error["msg"],
        })

    log_inference(
        input_data={},
        latency_ms=None,
        status="failure",
        error="Request validation failed",
        model_name=model_loader.model_name,
        model_version=model_loader.model_version,
        run_id=model_loader.run_id,
        request_id=request_id,
    )

    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Request validation failed",
            "details": errors,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": "Prediction service unavailable",
            "details": [
                {
                    "field": "prediction",
                    "message": str(exc.detail),
                }
            ],
        },
    )

@app.get("/")
def root():
    return {
        "message": "Churn Prediction API is running",
        "lab": "Lab 10",
        "status": "success",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.get("/metadata")
def metadata():

    return {
        "service": "Churn Prediction API",
        "version": "1.0.0",
        "model_name": model_loader.model_name,
        "model_uri": model_loader.model_uri,
        "model_loaded": model_loader.model is not None,
        "preprocessor_loaded": model_loader.preprocessor is not None,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerInput):
    request_id = uuid.uuid4().hex[:12]
    start_time = time.perf_counter()
    data_dict = data.model_dump()

    if not model_loader.is_ready():
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

        log_inference(
            input_data=data_dict,
            latency_ms=latency_ms,
            status="failure",
            error="Model or preprocessor is not available",
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
            request_id=request_id,
        )

        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor is not available",
        )

    try:
        # 1. Convert JSON data into a DataFrame
        df = pd.DataFrame([data_dict])

        # 2. Clean the raw input
        df = clean_prediction_data(df)

        # 3. Transform using the saved preprocessing pipeline
        X = model_loader.preprocessor.transform(df)

        # 4. Generate prediction
        prediction = model_loader.model.predict(X)[0]

        # 5. Generate churn probability
        probability = model_loader.model.predict_proba(X)[0][1]

        # 6. Calculate inference latency
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

        # 7. Log inference event
        log_inference(
            input_data=data_dict,
            prediction=int(prediction),
            churn_probability=float(probability),
            latency_ms=latency_ms,
            status="success",
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
            request_id=request_id,
        )

        # 8. Return the prediction
        return {
            "prediction": int(prediction),
            "churn_probability": float(probability),
        }

    except Exception as e:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

        log_inference(
            input_data=data_dict,
            latency_ms=latency_ms,
            status="failure",
            error=str(e),
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
            request_id=request_id,
        )

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction",
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(data: BatchPredictionRequest):
    batch_request_id = uuid.uuid4().hex[:12]
    start_time = time.perf_counter()

    data_dicts = [customer.model_dump() for customer in data.customers]

    if not model_loader.is_ready():
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

        for data_dict in data_dicts:
            log_inference(
                input_data=data_dict,
                latency_ms=latency_ms,
                status="failure",
                error="Model or preprocessor is not available",
                model_name=model_loader.model_name,
                model_version=model_loader.model_version,
                run_id=model_loader.run_id,
                request_id=batch_request_id,
            )

        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor is not available",
        )

    try:
        df = pd.DataFrame(data_dicts)
        df = clean_prediction_data(df)

        X = model_loader.preprocessor.transform(df)

        predictions = model_loader.model.predict(X)
        probabilities = model_loader.model.predict_proba(X)[:, 1]

        total_latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            3,
        )

        results = []

        for data_dict, prediction, probability in zip(
            data_dicts,
            predictions,
            probabilities,
        ):
            log_inference(
                input_data=data_dict,
                prediction=int(prediction),
                churn_probability=float(probability),
                latency_ms=total_latency_ms,
                status="success",
                model_name=model_loader.model_name,
                model_version=model_loader.model_version,
                run_id=model_loader.run_id,
                request_id=uuid.uuid4().hex[:12],
            )

            results.append(
                {
                    "prediction": int(prediction),
                    "churn_probability": float(probability),
                }
            )

        return {
            "count": len(results),
            "predictions": results,
        }

    except Exception as e:
        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            3,
        )

        for data_dict in data_dicts:
            log_inference(
                input_data=data_dict,
                latency_ms=latency_ms,
                status="failure",
                error=str(e),
                model_name=model_loader.model_name,
                model_version=model_loader.model_version,
                run_id=model_loader.run_id,
                request_id=uuid.uuid4().hex[:12],
            )

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during batch prediction",
        )