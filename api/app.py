"""Model API: receives 5 survey scores, returns the predicted score and at-risk flag."""

import time

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from api.db import save_prediction
from api.model import combined_version, load_models
from api.schemas import Prediction, StudentScores

app = FastAPI(title="Spark Check API")
score_model, risk_model = load_models()
MODEL_VERSION = combined_version(score_model, risk_model)

REQUESTS = Counter("api_requests_total", "Requests", ["endpoint", "status"])
LATENCY = Histogram("api_request_seconds", "Request time in seconds", ["endpoint"])


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)

    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        # Routing happens inside call_next. Group unmatched URLs under one label.
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        LATENCY.labels(endpoint).observe(time.perf_counter() - start)
        REQUESTS.labels(endpoint, status).inc()


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "score_model": score_model.version,
        "risk_model": risk_model.version,
    }


@app.post("/predict", response_model=Prediction)
def predict(scores: StudentScores):
    row = scores.model_dump()
    score = round(score_model.predict_score(row), 2)  # score model -> PHQ-9 estimate
    at_risk = bool(risk_model.predict_risk(row))  # risk model -> at-risk flag
    save_prediction(row, score, at_risk, MODEL_VERSION)
    return Prediction(
        predicted_phq9_post=score, at_risk=at_risk, model_version=MODEL_VERSION
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
