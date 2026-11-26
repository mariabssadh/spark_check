"""Receive Grafana drift alerts and start the existing Prefect deployment."""

import hashlib
import os
import secrets
from datetime import datetime
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

DEPLOYMENT_NAME = "spark-check-training/retrain"
ALERT_NAME = "Significant input drift"

app = FastAPI(title="Spark Check monitoring alerts")
basic_auth = HTTPBasic()


class Alert(BaseModel):
    status: str
    labels: dict[str, str]
    fingerprint: str = ""
    starts_at: datetime | None = Field(default=None, alias="startsAt")


class Notification(BaseModel):
    alerts: list[Alert]


def authenticate(credentials: Annotated[HTTPBasicCredentials, Depends(basic_auth)]):
    password = os.getenv("MONITORING_ALERT_PASSWORD")
    if not password:
        raise HTTPException(503, "Webhook authentication is not configured")
    if credentials.username != "grafana" or not secrets.compare_digest(
        credentials.password.encode(), password.encode()
    ):
        raise HTTPException(
            401, "Invalid credentials", headers={"WWW-Authenticate": "Basic"}
        )


def start_retraining(key: str):
    if not os.getenv("PREFECT_API_URL"):
        raise HTTPException(503, "PREFECT_API_URL is not configured")

    from prefect.deployments import run_deployment

    run_deployment(DEPLOYMENT_NAME, timeout=0, idempotency_key=key, as_subflow=False)


@app.post("/alerts/drift", dependencies=[Depends(authenticate)])
def receive_drift(notification: Notification):
    handled = 0
    for alert in notification.alerts:
        if (
            alert.status != "firing"
            or alert.labels.get("alertname") != ALERT_NAME
            or alert.labels.get("monitoring_action") != "retrain"
        ):
            continue
        if (
            not alert.fingerprint
            or alert.starts_at is None
            or alert.starts_at.utcoffset() is None
        ):
            raise HTTPException(
                422, "Firing drift alerts require a fingerprint and zoned startsAt"
            )

        incident = f"{alert.fingerprint}:{alert.starts_at.timestamp()}"
        key = f"spark-check-drift-{hashlib.sha256(incident.encode()).hexdigest()}"
        start_retraining(key)
        handled += 1
    return {"handled": handled}
