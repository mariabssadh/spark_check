from unittest.mock import Mock

from fastapi.testclient import TestClient

from monitoring import alerts

FIRING = {
    "status": "firing",
    "labels": {
        "alertname": "Significant input drift",
        "monitoring_action": "retrain",
    },
    "fingerprint": "23ef",
    "startsAt": "2026-01-01T12:00:00Z",
}


def test_only_firing_drift_alerts_are_sent_with_stable_keys(monkeypatch):
    monkeypatch.setenv("MONITORING_ALERT_PASSWORD", "test-webhook-secret")
    submit = Mock()
    monkeypatch.setattr(alerts, "start_retraining", submit)
    client = TestClient(alerts.app)
    client.auth = ("grafana", "test-webhook-secret")
    payload = {
        "alerts": [
            FIRING,
            {**FIRING, "status": "resolved"},
            {**FIRING, "labels": {"alertname": "DatasourceError"}},
        ]
    }

    for _ in range(2):
        response = client.post("/alerts/drift", json=payload)
        assert response.status_code == 200
        assert response.json() == {"handled": 1}

    assert submit.call_count == 2
    assert submit.call_args_list[0] == submit.call_args_list[1]

    new_episode = {**FIRING, "startsAt": "2026-01-01T14:00:00Z"}
    assert (
        client.post("/alerts/drift", json={"alerts": [new_episode]}).status_code == 200
    )
    assert submit.call_args_list[2] != submit.call_args_list[1]


def test_requires_authentication_and_prefect_url(monkeypatch):
    monkeypatch.setenv("MONITORING_ALERT_PASSWORD", "test-webhook-secret")
    monkeypatch.delenv("PREFECT_API_URL", raising=False)
    client = TestClient(alerts.app)
    payload = {"alerts": [FIRING]}
    assert client.post("/alerts/drift", json=payload).status_code == 401
    assert (
        client.post(
            "/alerts/drift", json=payload, auth=("grafana", "wrong")
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/alerts/drift", json=payload, auth=("grafana", "test-webhook-secret")
        ).status_code
        == 503
    )
