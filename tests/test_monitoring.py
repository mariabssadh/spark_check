from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

import api.app as api


def scrape(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    return {
        (
            sample.name,
            sample.labels["endpoint"],
            sample.labels.get("status"),
        ): sample.value
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
        if sample.name in {"api_requests_total", "api_request_seconds_count"}
    }


def test_request_metrics(monkeypatch):
    def fail(row):
        raise RuntimeError("test failure")

    monkeypatch.setattr(api.score_model, "predict_score", fail)
    client = TestClient(api.app, raise_server_exceptions=False)
    before = scrape(client)
    assert client.get("/health").status_code == 200
    assert client.post("/predict", json={}).status_code == 422
    response = client.post(
        "/predict",
        json={
            "phq9_pre": 8,
            "loneliness_pre": 45,
            "flourishing_pre": 40,
            "stress_pre": 18,
            "neuroticism_pre": 25,
        },
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert client.get("/unknown-one").status_code == 404
    assert client.get("/unknown-two").status_code == 404
    after = scrape(client)

    for endpoint, status, count in [
        ("/health", "200", 1),
        ("/predict", "422", 1),
        ("/predict", "500", 1),
        ("unmatched", "404", 2),
    ]:
        key = ("api_requests_total", endpoint, status)
        assert after[key] - before.get(key, 0) == count

    for endpoint, count in [("/health", 1), ("/predict", 2), ("unmatched", 2)]:
        key = ("api_request_seconds_count", endpoint, None)
        assert after[key] - before.get(key, 0) == count

    assert scrape(client) == after
    assert {endpoint for _, endpoint, _ in after}.isdisjoint(
        {"/metrics", "/unknown-one", "/unknown-two"}
    )
