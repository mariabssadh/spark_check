"""Integration test: start the API container, then send it real HTTP requests."""

import os
import time

import httpx
import pytest

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Skipped in normal test runs; only runs when an API container is up.
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="needs a running API container (set RUN_INTEGRATION=1)",
)

VALID_STUDENT = {
    "phq9_pre": 16,
    "loneliness_pre": 60,
    "flourishing_pre": 25,
    "stress_pre": 30,
    "neuroticism_pre": 32,
}


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=API_URL, timeout=10) as c:
        for _ in range(30):  # wait up to ~60 s for the container to start
            try:
                if c.get("/health").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(2)
        else:
            pytest.fail("API did not become healthy")
        yield c


def test_health(client):
    assert client.get("/health").status_code == 200


def test_valid_student_gets_a_prediction(client):
    response = client.post("/predict", json=VALID_STUDENT)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["predicted_phq9_post"] <= 27
    assert isinstance(body["at_risk"], bool)
    assert body["model_version"]


def test_out_of_range_score_is_rejected(client):
    response = client.post("/predict", json={**VALID_STUDENT, "phq9_pre": 40})
    assert response.status_code == 422