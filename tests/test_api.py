from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)

GOOD = {
    "phq9_pre": 8,
    "loneliness_pre": 45,
    "flourishing_pre": 40,
    "stress_pre": 18,
    "neuroticism_pre": 25,
}


def test_predict_returns_score_and_flag():
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["predicted_phq9_post"] <= 27
    assert isinstance(body["at_risk"], bool)


def test_bad_input_gives_clear_error_not_crash():
    r = client.post("/predict", json={**GOOD, "phq9_pre": 40})
    assert r.status_code == 422
    assert "phq9_pre" in r.text


def test_missing_field_is_rejected():
    bad = {k: v for k, v in GOOD.items() if k != "stress_pre"}
    assert client.post("/predict", json=bad).status_code == 422


def test_higher_phq9_gives_higher_prediction():
    low = client.post("/predict", json={**GOOD, "phq9_pre": 2}).json()
    high = client.post("/predict", json={**GOOD, "phq9_pre": 20}).json()
    assert high["predicted_phq9_post"] > low["predicted_phq9_post"]


def test_health_and_metrics():
    assert client.get("/health").json()["status"] == "ok"
    assert "api_requests_total" in client.get("/metrics").text


def test_neuroticism_range_is_8_to_40():
    assert client.post("/predict", json={**GOOD, "neuroticism_pre": 8}).status_code == 200
    assert client.post("/predict", json={**GOOD, "neuroticism_pre": 40}).status_code == 200
    assert client.post("/predict", json={**GOOD, "neuroticism_pre": 7}).status_code == 422
    assert client.post("/predict", json={**GOOD, "neuroticism_pre": 41}).status_code == 422


def test_decimal_scores_are_rejected():
    r = client.post("/predict", json={**GOOD, "phq9_pre": 8.5})
    assert r.status_code == 422
    assert "phq9_pre" in r.text
