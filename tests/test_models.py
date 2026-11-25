from api import model as m
from api.app import app  # noqa: F401  (import check)

ROW = {
    "phq9_pre": 8,
    "loneliness_pre": 45,
    "flourishing_pre": 40,
    "stress_pre": 18,
    "neuroticism_pre": 25,
}


def test_placeholders_when_no_uris(monkeypatch):
    for v in ("SCORE_MODEL_URI", "RISK_MODEL_URI", "MODEL_URI"):
        monkeypatch.delenv(v, raising=False)
    score, risk = m.load_models()
    assert isinstance(score, m.PlaceholderScoreModel)
    assert isinstance(risk, m.PlaceholderRiskModel)
    assert m.combined_version(score, risk) == "score=placeholder; risk=placeholder"


def test_both_uris_load_both_models(monkeypatch):
    loaded = []
    monkeypatch.setattr(
        m._MlflowBase, "__init__", lambda self, uri: loaded.append(uri) or None
    )
    monkeypatch.setenv("SCORE_MODEL_URI", "models:/spark-check-score@champion")
    monkeypatch.setenv("RISK_MODEL_URI", "models:/spark-check-risk@champion")
    score, risk = m.load_models()
    assert loaded == [
        "models:/spark-check-score@champion",
        "models:/spark-check-risk@champion",
    ]
    assert isinstance(score, m.MlflowScoreModel)
    assert isinstance(risk, m.MlflowRiskModel)


def test_risk_comes_from_risk_model_not_score(monkeypatch):
    monkeypatch.setattr(m._MlflowBase, "__init__", lambda self, uri: None)
    risk = m.MlflowRiskModel.__new__(m.MlflowRiskModel)
    risk._predict = lambda row: 1.0
    assert risk.predict_risk(ROW) is True
    risk._predict = lambda row: 0.0
    assert risk.predict_risk(ROW) is False


def test_mlflow_models_get_float_columns():
    """The registered models expect doubles; int64 input is refused by MLflow."""
    import pandas as pd

    seen = {}

    class FakeModel:
        def predict(self, frame):
            seen["dtypes"] = set(frame.dtypes.astype(str))
            return [5.0]

    score = m.MlflowScoreModel.__new__(m.MlflowScoreModel)
    score._pd, score._model = pd, FakeModel()
    assert score.predict_score(ROW) == 5.0
    assert seen["dtypes"] == {"float64"}
