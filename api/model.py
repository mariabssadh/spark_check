"""Loads the two models.

- score model: predicts the end-of-term PHQ-9 score (regression)
- risk model: predicts the at-risk flag (classification)

Set SCORE_MODEL_URI (e.g. "models:/spark-check-score@champion") and
RISK_MODEL_URI (e.g. "models:/spark-check-risk@champion") to use the MLflow models.
MLFLOW_TRACKING_URI must point at the MLflow server.
Anything left unset falls back to a simple placeholder.
"""

import os

AT_RISK_CUTOFF = 10  # PHQ-9 >= 10 means "at risk"


class PlaceholderScoreModel:
    """Stand-in: end-of-term score is mostly the start-of-term PHQ-9 score."""

    version = "placeholder"

    def predict_score(self, row: dict) -> float:
        score = 0.8 * row["phq9_pre"] + 0.1 * row["stress_pre"]
        return max(0.0, min(27.0, score))


class PlaceholderRiskModel:
    """Stand-in: at risk when the placeholder score reaches the cutoff."""

    version = "placeholder"

    def __init__(self):
        self._score = PlaceholderScoreModel()

    def predict_risk(self, row: dict) -> bool:
        return self._score.predict_score(row) >= AT_RISK_CUTOFF


class _MlflowBase:
    def __init__(self, uri: str):
        import mlflow.pyfunc
        import pandas as pd

        self._pd = pd
        self._model = mlflow.pyfunc.load_model(uri)
        self.version = uri

    def _predict(self, row: dict) -> float:
        return float(self._model.predict(self._pd.DataFrame([row]))[0])


class MlflowScoreModel(_MlflowBase):
    def predict_score(self, row: dict) -> float:
        return max(0.0, min(27.0, self._predict(row)))


class MlflowRiskModel(_MlflowBase):
    def predict_risk(self, row: dict) -> bool:
        # Classifier output: 0/1 label, or a probability. Both work with >= 0.5.
        return self._predict(row) >= 0.5


def load_models():
    """Return (score_model, risk_model)."""
    score_uri = os.getenv("SCORE_MODEL_URI") or os.getenv("MODEL_URI")  # old name
    risk_uri = os.getenv("RISK_MODEL_URI")
    score = MlflowScoreModel(score_uri) if score_uri else PlaceholderScoreModel()
    risk = MlflowRiskModel(risk_uri) if risk_uri else PlaceholderRiskModel()
    return score, risk


def combined_version(score_model, risk_model) -> str:
    return f"score={score_model.version}; risk={risk_model.version}"
