"""Loads the model. Uses a simple placeholder until Job 2 registers the real one in MLflow.

Set MODEL_URI (e.g. "models:/spark_check_score/Production") to use the MLflow model.
"""

import os

AT_RISK_CUTOFF = 10  # PHQ-9 >= 10 means "at risk"


class PlaceholderModel:
    """Stand-in: end-of-term score is mostly the start-of-term PHQ-9 score."""

    version = "placeholder"

    def predict_score(self, row: dict) -> float:
        score = 0.8 * row["phq9_pre"] + 0.1 * row["stress_pre"]
        return max(0.0, min(27.0, score))


class MlflowModel:
    def __init__(self, uri: str):
        import mlflow.pyfunc
        import pandas as pd

        self._pd = pd
        self._model = mlflow.pyfunc.load_model(uri)
        self.version = uri

    def predict_score(self, row: dict) -> float:
        score = float(self._model.predict(self._pd.DataFrame([row]))[0])
        return max(0.0, min(27.0, score))


def load_model():
    uri = os.getenv("MODEL_URI")
    return MlflowModel(uri) if uri else PlaceholderModel()
