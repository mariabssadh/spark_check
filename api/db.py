"""Saves every prediction to PostgreSQL. Does nothing if DATABASE_URL is not set."""

import logging
import os

log = logging.getLogger("spark_check.api")

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now(),
    phq9_pre REAL, loneliness_pre REAL, flourishing_pre REAL,
    stress_pre REAL, neuroticism_pre REAL,
    predicted_phq9_post REAL, at_risk BOOLEAN, model_version TEXT
)
"""


def save_prediction(inputs: dict, score: float, at_risk: bool, version: str) -> None:
    url = os.getenv("DATABASE_URL")
    if not url:
        return
    try:
        import psycopg

        with psycopg.connect(url) as conn:
            conn.execute(CREATE_TABLE)
            conn.execute(
                "INSERT INTO predictions (phq9_pre, loneliness_pre, flourishing_pre,"
                " stress_pre, neuroticism_pre, predicted_phq9_post, at_risk,"
                " model_version) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    inputs["phq9_pre"],
                    inputs["loneliness_pre"],
                    inputs["flourishing_pre"],
                    inputs["stress_pre"],
                    inputs["neuroticism_pre"],
                    score,
                    at_risk,
                    version,
                ),
            )
    except Exception:  # a database problem should never break a prediction
        log.exception("could not save prediction")
