"""Read API predictions and persist aggregate results in PostgreSQL."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

FEATURES = (
    "phq9_pre",
    "loneliness_pre",
    "flourishing_pre",
    "stress_pre",
    "neuroticism_pre",
)

CREATE_OUTCOMES = """
CREATE TABLE IF NOT EXISTS monitoring_outcomes (
    prediction_id integer PRIMARY KEY REFERENCES predictions(id) ON DELETE CASCADE,
    observed_phq9_post double precision NOT NULL
        CHECK (observed_phq9_post BETWEEN 0 AND 27)
)
"""

CREATE_RESULTS = """
CREATE TABLE IF NOT EXISTS monitoring_results (
    model_version text NOT NULL,
    reference_id text NOT NULL,
    window_start timestamptz NOT NULL,
    window_end timestamptz NOT NULL CHECK (window_end > window_start),
    computed_at timestamptz NOT NULL DEFAULT now(),
    prediction_count integer NOT NULL,
    reference_count integer NOT NULL,
    labeled_count integer NOT NULL,
    drift_share double precision,
    drift_detected boolean,
    feature_drift jsonb NOT NULL,
    mae double precision,
    rmse double precision,
    PRIMARY KEY (model_version, reference_id, window_start, window_end)
)
"""

READ_PREDICTIONS = """
SELECT p.id, p.phq9_pre, p.loneliness_pre, p.flourishing_pre,
       p.stress_pre, p.neuroticism_pre, p.predicted_phq9_post,
       o.observed_phq9_post
FROM predictions AS p
LEFT JOIN monitoring_outcomes AS o ON o.prediction_id = p.id
WHERE p.model_version = %s AND p.created_at >= %s AND p.created_at < %s
ORDER BY p.created_at, p.id
"""

UPSERT_RESULT = """
INSERT INTO monitoring_results (
    model_version, reference_id, window_start, window_end, prediction_count,
    reference_count, labeled_count, drift_share, drift_detected, feature_drift, mae, rmse
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (model_version, reference_id, window_start, window_end)
DO UPDATE SET computed_at = now(), prediction_count = EXCLUDED.prediction_count,
    reference_count = EXCLUDED.reference_count, labeled_count = EXCLUDED.labeled_count,
    drift_share = EXCLUDED.drift_share, drift_detected = EXCLUDED.drift_detected,
    feature_drift = EXCLUDED.feature_drift, mae = EXCLUDED.mae, rmse = EXCLUDED.rmse
"""


@dataclass(frozen=True)
class Window:
    model_version: str
    reference_id: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class Result:
    prediction_count: int
    reference_count: int
    labeled_count: int
    drift_share: float | None
    drift_detected: bool | None
    feature_drift: dict[str, dict[str, Any]]
    mae: float | None
    rmse: float | None


def prepare_storage(conn):
    """Initialize monitoring tables after the API has created predictions."""
    conn.execute(CREATE_OUTCOMES)
    conn.execute(CREATE_RESULTS)


def read_predictions(conn, window: Window):
    return conn.execute(
        READ_PREDICTIONS, (window.model_version, window.start, window.end)
    ).fetchall()


def write_result(conn, window: Window, result: Result):
    from psycopg.types.json import Jsonb

    conn.execute(
        UPSERT_RESULT,
        (
            window.model_version,
            window.reference_id,
            window.start,
            window.end,
            result.prediction_count,
            result.reference_count,
            result.labeled_count,
            result.drift_share,
            result.drift_detected,
            Jsonb(result.feature_drift),
            result.mae,
            result.rmse,
        ),
    )
