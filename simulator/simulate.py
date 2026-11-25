"""Fake traffic: sends made-up students to the Spark Check API, with an exam-season drift switch.

Normal mode: each fake student is a copy of a real training student with small random changes.
Drift mode ("exam season"): same idea, but stress, depression and neuroticism go up and
flourishing goes down, so the new students look different from the ones the model learned from.
The monitoring job (Evidently + Grafana) should notice that difference and raise the alarm.
"""

from __future__ import annotations

import time

import httpx
import numpy as np
import pandas as pd

from spark_check.data import (
    FEATURE_COLUMNS,
    SCORE_RANGES,
    load_processed,
    split_train_test,
)

# How much random wiggle each fake student gets, in survey points (one standard deviation).
# Small compared with each scale's range, so fake students still look like real ones.
NOISE = {
    "phq9_pre": 2,
    "loneliness_pre": 4,
    "flourishing_pre": 3,
    "stress_pre": 3,
    "neuroticism_pre": 3,
}

# Exam season: how far each score moves on average when drift is switched on.
# More stressed, more depressed, more anxious, less flourishing.
EXAM_SEASON_SHIFT = {
    "phq9_pre": 6,
    "loneliness_pre": 5,
    "flourishing_pre": -8,
    "stress_pre": 8,
    "neuroticism_pre": 4,
}


def load_training_students() -> pd.DataFrame:
    """The 30 training students (same split as the notebook), with missing scores filled in."""
    train, _ = split_train_test(load_processed())
    students = train[FEATURE_COLUMNS]
    return students.fillna(students.median())


def make_students(
    n: int, drift: bool = False, seed: int | None = None, base: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Make `n` fake students by copying random training students and adding small changes."""
    rng = np.random.default_rng(seed)
    base = load_training_students() if base is None else base

    # Pick n real students at random (the same student can be picked more than once)
    picked = base.sample(n=n, replace=True, random_state=rng).reset_index(drop=True)

    fake = pd.DataFrame(index=picked.index)
    for column in FEATURE_COLUMNS:
        values = picked[column] + rng.normal(0, NOISE[column], size=n)
        if drift:
            values = values + EXAM_SEASON_SHIFT[column]
        low, high = SCORE_RANGES[column]
        # Whole numbers inside the valid range, so the API accepts every fake student
        fake[column] = values.round().clip(low, high).astype(int)
    return fake


def send_students(students: pd.DataFrame, client: httpx.Client, delay: float = 0.0) -> dict:
    """Send each student to POST /predict and count what happened."""
    counts = {"sent": 0, "ok": 0, "at_risk": 0, "errors": 0}
    for row in students.to_dict(orient="records"):
        row = {column: int(value) for column, value in row.items()}
        counts["sent"] += 1
        try:
            response = client.post("/predict", json=row)
        except httpx.HTTPError:  # API down or unreachable: count it and keep going
            counts["errors"] += 1
            continue
        if response.status_code == 200:
            counts["ok"] += 1
            counts["at_risk"] += response.json()["at_risk"]
        else:
            counts["errors"] += 1
        if delay:
            time.sleep(delay)
    return counts