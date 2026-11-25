"""Fake traffic: makes up students to send to the Spark Check API.

Each fake student is a copy of a real training student with small random changes.
"""

from __future__ import annotations

import pandas as pd

from spark_check.data import FEATURE_COLUMNS, load_processed, split_train_test

# How much random wiggle each fake student gets, in survey points (one standard deviation).
# Small compared with each scale's range, so fake students still look like real ones.
NOISE = {
    "phq9_pre": 2,
    "loneliness_pre": 4,
    "flourishing_pre": 3,
    "stress_pre": 3,
    "neuroticism_pre": 3,
}


def load_training_students() -> pd.DataFrame:
    """The 30 training students (same split as the notebook), with missing scores filled in."""
    train, _ = split_train_test(load_processed())
    students = train[FEATURE_COLUMNS]
    return students.fillna(students.median())