"""Fake traffic: makes up students to send to the Spark Check API.

Each fake student is a copy of a real training student with small random changes.
"""

from __future__ import annotations

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


def load_training_students() -> pd.DataFrame:
    """The 30 training students (same split as the notebook), with missing scores filled in."""
    train, _ = split_train_test(load_processed())
    students = train[FEATURE_COLUMNS]
    return students.fillna(students.median())


def make_students(
    n: int, seed: int | None = None, base: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Make `n` fake students by copying random training students and adding small changes."""
    rng = np.random.default_rng(seed)
    base = load_training_students() if base is None else base

    # Pick n real students at random (the same student can be picked more than once)
    picked = base.sample(n=n, replace=True, random_state=rng).reset_index(drop=True)

    fake = pd.DataFrame(index=picked.index)
    for column in FEATURE_COLUMNS:
        values = picked[column] + rng.normal(0, NOISE[column], size=n)
        low, high = SCORE_RANGES[column]
        # Whole numbers inside the valid range, so the API accepts every fake student
        fake[column] = values.round().clip(low, high).astype(int)
    return fake