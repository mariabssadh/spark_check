"""Preprocessing pipeline (moved from the INSY 674 notebook, section "Prepare the Data").

Lives inside the model Pipeline, so imputation medians and scaling are learned from the
training data only and re-fit inside every cross-validation fold.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from spark_check.data import FEATURE_COLUMNS

# Nested feature sets compared in 674. Final models: Ridge + A (score), Logistic Regression + C (flag)
FEATURE_SETS = {
    "A": ["phq9_pre"],
    "B": ["phq9_pre", "loneliness_pre", "flourishing_pre"],
    "C": FEATURE_COLUMNS,
}


def make_preprocessor(features: list[str], log_phq9: bool = True) -> ColumnTransformer:
    """Build the preprocessing steps for a given list of predictor columns.

    - phq9_pre: optional log1p (reduces right skew), then standardize
    - all other predictors: impute missing totals with the training median, then standardize
    """
    transformers = []

    if "phq9_pre" in features:
        phq9_steps = [("scale", StandardScaler())]
        if log_phq9:
            phq9_steps.insert(0, ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
        transformers.append(("phq9", Pipeline(phq9_steps), ["phq9_pre"]))

    other_features = [f for f in features if f != "phq9_pre"]
    if other_features:
        transformers.append(
            (
                "surveys",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                other_features,
            )
        )

    return ColumnTransformer(transformers)
