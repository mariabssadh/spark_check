"""Training tests (Job 2). ML Test Score group: model.

The first three tests train the final models directly (no MLflow, no Prefect), so they run
in a second. The last one runs the whole Prefect flow against a temporary MLflow database.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from spark_check.data import (
    FEATURE_COLUMNS,
    RISK_LABEL,
    TARGET,
    load_processed,
    split_train_test,
)
from spark_check.train import RISK_CANDIDATES, SCORE_CANDIDATES, build_pipeline, ranking


def get_candidate(candidates, name):
    """Build the pipeline for one candidate, e.g. get_candidate(SCORE_CANDIDATES, "ridge_A")."""
    for candidate_name, model, feature_set, log_phq9 in candidates:
        if candidate_name == name:
            return build_pipeline(model, feature_set, log_phq9)
    raise KeyError(name)


@pytest.fixture(scope="module")
def data():
    train, test = split_train_test(load_processed())
    return train, test


def test_score_model_beats_baseline(data):
    """Model vs. baseline: Ridge has a lower test RMSE than always predicting the average."""
    train, test = data
    errors = {}
    for name in ["ridge_A", "baseline_mean"]:
        model = get_candidate(SCORE_CANDIDATES, name).fit(train[FEATURE_COLUMNS], train[TARGET])
        predictions = model.predict(test[FEATURE_COLUMNS])
        errors[name] = np.sqrt(np.mean((test[TARGET] - predictions) ** 2))

    assert errors["ridge_A"] < errors["baseline_mean"]


def test_risk_model_beats_phq9_rule(data):
    """Model vs. baseline: in cross-validation, the risk model catches more at-risk students
    (recall) than today's PHQ-9 >= 10 rule."""
    train, _ = data
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=42)
    recall = {}
    for name in ["logreg_C", "baseline_phq9_rule"]:
        model = get_candidate(RISK_CANDIDATES, name)
        recall[name] = cross_val_score(
            model, train[FEATURE_COLUMNS], train[RISK_LABEL], cv=cv, scoring="recall"
        ).mean()

    assert recall["logreg_C"] > recall["baseline_phq9_rule"]


def test_predictions_are_sensible(data):
    """Sensible predictions: scores stay in the PHQ-9 range (0-27), and a worse week-one
    PHQ-9 (everything else equal) means a higher predicted score and a higher risk."""
    train, _ = data
    score_model = get_candidate(SCORE_CANDIDATES, "ridge_A").fit(train[FEATURE_COLUMNS], train[TARGET])
    risk_model = get_candidate(RISK_CANDIDATES, "logreg_C").fit(train[FEATURE_COLUMNS], train[RISK_LABEL])

    # Same student, only week-one PHQ-9 changes: 0 (lowest) to 27 (highest)
    students = pd.DataFrame({
        "phq9_pre": [0.0, 5.0, 10.0, 15.0, 20.0, 27.0],
        "loneliness_pre": 40.0,
        "flourishing_pre": 45.0,
        "stress_pre": 18.0,
        "neuroticism_pre": 24.0,
    })
    scores = score_model.predict(students)
    risk = risk_model.predict_proba(students)[:, 1]

    assert ((scores >= 0) & (scores <= 27)).all()
    assert (np.diff(scores) > 0).all()
    assert (np.diff(risk) > 0).all()
    assert set(risk_model.predict(students)) <= {0, 1}


def test_ranking_breaks_recall_ties_on_precision():
    """Champion selection: with equal recall, the model with better precision ranks higher,
    so a model that flags everyone can't win on recall alone. Works for lower-is-better too."""
    flags_everyone = {"cv_recall_mean": 1.0, "cv_precision_mean": 0.2}
    selective = {"cv_recall_mean": 1.0, "cv_precision_mean": 0.6}
    assert ranking(selective, "cv_recall_mean", "cv_precision_mean", True) > ranking(
        flags_everyone, "cv_recall_mean", "cv_precision_mean", True
    )

    low_error = {"cv_rmse_mean": 4.0, "cv_mae_mean": 3.0}
    high_error = {"cv_rmse_mean": 4.5, "cv_mae_mean": 2.0}
    assert ranking(low_error, "cv_rmse_mean", "cv_mae_mean", False) > ranking(
        high_error, "cv_rmse_mean", "cv_mae_mean", False
    )


def test_training_flow_registers_champions(tmp_path, monkeypatch):
    """Integration: the full Prefect flow registers a champion for both models, and
    retraining only replaces the champion if the new model is better."""
    import mlflow
    from prefect.testing.utilities import prefect_test_harness

    from spark_check.train import training_flow

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.chdir(tmp_path)  # model files are written here, not in the repo

    with prefect_test_harness():
        champions = training_flow()
        training_flow()  # retrain with the same data: same scores, so not better

    assert champions == {"score": "ridge_A", "risk": "logreg_C"}
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    for model_name in ["spark-check-score", "spark-check-risk"]:
        versions = client.search_model_versions(f"name='{model_name}'")
        assert len(versions) == 2  # both runs are kept in the registry
        champion = client.get_model_version_by_alias(model_name, "champion")
        assert int(champion.version) == 1  # the second run did not replace the champion