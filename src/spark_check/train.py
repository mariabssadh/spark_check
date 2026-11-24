"""Train candidate models, track each one as a run in MLflow, and register the best ones.

Two experiments, one per prediction:
- spark-check-score: predict the end-of-term PHQ-9 score (regression, RMSE first)
- spark-check-risk:  flag students at risk, PHQ-9 >= 10 (classification, recall first)
"""

import logging
import os
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from spark_check.data import (
    FEATURE_COLUMNS,
    RISK_CUTOFF,
    RISK_LABEL,
    TARGET,
    load_processed,
    split_train_test,
)
from spark_check.features import FEATURE_SETS, make_preprocessor


class Phq9Rule(BaseEstimator, ClassifierMixin):
    """Today's screening rule, written as a model: flag a student if week-one PHQ-9 >= 10."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return (X["phq9_pre"] >= RISK_CUTOFF).astype(int).to_numpy()


# --------------------------------------------------------------------------------------------
# The candidates: (run name, model, feature set, log PHQ-9?). Settings come from INSY 674 tuning.
# --------------------------------------------------------------------------------------------
SCORE_CANDIDATES = [
    ("baseline_mean", DummyRegressor(strategy="mean"), "A", False),
    ("ridge_A", Ridge(alpha=3.0), "A", False),
    ("ridge_A_log", Ridge(alpha=3.0), "A", True),
    ("random_forest_C", RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=42), "C", False),
]

RISK_CANDIDATES = [
    ("baseline_phq9_rule", Phq9Rule(), "A", False),
    ("logreg_A", LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000), "A", True),
    ("logreg_B", LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000), "B", True),
    ("logreg_C", LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000), "C", True),
    ("svc_C", SVC(C=3.0, kernel="linear", class_weight="balanced"), "C", True),
    ("random_forest_C", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42), "C", True),
]


def build_pipeline(model, feature_set, log_phq9):
    """Preprocessing + model in one pipeline. The PHQ-9 rule needs no preprocessing."""
    if isinstance(model, Phq9Rule):
        return model
    features = FEATURE_SETS[feature_set]
    return Pipeline([
        ("preprocessor", make_preprocessor(features, log_phq9=log_phq9)),
        ("model", model),
    ])


def score_metrics(y_true, y_pred, prefix):
    return {
        f"{prefix}_rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        f"{prefix}_mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}_r2": r2_score(y_true, y_pred),
    }


def risk_metrics(y_true, y_pred, prefix):
    return {
        f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def train_and_log(name, model, feature_set, log_phq9, task, train, test, cv):
    """Train one candidate, evaluate it, and record everything as one MLflow run."""
    target = TARGET if task == "score" else RISK_LABEL
    # Every model gets all 5 columns; its pipeline keeps only the ones in its feature set
    X_train, y_train = train[FEATURE_COLUMNS], train[target]
    X_test, y_test = test[FEATURE_COLUMNS], test[target]
    pipeline = build_pipeline(model, feature_set, log_phq9)

    with mlflow.start_run(run_name=name) as run:
        # 1. Cross-validation on the training students: 30 splits, our main estimate
        if task == "score":
            scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}
        else:
            scoring = ["recall", "precision", "f1"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

        metrics = {}
        for metric in scoring:
            values = np.abs(cv_scores[f"test_{metric}"])  # sklearn reports errors as negatives
            metrics[f"cv_{metric}_mean"] = values.mean()
            metrics[f"cv_{metric}_std"] = values.std()

        # 2. Train on all training students, then check on the 8 held-out test students
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        if task == "score":
            metrics.update(score_metrics(y_test, y_pred, "test"))
        else:
            metrics.update(risk_metrics(y_test, y_pred, "test"))

        # 3. Record settings, results and the trained model
        mlflow.log_params({
            "model": type(model).__name__,
            "feature_set": feature_set,
            "features": ", ".join(FEATURE_SETS[feature_set]),
            "log_phq9": log_phq9,
            **{k: v for k, v in model.get_params().items() if k in {"alpha", "C", "kernel", "n_estimators", "min_samples_leaf"}},
        })
        mlflow.log_metrics(metrics)
        if not isinstance(model, Phq9Rule):  # the rule is a baseline, nothing to save
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                input_example=X_train.head(3).astype(float),  # decimals, so missing values are allowed
                serialization_format="cloudpickle",  # we only load models we trained ourselves
            )

    return run.info.run_id, metrics


def register_best(results, main_metric, higher_is_better, model_name):
    """Pick the best candidate (baselines excluded) and put it in the MLflow Model Registry.

    The new version gets the alias "champion", which is what the API loads.
    """
    candidates = {name: r for name, r in results.items() if not name.startswith("baseline")}
    pick = max if higher_is_better else min
    best_name = pick(candidates, key=lambda name: candidates[name]["metrics"][main_metric])
    best = candidates[best_name]

    version = mlflow.register_model(f"runs:/{best['run_id']}/model", model_name)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(model_name, "champion", version.version)
    client.set_model_version_tag(model_name, version.version, "candidate", best_name)

    print(f"  -> registered {best_name} as {model_name} v{version.version} (@champion)")
    return best_name


def main():
    logging.getLogger("mlflow").setLevel(logging.ERROR)  # hide MLflow's info messages
    # The at-risk models output 0/1 (whole numbers); that's expected, so hide MLflow's hint about it
    warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

    df = load_processed()
    train, test = split_train_test(df)

    # Same 30 CV splits for every model, stratified so each split has at-risk students
    cv = list(RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=42).split(train, train[RISK_LABEL]))

    # (task, candidates, metric to compare on, is higher better?)
    for task, candidates, main_metric, higher_is_better in [
        ("score", SCORE_CANDIDATES, "cv_rmse_mean", False),
        ("risk", RISK_CANDIDATES, "cv_recall_mean", True),
    ]:
        mlflow.set_experiment(f"spark-check-{task}")
        print(f"\n{task.upper()} models ({main_metric}):")
        results = {}
        for name, model, feature_set, log_phq9 in candidates:
            run_id, metrics = train_and_log(name, model, feature_set, log_phq9, task, train, test, cv)
            results[name] = {"run_id": run_id, "metrics": metrics}
            print(f"  {name:20s} {metrics[main_metric]:.3f}")

        register_best(results, main_metric, higher_is_better, model_name=f"spark-check-{task}")


if __name__ == "__main__":
    main()