"""Spark Check training pipeline, orchestrated with Prefect.

The flow trains every candidate model, tracks each one as a run in MLflow, and registers the
best model of each experiment in the MLflow Model Registry (alias "champion").

Two experiments, one per prediction:
- spark-check-score: predict the end-of-term PHQ-9 score (regression, RMSE first)
- spark-check-risk:  flag students at risk, PHQ-9 >= 10 (classification, recall first)
"""

import argparse
import logging
import os
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
from prefect import flow, task
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

@task
def load_and_split():
    """Load the cleaned student table (validated by spark_check.data) and split it 80/20."""
    df = load_processed()
    return split_train_test(df)


@task
def make_cv_splits(train):
    """Same 30 CV splits for every model, stratified so each split has at-risk students."""
    return list(RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=42).split(train, train[RISK_LABEL]))

@task
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

def current_champion_score(client, model_name, main_metric):
    """The main metric of the current champion, or None if there is no champion yet."""
    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
    except mlflow.exceptions.MlflowException:
        return None  # first training run: nothing registered yet
    return client.get_run(champion.run_id).data.metrics[main_metric]

@task
def register_best(results, main_metric, higher_is_better, model_name):
    """Pick the best candidate (baselines excluded) and register it in the MLflow Model Registry.

    It only becomes the "champion" (what the API loads) if it beats the current champion,
    so a retraining run can never replace a better model with a worse one.
    """
    candidates = {name: r for name, r in results.items() if not name.startswith("baseline")}
    pick = max if higher_is_better else min
    best_name = pick(candidates, key=lambda name: candidates[name]["metrics"][main_metric])
    best = candidates[best_name]
    new_score = best["metrics"][main_metric]

    client = mlflow.MlflowClient()
    old_score = current_champion_score(client, model_name, main_metric)
    if old_score is None:
        is_better = True
    else:
        is_better = new_score > old_score if higher_is_better else new_score < old_score

    # Register every new best model, so the registry keeps a history of all retraining runs
    version = mlflow.register_model(f"runs:/{best['run_id']}/model", model_name)
    client.set_model_version_tag(model_name, version.version, "candidate", best_name)

    if is_better:
        client.set_registered_model_alias(model_name, "champion", version.version)
        print(f"  -> {best_name} ({new_score:.3f}) is the new champion: {model_name} v{version.version}")
    else:
        print(
            f"  -> {best_name} ({new_score:.3f}) does not beat the current champion "
            f"({old_score:.3f}); registered as v{version.version}, champion unchanged"
        )
    return best_name


@flow(name="spark-check-training", log_prints=True)
def training_flow():
    """Train all candidates for both experiments and register the best model of each."""
    logging.getLogger("mlflow").setLevel(logging.ERROR)  # hide MLflow's info messages
    # The at-risk models output 0/1 (whole numbers); that's expected, so hide MLflow's hint about it
    warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

    train, test = load_and_split()
    cv = make_cv_splits(train)

    champions = {}
    # (task, candidates, metric to compare on, is higher better?)
    for task_name, candidates, main_metric, higher_is_better in [
        ("score", SCORE_CANDIDATES, "cv_rmse_mean", False),
        ("risk", RISK_CANDIDATES, "cv_recall_mean", True),
    ]:
        mlflow.set_experiment(f"spark-check-{task_name}")
        print(f"{task_name.upper()} models ({main_metric}):")
        results = {}
        for name, model, feature_set, log_phq9 in candidates:
            run_id, metrics = train_and_log(name, model, feature_set, log_phq9, task_name, train, test, cv)
            results[name] = {"run_id": run_id, "metrics": metrics}
            print(f"  {name:20s} {metrics[main_metric]:.3f}")

        champions[task_name] = register_best(results, main_metric, higher_is_better, f"spark-check-{task_name}")

    return champions


# --------------------------------------------------------------------------------------------
# Retraining: the drift alert starts the same flow through a Prefect deployment
# --------------------------------------------------------------------------------------------
DEPLOYMENT_NAME = "spark-check-training/retrain"


def trigger_retraining(reason: str = "drift alert"):
    """Start a new run of the training flow. Called when monitoring detects drift.

    Needs the deployment to be running (python -m spark_check.train --serve).
    """
    from prefect.deployments import run_deployment

    print(f"Retraining requested: {reason}")
    return run_deployment(DEPLOYMENT_NAME, timeout=0)  # timeout=0: start it, don't wait


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spark Check training pipeline")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="keep running and wait for retraining requests (Prefect deployment 'retrain')",
    )
    args = parser.parse_args()

    if args.serve:
        training_flow.serve(name="retrain")
    else:
        training_flow()