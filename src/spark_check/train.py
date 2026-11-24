"""Step 2: train the Spark Check score model (Ridge) and track the run in MLflow."""

import os

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from spark_check.data import TARGET, load_processed, split_train_test
from spark_check.features import FEATURE_SETS, make_preprocessor

# 0. Tell MLflow where to save runs (a local database file unless MLFLOW_TRACKING_URI is set)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("spark-check-score")

# 1. Load the cleaned data and split it (same 80/20 split as INSY 674)
df = load_processed()
train, test = split_train_test(df)

feature_set = "A"
features = FEATURE_SETS[feature_set]  # Ridge uses PHQ-9 only
X_train, y_train = train[features], train[TARGET]
X_test, y_test = test[features], test[TARGET]

# The settings we want to record for this run
params = {"model": "Ridge", "alpha": 3.0, "feature_set": feature_set, "log_phq9": False}

with mlflow.start_run(run_name="ridge_A"):
    # 2. Build and train the model: preprocessing + Ridge in one pipeline
    model = Pipeline([
        ("preprocessor", make_preprocessor(features, log_phq9=params["log_phq9"])),
        ("model", Ridge(alpha=params["alpha"])),
    ])
    model.fit(X_train, y_train)

    # 3. Evaluate on the test students, and compare with a baseline that predicts the mean
    predictions = model.predict(X_test)
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    metrics = {
        "test_rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "test_mae": mean_absolute_error(y_test, predictions),
        "test_r2": r2_score(y_test, predictions),
        "baseline_rmse": np.sqrt(mean_squared_error(y_test, baseline.predict(X_test))),
    }

    # 4. Record everything in MLflow: settings, results, and the trained model itself
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, name="model", input_example=X_train.head(3))

    print(metrics)