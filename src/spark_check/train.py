"""Step 1: train the Spark Check score model (Ridge) and evaluate it on the test set."""

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from spark_check.data import TARGET, load_processed, split_train_test
from spark_check.features import FEATURE_SETS, make_preprocessor

# 1. Load the cleaned data and split it (same 80/20 split as INSY 674)
df = load_processed()
train, test = split_train_test(df)

features = FEATURE_SETS["A"]  # Ridge uses PHQ-9 only
X_train, y_train = train[features], train[TARGET]
X_test, y_test = test[features], test[TARGET]

# 2. Build the model: preprocessing + Ridge in one pipeline
model = Pipeline([
    ("preprocessor", make_preprocessor(features, log_phq9=False)),
    ("model", Ridge(alpha=3.0)),
])

# 3. Train on the training students only
model.fit(X_train, y_train)

# 4. Evaluate on the test students, and compare with a baseline that predicts the mean
rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline.predict(X_test)))

print(f"Ridge test RMSE:    {rmse:.3f}")
print(f"Baseline test RMSE: {baseline_rmse:.3f}")