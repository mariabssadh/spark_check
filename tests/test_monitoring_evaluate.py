import pandas as pd
import pytest

from monitoring.evaluate import OUTCOME, PREDICTION, evaluate
from monitoring.storage import FEATURES


@pytest.fixture
def reference():
    pytest.importorskip("evidently")
    return pd.DataFrame(
        {column: [float(i % 5) for i in range(30)] for column in FEATURES}
    )


def test_normal_and_shifted_inputs(reference):
    normal = evaluate(reference.to_dict("records"), reference, 20, 0.5)
    assert normal.drift_detected is False
    assert normal.drift_share == 0
    assert normal.mae is None and normal.rmse is None
    assert normal.labeled_count == 0

    shifted = reference.copy()
    shifted[list(FEATURES[:3])] += 10
    drifted = evaluate(shifted.to_dict("records"), reference, 20, 0.5)
    assert drifted.drift_detected is True
    assert drifted.drift_share == pytest.approx(0.6)
    assert set(drifted.feature_drift) == set(FEATURES)


def test_performance_only_uses_labeled_pairs(reference):
    rows = [
        {PREDICTION: 7, OUTCOME: 10},
        {PREDICTION: 12, OUTCOME: 11},
        {PREDICTION: 27, OUTCOME: None},
    ]
    result = evaluate(rows, reference, 20, 0.5)
    assert result.prediction_count == 3
    assert result.labeled_count == 2
    assert result.drift_detected is None
    assert result.mae == 2
    assert result.rmse == pytest.approx(5**0.5)


def test_single_labeled_prediction(reference):
    result = evaluate([{PREDICTION: 7, OUTCOME: 10}], reference, 20, 0.5)
    assert result.mae == result.rmse == 3
