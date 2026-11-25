import sys
import types

from api.db import save_prediction


def test_prediction_values_go_in_the_right_columns(monkeypatch):
    """Each score is saved under its own column name, whatever order the dict is in."""
    saved = {}

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql, params=None):
            if params is not None:
                saved["sql"], saved["params"] = sql, params

    fake_psycopg = types.SimpleNamespace(connect=lambda url: FakeConnection())
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")

    # Deliberately shuffled order
    inputs = {
        "neuroticism_pre": 25,
        "phq9_pre": 8,
        "stress_pre": 18,
        "loneliness_pre": 45,
        "flourishing_pre": 40,
    }
    save_prediction(inputs, 9.5, False, "test")

    columns = saved["sql"].split("(")[1].split(")")[0].replace("\n", " ").split(",")
    row = dict(zip([c.strip() for c in columns], saved["params"]))
    assert row["phq9_pre"] == 8
    assert row["loneliness_pre"] == 45
    assert row["flourishing_pre"] == 40
    assert row["stress_pre"] == 18
    assert row["neuroticism_pre"] == 25
    assert row["predicted_phq9_post"] == 9.5
    assert row["at_risk"] is False
    assert row["model_version"] == "test"
