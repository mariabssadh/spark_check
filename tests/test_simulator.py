import httpx
from fastapi.testclient import TestClient

from api.app import app
from api.schemas import StudentScores
from simulator.simulate import (
    FEATURE_COLUMNS,
    load_training_students,
    make_students,
    send_students,
)


def test_training_students_are_the_30_from_the_notebook_split():
    students = load_training_students()
    assert len(students) == 30
    assert list(students.columns) == FEATURE_COLUMNS
    assert not students.isna().any().any()


def test_fake_students_have_valid_scores():
    students = make_students(200, seed=1)
    assert list(students.columns) == FEATURE_COLUMNS
    assert not students.isna().any().any()
    for row in students.to_dict(orient="records"):
        StudentScores(**row)  # raises an error if any score is out of range


def test_same_seed_gives_same_students():
    assert make_students(20, seed=7).equals(make_students(20, seed=7))


def test_drifted_students_also_have_valid_scores():
    students = make_students(200, drift=True, seed=1)
    for row in students.to_dict(orient="records"):
        StudentScores(**row)


def test_exam_season_drift_raises_stress_and_depression():
    normal = make_students(500, seed=3)
    exam = make_students(500, drift=True, seed=3)
    assert exam["stress_pre"].mean() > normal["stress_pre"].mean() + 5
    assert exam["phq9_pre"].mean() > normal["phq9_pre"].mean() + 3
    assert exam["flourishing_pre"].mean() < normal["flourishing_pre"].mean() - 5


def test_simulator_sends_students_to_the_api():
    with TestClient(app) as client:
        counts = send_students(make_students(10, seed=2), client)
    assert counts == {"sent": 10, "ok": 10, "at_risk": counts["at_risk"], "errors": 0}


def test_drift_flags_more_students_at_risk():
    with TestClient(app) as client:
        normal = send_students(make_students(100, seed=4), client)
        exam = send_students(make_students(100, drift=True, seed=4), client)
    assert exam["at_risk"] > normal["at_risk"]


def test_api_down_is_counted_not_a_crash():
    with httpx.Client(base_url="http://127.0.0.1:9", timeout=0.5) as client:
        counts = send_students(make_students(3, seed=5), client)
    assert counts["errors"] == 3