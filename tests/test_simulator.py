from api.schemas import StudentScores
from simulator.simulate import FEATURE_COLUMNS, load_training_students, make_students


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