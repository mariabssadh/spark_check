from simulator.simulate import FEATURE_COLUMNS, load_training_students


def test_training_students_are_the_30_from_the_notebook_split():
    students = load_training_students()
    assert len(students) == 30
    assert list(students.columns) == FEATURE_COLUMNS
    assert not students.isna().any().any()