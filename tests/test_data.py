"""Data tests (Job 1). ML Test Score group: data.

Most tests use a few made-up students with known answers, so they run anywhere without
downloading StudentLife. The last test checks the real committed table once it exists.
"""

import numpy as np
import pandas as pd
import pytest

from spark_check.data import (
    DATA_PATH_ENV,
    FEATURE_COLUMNS,
    PROCESSED_PATH,
    RISK_LABEL,
    TARGET,
    build_student_table,
    load_processed,
    score_loneliness,
    score_phq9,
    split_train_test,
    validate_student_table,
)
from spark_check.features import FEATURE_SETS, make_preprocessor


# --------------------------------------------------------------------------------------------
# Made-up raw surveys in the same format as the StudentLife CSVs
# --------------------------------------------------------------------------------------------
def survey(columns, rows):
    """rows = list of (uid, type, answer for every question)."""
    return pd.DataFrame(
        [[uid, kind, *[answer] * len(columns)] for uid, kind, answer in rows],
        columns=["uid", "type", *columns],
    )


PHQ9_COLS = [f"PHQ question {n}" for n in range(1, 10)]
LONELINESS_COLS = [f"{n}. loneliness question" for n in range(1, 21)]
FLOURISHING_COLS = [f"{n}. flourishing question" for n in range(1, 9)]
STRESS_COLS = [f"{n}. In the last month, stress question" for n in range(1, 11)]
BIGFIVE_COLS = [f"I see Myself as Someone Who... - {n}. trait" for n in range(1, 45)]


@pytest.fixture
def raw_surveys():
    phq9 = survey(PHQ9_COLS, [
        (" u01 ", "Pre ", "Several days"),           # messy spacing on purpose
        ("u01", "post", "More than half the days"),
        ("u02", "pre", "Not at all"),
        ("u02", "post", "Not at all"),
        ("u03", "pre", "Nearly every day"),          # no post survey -> dropped
    ])
    phq9["Response"] = "Not difficult at all"        # extra column that is not scored

    flourishing = survey(FLOURISHING_COLS, [("u01", "pre", 6), ("u02", "pre", 7), ("u03", "pre", 1)])
    flourishing.loc[1, FLOURISHING_COLS[0]] = np.nan  # u02 skipped a question -> missing total

    return {
        "phq9": phq9,
        "loneliness": survey(LONELINESS_COLS, [
            ("u01", "pre", "Sometimes"), ("u02", "pre", "Never"), ("u03", "pre", "Often"),
            ("u01", "post", "Often"),                # post answers are ignored
        ]),
        "flourishing": flourishing,
        "stress": survey(STRESS_COLS, [
            ("u01", "pre", "Fairly often"), ("u02", "pre", "Never"), ("u03", "pre", "Very often"),
        ]),
        "bigfive": survey(BIGFIVE_COLS, [
            ("u01", "pre", "Agree a little"), ("u02", "pre", "disagree strongly"),
            ("u03", "pre", "Agree strongly"),
        ]),
    }


def valid_table(n=10):
    """A small student table that passes validation (half the students at risk)."""
    phq9_post = [5, 15] * (n // 2)
    return pd.DataFrame({
        "uid": [f"u{i:02d}" for i in range(n)],
        "phq9_pre": [4, 12] * (n // 2),
        "loneliness_pre": [40.0] * n,
        "flourishing_pre": [48.0] * (n - 1) + [np.nan],  # missing predictors are allowed
        "stress_pre": [20.0] * n,
        "neuroticism_pre": [24.0] * n,
        "phq9_post": phq9_post,
        "at_risk": [int(score >= 10) for score in phq9_post],
    })


# --------------------------------------------------------------------------------------------
# Cleaning check: a known student goes in, the expected numbers come out
# --------------------------------------------------------------------------------------------
def test_known_student_gets_expected_scores(raw_surveys):
    students = build_student_table(raw_surveys).set_index("uid")
    u01 = students.loc["u01"]

    assert u01["phq9_pre"] == 9          # 9 questions x "Several days" (1)
    assert u01["phq9_post"] == 18        # 9 questions x "More than half the days" (2)
    assert u01["loneliness_pre"] == 50   # 10 x "Sometimes" (3) + 10 reversed (5 - 3 = 2)
    assert u01["flourishing_pre"] == 48  # 8 x 6
    assert u01["stress_pre"] == 22       # 6 x "Fairly often" (3) + 4 reversed (4 - 3 = 1)
    assert u01["neuroticism_pre"] == 26  # 5 x "Agree a little" (4) + 3 reversed (6 - 4 = 2)
    assert u01["at_risk"] == 1           # 18 >= 10


def test_reverse_scoring_keeps_higher_meaning_lonelier(raw_surveys):
    scores = score_loneliness(raw_surveys["loneliness"]).set_index("uid")["loneliness_pre"]
    # "Never" on everything: 10 normal items score 1, 10 reversed items score 4
    assert scores["u02"] == 10 * 1 + 10 * 4


def test_only_students_with_pre_and_post_are_kept(raw_surveys):
    paired = score_phq9(raw_surveys["phq9"])
    assert sorted(paired["uid"]) == ["u01", "u02"]   # u03 has no post survey


def test_skipped_question_leaves_total_missing(raw_surveys):
    students = build_student_table(raw_surveys).set_index("uid")
    assert np.isnan(students.loc["u02", "flourishing_pre"])


def test_unknown_answer_label_raises_error(raw_surveys):
    raw_surveys["phq9"].loc[0, PHQ9_COLS[0]] = "Sometimes?"
    with pytest.raises(ValueError, match="unexpected answer labels"):
        score_phq9(raw_surveys["phq9"])


# --------------------------------------------------------------------------------------------
# Data check: required columns, no missing outcome, every score in its normal range
# --------------------------------------------------------------------------------------------
def test_valid_table_passes():
    validate_student_table(valid_table())


@pytest.mark.parametrize(
    "break_table, message",
    [
        (lambda df: df.drop(columns="stress_pre"), "Missing columns"),
        (lambda df: df.assign(phq9_post=df["phq9_post"].where(df.index != 0)), "missing values in phq9_post"),
        (lambda df: df.assign(loneliness_pre=df["loneliness_pre"].where(df.index != 0, 95)), "loneliness_pre outside"),
        (lambda df: df.assign(uid="u00"), "duplicate student IDs"),
        (lambda df: df.assign(at_risk=0), "at_risk does not match"),
    ],
)
def test_validation_catches_bad_data(break_table, message):
    with pytest.raises(ValueError, match=message):
        validate_student_table(break_table(valid_table()))


def test_split_is_reproducible_and_stratified():
    table = valid_table(n=20)
    train_a, test_a = split_train_test(table)
    train_b, test_b = split_train_test(table)

    assert list(test_a["uid"]) == list(test_b["uid"])          # same seed, same split
    assert list(train_a["uid"]) == list(train_b["uid"])
    assert len(test_a) == 4 and len(train_a) == 16
    assert test_a[RISK_LABEL].sum() == 2                         # half at risk, like the full table


def test_data_path_can_be_set_with_environment_variable(tmp_path, monkeypatch):
    csv = tmp_path / "students.csv"
    valid_table().to_csv(csv, index=False)
    monkeypatch.setenv(DATA_PATH_ENV, str(csv))      # e.g. inside a Docker container

    assert len(load_processed()) == 10


def test_missing_data_file_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_PATH_ENV, str(tmp_path / "nowhere.csv"))
    with pytest.raises(FileNotFoundError, match=DATA_PATH_ENV):
        load_processed()


# --------------------------------------------------------------------------------------------
# Preprocessing: known input produces the expected features
# --------------------------------------------------------------------------------------------
def test_preprocessor_fills_missing_and_standardizes():
    table = valid_table()
    prepared = make_preprocessor(FEATURE_SETS["C"]).fit_transform(table[FEATURE_COLUMNS])

    assert prepared.shape == (len(table), 5)
    assert not np.isnan(prepared).any()                         # missing flourishing imputed
    assert np.allclose(prepared[:, 0].mean(), 0)                # phq9_pre standardized


# --------------------------------------------------------------------------------------------
# The real committed table (skipped until data/processed/students.csv is in the repo)
# --------------------------------------------------------------------------------------------
@pytest.mark.skipif(not PROCESSED_PATH.exists(), reason="processed table not committed yet")
def test_processed_table_matches_674_numbers():
    students = load_processed()          # also runs validate_student_table
    assert len(students) == 38
    assert students[RISK_LABEL].sum() == 7
    assert students[TARGET].notna().all()
    assert students[FEATURE_COLUMNS].isna().sum().to_dict() == {
        "phq9_pre": 0, "loneliness_pre": 0, "flourishing_pre": 3, "stress_pre": 2, "neuroticism_pre": 2,
    }

    # First five students exactly as printed in the 674 notebook
    expected = pd.DataFrame({
        "uid": ["u00", "u01", "u02", "u03", "u04"],
        "phq9_pre": [2, 5, 13, 2, 6],
        "loneliness_pre": [35, 30, 29, 47, 53],
        "flourishing_pre": [np.nan, 45, 46, 34, 27],
        "stress_pre": [24, 15, 21, 17, 21],
        "neuroticism_pre": [25, 22, 28, 27, 33],
        "phq9_post": [3, 4, 5, 4, 8],
    }).set_index("uid")
    actual = students.set_index("uid").loc[expected.index, expected.columns]
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

    # Same split as 674: 30 training students (6 at risk), 8 test students (1 at risk)
    train, test = split_train_test(students)
    assert (len(train), train[RISK_LABEL].sum(), len(test), test[RISK_LABEL].sum()) == (30, 6, 8, 1)
