"""Load, clean and score the StudentLife surveys used by Spark Check.

Moved out of the INSY 674 notebook (sections "Data Acquisition" to "Test-Train Split").
The scoring rules are the published ones for each questionnaire, so nothing here is learned
from the data: imputation and scaling happen later, inside the model pipeline (features.py).

Build the processed table once and save it in the repo:

    python -m spark_check.data                       # downloads from Kaggle (kagglehub)
    python -m spark_check.data --raw-dir data/raw    # or uses CSVs you downloaded yourself

Everyone else just calls ``load_processed()``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------------------------
# Constants shared by the rest of the project
# --------------------------------------------------------------------------------------------
KAGGLE_DATASET = "dartweichen/student-life"
SURVEY_FILES = {
    "phq9": "PHQ-9.csv",
    "loneliness": "LonelinessScale.csv",
    "flourishing": "FlourishingScale.csv",
    "stress": "PerceivedStressScale.csv",
    "bigfive": "BigFive.csv",
}

FEATURE_COLUMNS = ["phq9_pre", "loneliness_pre", "flourishing_pre", "stress_pre", "neuroticism_pre"]
TARGET = "phq9_post"          # regression target (0-27)
RISK_LABEL = "at_risk"        # classification target (1 = PHQ-9 post >= 10)
RISK_CUTOFF = 10              # Kroenke et al. (2001) cutoff for moderate depression

# Valid range of each total score, from the published scoring of each questionnaire
SCORE_RANGES = {
    "phq9_pre": (0, 27),
    "loneliness_pre": (20, 80),
    "flourishing_pre": (8, 56),
    "stress_pre": (0, 40),
    "neuroticism_pre": (8, 40),
    "phq9_post": (0, 27),
}

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "students.csv"


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------
def load_survey(file_name: str, raw_dir: str | Path | None = None) -> pd.DataFrame:
    """Load one raw survey file, from a local folder if given, otherwise from Kaggle."""
    if raw_dir is not None:
        return pd.read_csv(Path(raw_dir) / file_name)

    import kagglehub  # only needed when downloading
    from kagglehub import KaggleDatasetAdapter

    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS, KAGGLE_DATASET, f"dataset/survey/{file_name}"
    )


def load_raw_surveys(raw_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Load all five raw surveys into a dict keyed like SURVEY_FILES."""
    return {name: load_survey(file, raw_dir) for name, file in SURVEY_FILES.items()}


# --------------------------------------------------------------------------------------------
# Scoring each survey
# --------------------------------------------------------------------------------------------
def _standardize_ids(raw: pd.DataFrame) -> pd.DataFrame:
    """Copy the table and strip spaces from student IDs and survey labels ("Pre " -> "pre")."""
    df = raw.copy()
    df["uid"] = df["uid"].str.strip()
    df["type"] = df["type"].str.strip().str.lower()
    return df


def _question_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in ["uid", "type"]]


def _encode(answers: pd.DataFrame, scores: dict[str, int], survey: str) -> pd.DataFrame:
    """Turn answer labels into numbers. Unknown labels raise an error instead of becoming NaN."""
    cleaned = answers.apply(lambda column: column.str.strip())
    encoded = cleaned.apply(lambda column: column.map(scores))
    unknown = encoded.isna() & cleaned.notna()
    if unknown.any().any():
        labels = sorted({str(label) for label in cleaned.to_numpy()[unknown.to_numpy()]})
        raise ValueError(f"{survey}: unexpected answer labels {labels}")
    return encoded


def score_phq9(raw: pd.DataFrame) -> pd.DataFrame:
    """PHQ-9: 9 items scored 0-3, total 0-27. One row per student with pre and post totals.

    Students without both a pre and a post survey are dropped (we need the outcome).
    """
    df = _standardize_ids(raw)
    items = list(raw.columns[2:11])  # the nine symptom questions (the extra "Response" column is not scored)
    scores = {"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}

    df[items] = _encode(df[items], scores, "PHQ-9")
    df["phq9_total"] = df[items].sum(axis=1, min_count=9)

    paired = (
        df.pivot(index="uid", columns="type", values="phq9_total")
        .rename(columns={"pre": "phq9_pre", "post": "phq9_post"})
        .rename_axis(columns=None)
        .reset_index()
        .dropna(subset=["phq9_pre", "phq9_post"])
        .reset_index(drop=True)
    )
    paired[["phq9_pre", "phq9_post"]] = paired[["phq9_pre", "phq9_post"]].astype(int)
    return paired[["uid", "phq9_pre", "phq9_post"]]


def score_loneliness(raw: pd.DataFrame) -> pd.DataFrame:
    """UCLA Loneliness (v3): 20 items scored 1-4, 10 reverse-scored, total 20-80. Pre-survey only."""
    df = _standardize_ids(raw)
    df = df.loc[df["type"] == "pre"].copy()
    items = _question_columns(df)

    if len(items) != 20:
        raise ValueError(f"Loneliness: expected 20 questions, found {len(items)}")
    if df["uid"].duplicated().any():
        raise ValueError("Loneliness: duplicate pre-survey responses")
    if df[items].isna().any().any():
        raise ValueError("Loneliness: missing questionnaire responses")

    scores = {"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4}
    df[items] = _encode(df[items], scores, "Loneliness")

    # Positively worded questions are reverse-scored so higher always means lonelier
    reverse_numbers = {1, 4, 5, 6, 9, 10, 15, 16, 19, 20}
    reverse_columns = [c for c in items if int(c.split(".")[0].strip()) in reverse_numbers]
    if len(reverse_columns) != 10:
        raise ValueError("Loneliness: check reverse-scored column names")
    df[reverse_columns] = 5 - df[reverse_columns]

    df["loneliness_pre"] = df[items].sum(axis=1, min_count=20)
    return df[["uid", "loneliness_pre"]].reset_index(drop=True)


def score_flourishing(raw: pd.DataFrame) -> pd.DataFrame:
    """Flourishing Scale: 8 items already numeric (1-7), total 8-56. Pre-survey only.

    Students who skipped a question keep a missing total; it is imputed later in the pipeline.
    """
    df = _standardize_ids(raw)
    df = df.loc[df["type"] == "pre"].copy()
    items = _question_columns(df)

    if len(items) != 8:
        raise ValueError(f"Flourishing: expected 8 questions, found {len(items)}")
    valid = df[items].isin(range(1, 8)) | df[items].isna()
    if not valid.all().all():
        raise ValueError("Flourishing: item values outside 1-7")

    df["flourishing_pre"] = df[items].sum(axis=1, min_count=8)
    return df[["uid", "flourishing_pre"]].reset_index(drop=True)


def score_stress(raw: pd.DataFrame) -> pd.DataFrame:
    """Perceived Stress Scale (PSS-10): 10 items scored 0-4, items 4, 5, 7, 8 reversed, total 0-40."""
    df = _standardize_ids(raw)
    df = df.loc[df["type"] == "pre"].copy()
    items = _question_columns(df)

    if len(items) != 10:
        raise ValueError(f"Stress: expected 10 questions, found {len(items)}")

    scores = {"Never": 0, "Almost never": 1, "Sometime": 2, "Fairly often": 3, "Very often": 4}
    df[items] = _encode(df[items], scores, "Stress")

    reverse_columns = [c for c in items if int(c.split(".")[0]) in {4, 5, 7, 8}]
    if len(reverse_columns) != 4:
        raise ValueError("Stress: check reverse-scored column names")
    df[reverse_columns] = 4 - df[reverse_columns]

    df["stress_pre"] = df[items].sum(axis=1, min_count=10)
    return df[["uid", "stress_pre"]].reset_index(drop=True)


def _bfi_item_number(column: str) -> int:
    """Item number is the text between "- " and the first "." in each Big Five column name."""
    return int(column.split("- ")[1].split(".")[0])


def score_neuroticism(raw: pd.DataFrame) -> pd.DataFrame:
    """Big Five (BFI-44) Neuroticism subscale: items 4, 9, 14, 19, 24, 29, 34, 39 scored 1-5,
    items 9, 24, 34 reversed, total 8-40 (John & Srivastava, 1999). Pre-survey only."""
    df = _standardize_ids(raw)
    df = df.loc[df["type"] == "pre"].copy()

    neuroticism_numbers = {4, 9, 14, 19, 24, 29, 34, 39}
    items = [c for c in _question_columns(df) if _bfi_item_number(c) in neuroticism_numbers]
    if len(items) != 8:
        raise ValueError(f"Neuroticism: expected 8 items, found {len(items)}")

    scores = {
        "disagree strongly": 1,
        "disagree a little": 2,
        "neither agree nor disagree": 3,
        "agree a little": 4,
        "agree strongly": 5,
    }
    df[items] = _encode(df[items].apply(lambda column: column.str.lower()), scores, "Neuroticism")

    reverse_columns = [c for c in items if _bfi_item_number(c) in {9, 24, 34}]
    df[reverse_columns] = 6 - df[reverse_columns]

    df["neuroticism_pre"] = df[items].sum(axis=1, min_count=8)
    return df[["uid", "neuroticism_pre"]].reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# One table for the model
# --------------------------------------------------------------------------------------------
def build_student_table(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Score every survey and merge them into one row per student, plus the at-risk label."""
    students = score_phq9(raw["phq9"])
    for scored in [
        score_loneliness(raw["loneliness"]),
        score_flourishing(raw["flourishing"]),
        score_stress(raw["stress"]),
        score_neuroticism(raw["bigfive"]),
    ]:
        students = students.merge(scored, on="uid", how="left", validate="one_to_one")

    students[RISK_LABEL] = (students[TARGET] >= RISK_CUTOFF).astype(int)
    return students[["uid", *FEATURE_COLUMNS, TARGET, RISK_LABEL]]


def validate_student_table(df: pd.DataFrame) -> None:
    """Data check: required columns, one row per student, no missing outcome, valid ranges.

    Raises ValueError listing every problem found (empty list = data is fine).
    """
    required = ["uid", *FEATURE_COLUMNS, TARGET, RISK_LABEL]
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    problems = []
    if not df["uid"].is_unique:
        problems.append("duplicate student IDs")
    for column in ["phq9_pre", TARGET, RISK_LABEL]:
        if df[column].isna().any():
            problems.append(f"missing values in {column}")
    for column, (low, high) in SCORE_RANGES.items():
        values = df[column].dropna()
        if not values.between(low, high).all():
            problems.append(f"{column} outside {low}-{high}")
    expected_risk = (df[TARGET] >= RISK_CUTOFF).astype(int)
    if not df[RISK_LABEL].equals(expected_risk):
        problems.append(f"{RISK_LABEL} does not match {TARGET} >= {RISK_CUTOFF}")

    if problems:
        raise ValueError("Student table failed validation: " + "; ".join(problems))


def split_train_test(
    df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same split as the 674 notebook: 80/20, stratified on the at-risk flag, seed 42."""
    return train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[RISK_LABEL]
    )


def load_processed(path: str | Path = PROCESSED_PATH) -> pd.DataFrame:
    """Load the saved student table (the one committed in data/processed/) and validate it."""
    df = pd.read_csv(path)
    validate_student_table(df)
    return df


# --------------------------------------------------------------------------------------------
# Command line: build data/processed/students.csv
# --------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the cleaned Spark Check student table.")
    parser.add_argument("--raw-dir", help="Folder with the raw survey CSVs (default: download from Kaggle)")
    parser.add_argument("--out", default=str(PROCESSED_PATH), help="Where to save the table")
    args = parser.parse_args(argv)

    students = build_student_table(load_raw_surveys(args.raw_dir))
    validate_student_table(students)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    students.to_csv(out, index=False)

    print(f"Saved {len(students)} students to {out}")
    print(f"At risk: {students[RISK_LABEL].sum()} | missing predictor totals:")
    print(students[FEATURE_COLUMNS].isna().sum().to_string())


if __name__ == "__main__":
    main()
