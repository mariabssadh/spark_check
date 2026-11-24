"""Request/response shapes and the allowed range of each survey score."""

from pydantic import BaseModel, Field

# Allowed range for each start-of-term survey score.
# Ranges confirmed against the INSY 674 notebook.
# Neuroticism is 8 questions scored 1-5, so 8-40.
RANGES = {
    "phq9_pre": (0, 27),
    "loneliness_pre": (20, 80),
    "flourishing_pre": (8, 56),
    "stress_pre": (0, 40),
    "neuroticism_pre": (8, 40),
}


class StudentScores(BaseModel):
    phq9_pre: int = Field(ge=RANGES["phq9_pre"][0], le=RANGES["phq9_pre"][1])
    loneliness_pre: int = Field(
        ge=RANGES["loneliness_pre"][0], le=RANGES["loneliness_pre"][1]
    )
    flourishing_pre: int = Field(
        ge=RANGES["flourishing_pre"][0], le=RANGES["flourishing_pre"][1]
    )
    stress_pre: int = Field(ge=RANGES["stress_pre"][0], le=RANGES["stress_pre"][1])
    neuroticism_pre: int = Field(
        ge=RANGES["neuroticism_pre"][0], le=RANGES["neuroticism_pre"][1]
    )


class Prediction(BaseModel):
    predicted_phq9_post: float  # end-of-term depression score, 0-27
    at_risk: bool  # True when the predicted score is 10 or more
    model_version: str
