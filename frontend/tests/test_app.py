"""Tests for the Spark Check web page (test_app.py).

ROLE OF THIS FILE: automatic checks.
Each test pretends to use the page and checks it behaves correctly.
They run in test mode, so no API is needed.

How to run them (in Terminal, inside the frontend folder):
    python3 -m pytest
You should see "4 passed".

Every test follows the same pattern:
    1. pretend to use the page
    2. "assert" = this MUST be true, otherwise the test fails
"""

# =====================================================================
# STEP 1: Find app.py
# =====================================================================
# This file sits inside the "tests" folder, but app.py is one folder up.
#   __file__                     -> this test file
#   os.path.dirname(...) once    -> the tests folder
#   os.path.dirname(...) twice   -> the frontend folder
# sys.path.insert(0, ...) tells Python to look there first.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Bring in our web page program from app.py.
from app import app

# =====================================================================
# STEP 2: One normal student with valid scores
# =====================================================================
# Reused in every test. The numbers are in quotes because web forms
# always send text.
GOOD_SCORES = {
    "phq9_pre": "12",
    "loneliness_pre": "40",
    "flourishing_pre": "45",
    "stress_pre": "20",
    "neuroticism_pre": "25",
}


# =====================================================================
# TEST 1: Does the page open?
# =====================================================================
def test_page_loads():
    # A pretend browser, so we don't need a real one.
    client = app.test_client()
    # Open the main page.
    response = client.get("/")
    # 200 is the web's code for "OK".
    assert response.status_code == 200
    # The page must contain the title. (b"..." = the page comes back as raw bytes.)
    assert b"Spark Check" in response.data


# =====================================================================
# TEST 2: Do good scores give a result?
# =====================================================================
def test_good_scores_show_a_result():
    client = app.test_client()
    # "post" = like clicking Predict with these scores.
    response = client.post("/", data=GOOD_SCORES)
    assert b"Predicted end-of-term score" in response.data


# =====================================================================
# TEST 3: Does a number that's too high show an error?
# =====================================================================
def test_out_of_range_score_shows_error():
    client = app.test_client()
    # Copy the good student, but make PHQ-9 = 40 (the limit is 27).
    bad = dict(GOOD_SCORES, phq9_pre="40")
    response = client.post("/", data=bad)
    # The PHQ-9 error MUST show...
    assert b"Please enter a whole number from 0 to 27." in response.data
    # ...and there must be NO result.
    assert b"Predicted end-of-term score" not in response.data


# =====================================================================
# TEST 4: Does an empty box show an error?
# =====================================================================
def test_empty_box_shows_error():
    client = app.test_client()
    # Copy the good student, but leave the stress box empty.
    bad = dict(GOOD_SCORES, stress_pre="")
    response = client.post("/", data=bad)
    assert b"Please enter a whole number from 0 to 40." in response.data
