"""Spark Check web page (app.py).

This file acts as the "brain" of the web page.
It reads what the person types, checks it, gets a prediction from the
model API, and fills in the page (templates/index.html) with the result.

If API_URL is not set, the page runs in "test mode" and shows a pretend
answer, so it can be built and tried before the real API is ready.
"""

# =====================================================================
# STEP 1: Bring in the tools we need
# =====================================================================
# os       -> reads settings from the computer (we use it for API_URL)
# requests -> sends the scores to the model API and gets the answer back
# Flask    -> the tool that turns Python code into a web page
#   - Flask:           creates the web page program
#   - render_template: fills in the blanks in index.html
#   - request:         reads what the person typed in the form
import os

import requests
from flask import Flask, render_template, request

# =====================================================================
# STEP 2: Create the web page program
# =====================================================================
# "app" is our web page. Everything below attaches to it.
# __name__ is Python's name for "this file"; Flask uses it to find the
# templates folder.
app = Flask(__name__)

# =====================================================================
# STEP 3: Where is the model API?
# =====================================================================
# Look for a setting called API_URL (set later in docker-compose.yml,
# e.g. http://api:8000/predict).
# If there is no such setting, use "" (empty) -> the page runs in TEST MODE
# and shows a pretend answer instead of calling the API.
API_URL = os.getenv("API_URL", "")

# =====================================================================
# STEP 4: The 5 survey scores the form asks for
# =====================================================================
# Each line describes ONE box on the page:
#   (name sent to the API, label the person sees, lowest allowed, highest allowed)
# The names match last term's notebook and the API.
# The page draws its boxes AND checks the ranges from this one list,
# so to change a range, change it only here.
FIELDS = [
    ("phq9_pre", "Depression (PHQ-9)", 0, 27),
    ("loneliness_pre", "Loneliness (UCLA Loneliness Scale)", 20, 80),
    ("flourishing_pre", "Flourishing (Flourishing Scale)", 8, 56),
    ("stress_pre", "Stress (Perceived Stress Scale)", 0, 40),
    ("neuroticism_pre", "Neuroticism (Big Five Inventory)", 8, 40),
]


# =====================================================================
# STEP 5: Check what the person typed
# =====================================================================
def check_inputs(form):
    """Read the 5 scores from the form.

    Returns two things:
      scores -> the valid whole numbers, ready to send to the API
      errors -> a message for each box that is empty, not a whole number,
                or outside its allowed range
    """
    # 5a. Start with two empty containers: one for good numbers, one for problems.
    scores, errors = {}, {}

    # 5b. Go through the 5 boxes one at a time.
    #     (_label has a "_" in front because we don't need it here.)
    for name, _label, low, high in FIELDS:
        # 5c. Get what was typed in this box and remove extra spaces.
        raw = form.get(name, "").strip()

        # 5d. Prepare the error message, e.g.
        #     "Please enter a whole number from 0 to 27."
        message = f"Please enter a whole number from {low} to {high}."

        # 5e. Try to turn the text into a whole number.
        #     If it fails (empty box, letters, decimals), save the error
        #     and skip to the next box.
        try:
            value = int(raw)
        except ValueError:
            errors[name] = message
            continue

        # 5f. Is the number inside the allowed range?
        #     Yes -> keep it.  No -> save the error.
        if low <= value <= high:
            scores[name] = value
        else:
            errors[name] = message

    # 5g. Hand back the good numbers and the problems.
    return scores, errors


# =====================================================================
# STEP 6: Get the prediction
# =====================================================================
def get_prediction(scores):
    """Ask the model API for a prediction (or give a pretend one in test mode).

    The API's answer looks like:
      {"predicted_phq9_post": 8.2, "at_risk": false, "model_version": "..."}
    """
    # 6a. TEST MODE: no API address yet -> return a pretend answer
    #     in the same format the real API uses.
    if not API_URL:
        return {"predicted_phq9_post": 8.0, "at_risk": False}

    # 6b. REAL MODE: send the 5 scores to the API.
    #     json=scores -> pack the scores in JSON, the format programs use to talk
    #     timeout=5   -> give up if there is no answer within 5 seconds
    response = requests.post(API_URL, json=scores, timeout=5)

    # 6c. If the API answered with an error, raise an alarm
    #     (caught in Step 7, which shows a friendly message instead).
    response.raise_for_status()

    # 6d. Unpack the API's answer and hand it back.
    return response.json()


# =====================================================================
# STEP 7: The page itself
# =====================================================================
# @app.route("/") connects the main web address (localhost:5000/) to the
# function below. It accepts two kinds of visits:
#   GET  -> someone opens the page
#   POST -> someone clicks "Predict" and sends the form
@app.route("/", methods=["GET", "POST"])
def index():
    # 7a. Start with everything empty.
    #     values    -> what was typed (so it stays in the boxes)
    #     errors    -> problems with the input
    #     result    -> the prediction (None = nothing yet)
    #     api_error -> a message if the API can't be reached
    values, errors, result, api_error = {}, {}, None, None

    # 7b. Only when the person clicked "Predict":
    if request.method == "POST":
        # Remember what was typed in each box, so the numbers stay in the
        # boxes after the page reloads (no retyping after one mistake).
        values = {name: request.form.get(name, "") for name, *_ in FIELDS}

        # Check the input (Step 5).
        scores, errors = check_inputs(request.form)

        # 7c. If there were no problems, get the prediction (Step 6).
        #     If the API is down or too slow, show a friendly message
        #     instead of crashing.
        if not errors:
            try:
                result = get_prediction(scores)
            except requests.RequestException:
                api_error = (
                    "The prediction service isn't available right now. "
                    "Please try again in a moment."
                )

    # 7d. Fill in the blanks in templates/index.html and send the finished
    #     page to the browser. Each line hands one thing to the page.
    return render_template(
        "index.html",
        fields=FIELDS,  # the 5 boxes to draw
        values=values,  # what to put back in the boxes
        errors=errors,  # which red error messages to show
        result=result,  # the prediction, if there is one
        api_error=api_error,  # the "service isn't available" message, if needed
        test_mode=not API_URL,  # True -> show the yellow "Test mode" note
    )


# =====================================================================
# STEP 8: Health check
# =====================================================================
# A second address, localhost:5000/health, that just answers "ok".
# Docker and the monitoring tools use it to check the page is alive.
@app.route("/health")
def health():
    return {"status": "ok"}


# =====================================================================
# STEP 9: Start the page on your own laptop
# =====================================================================
# Runs only when you type "python3 app.py" yourself.
# The page then lives at http://localhost:5000 (port 5000 = "door 5000").
# debug=True shows detailed error messages while building.
# (Inside Docker, gunicorn starts the page instead, so this part is skipped.)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
