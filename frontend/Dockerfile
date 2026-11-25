# =====================================================================
# Dockerfile for the Spark Check web page
#
# This file acts as the recipe for packing the web page into its own
# Docker container, so it runs the same way on every computer.
# Docker reads it from top to bottom.
# =====================================================================

# STEP 1: Start from a ready-made container that already has Python 3.11.
#         "slim" = a small version without extras, so it downloads faster.
FROM python:3.11-slim

# STEP 2: Inside the container, make a folder called /app and work there.
WORKDIR /app

# STEP 3: Copy the shopping list of tools into the container.
#         ("." = into the current folder, /app)
COPY requirements.txt .

# STEP 4: Install every tool on the shopping list.
#         --no-cache-dir = don't keep the installer files, so the container stays small.
#         (We copy the list BEFORE the code: if only the code changes, Docker
#          can skip re-installing the tools, which makes rebuilding faster.)
RUN pip install --no-cache-dir -r requirements.txt

# STEP 5: Copy everything else (app.py, templates, tests) into the container.
COPY . .

# STEP 6: Label: this page uses port 5000 ("door number 5000").
EXPOSE 5000

# STEP 7: The command that runs when the container starts.
#   gunicorn            -> runs the page properly (Flask's own runner is only for testing)
#   --bind 0.0.0.0:5000 -> answer on door 5000, so other containers can reach it
#   app:app             -> open app.py and run the thing called "app" inside it
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
