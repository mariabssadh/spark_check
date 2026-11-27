# Training service: trains all candidate models once, registers the champions in MLflow,
# then stays on call for retraining requests (Prefect deployment spark-check-training/retrain).
FROM python:3.13-slim

WORKDIR /app

# Install the spark_check package and its dependencies (pandas, scikit-learn, MLflow, Prefect).
# scikit-learn is pinned to the same version as the API, so saved models load without surprises.
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir . "scikit-learn==1.9.1"

# The cleaned student table produced by the data step
COPY data/processed/students.csv data/processed/students.csv
ENV SPARK_CHECK_DATA=/app/data/processed/students.csv

# Set by docker-compose:
#   MLFLOW_TRACKING_URI  e.g. http://mlflow:5000            (where runs and models are saved)
#   PREFECT_API_URL      e.g. http://prefect-server:4200/api (where the retrain deployment lives)
CMD ["python", "-m", "spark_check.train", "--serve"]