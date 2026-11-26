FROM python:3.13-slim
WORKDIR /app
COPY simulator/requirements.txt simulator/requirements.txt
RUN pip install --no-cache-dir -r simulator/requirements.txt
COPY src/ src/
COPY data/processed/ data/processed/
COPY simulator/ simulator/
ENV PYTHONPATH=/app/src:/app
ENV SPARK_CHECK_DATA=/app/data/processed/students.csv
# In docker-compose: set API_URL=http://api:8000, and DRIFT=1 for exam season
CMD ["python", "-m", "simulator.simulate", "--n", "200"]