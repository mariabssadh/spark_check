FROM python:3.13-slim
WORKDIR /app
COPY monitoring/requirements.txt monitoring/requirements.txt
RUN pip install --no-cache-dir -r monitoring/requirements.txt
COPY monitoring/ monitoring/
CMD ["python", "-m", "monitoring.evaluate", "--watch"]
