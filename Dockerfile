# Single-container image: FastAPI serving the model + the static frontend.
FROM python:3.12-slim

WORKDIR /app

# libgomp1 = the OpenMP runtime XGBoost needs at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
COPY frontend/ frontend/
COPY models/baseline.joblib models/baseline.joblib

ENV PYTHONPATH=/app/src
# Hugging Face Spaces (Docker SDK) expects the app on port 7860.
EXPOSE 7860
CMD ["uvicorn", "xg.serve.app:app", "--host", "0.0.0.0", "--port", "7860"]
