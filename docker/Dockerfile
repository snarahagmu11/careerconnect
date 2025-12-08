FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required for FAISS + chatbot
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . /app

# Ensure config folder exists in container
RUN mkdir -p /app/config /app/data/db

EXPOSE 8010 8501
CMD ["uvicorn", "src.api_server_http:app", "--host", "0.0.0.0", "--port", "8010", "--log-level", "info", "--workers", "1"]

