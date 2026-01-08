FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY Requirements.txt ./
RUN python -m pip install --upgrade pip && pip install -r Requirements.txt

# Copy app
COPY . /app

EXPOSE 5000

# Use gunicorn for production; serve Flask app from app:app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--workers", "2"]
