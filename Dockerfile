# Use Python 3.11 slim as base for smaller image size
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UPLOADS_PATH=/app/uploads
ENV DATA_PATH=/app/data/evaluations

# Install system dependencies for OpenCV and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Pre-download AI models to the container (Instant startup)
RUN python pre_download.py

# Create persistent directories (if they don't exist)
RUN mkdir -p /app/uploads /app/data/evaluations

# Expose the port Flask runs on
EXPOSE 5000

# Run with Gunicorn for production performance
# -w 2: 2 workers
# -b 0.0.0.0:5000: bind to all interfaces on port 5000
# --timeout 120: allow long forensic processing
# --preload: Loads the AI models into the master process ONCE, sharing memory to all workers (Saves 500MB RAM & fixes slow cold-starts)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app", "--timeout", "120", "--preload"]
