# ==============================================================================
# Dockerfile for IDP Layer 1 (Routing) + Layer 2 (Conversion) Pipeline
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/root/.cache/huggingface \
    PADDLE_HOME=/root/.cache/paddle \
    PADDLEOCR_HOME=/root/.paddleocr

# Install system dependencies required for OpenCV, PyMuPDF, PaddleOCR, and Docling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for efficient layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Ensure standard runtime directories exist
RUN mkdir -p dataset dataset_output schema_registry logs data

# Default entrypoint runs the resume pipeline verification demo
CMD ["python", "demo_resume_pipeline.py"]
