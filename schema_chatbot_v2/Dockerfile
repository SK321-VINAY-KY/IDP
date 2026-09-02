# ==============================================================================
# Dockerfile for IDP Web Console & Schema Discovery Chatbot (FastAPI + Frontend)
# Serves the user-facing UI at http://localhost:8000/app and orchestrates
# both the Schema Discovery Chatbot and the Layer 1+2 Document Pipeline.
# ==============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/schema_chatbot_v2:/app \
    HF_HOME=/root/.cache/huggingface \
    PADDLE_HOME=/root/.cache/paddle \
    PADDLEOCR_HOME=/root/.paddleocr

# Install system dependencies required for PyMuPDF, OpenCV, Docling, and PaddleOCR
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

# Install dependencies for both Layer 1+2 pipeline and Web / Chatbot service
COPY requirements.txt /app/requirements.txt
COPY schema_chatbot_v2/requirements.txt /app/schema_chatbot_v2/requirements.txt

RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt -r /app/schema_chatbot_v2/requirements.txt

# Copy complete project codebase and frontend assets
COPY . /app

# Ensure standard runtime directories exist
RUN mkdir -p dataset dataset_output schema_registry logs schema_chatbot_v2/logs data schema_chatbot_v2/data

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

WORKDIR /app/schema_chatbot_v2

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
