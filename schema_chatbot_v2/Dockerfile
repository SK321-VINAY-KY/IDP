# ==============================================================================
# Dockerfile for IDP Web Console & Schema Discovery Chatbot (FastAPI + Frontend)
# Serves the user-facing UI at http://localhost:8000/app and orchestrates
# both the Schema Discovery Chatbot and the Layer 1+2 Document Pipeline.
# ==============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/schema_chatbot_v2:/app

# Install system dependencies required for psycopg2, PyMuPDF, PaddleOCR, and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies for Web Console, Schema Discovery, and Graph Memory
COPY schema_chatbot_v2/requirements.txt /app/schema_chatbot_v2/requirements.txt

RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r /app/schema_chatbot_v2/requirements.txt

# ----------------------------------------------------------------------------
# Phase 4 Query Bot hybrid retrieval — runtime dependency + model weights.
#
# query_service.py -> evidence_retriever.py uses sentence-transformers with the
# pinned model all-MiniLM-L6-v2 @ 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
# (384-dim, ~90MB). sentence-transformers is now listed in
# schema_chatbot_v2/requirements.txt, so the LIBRARY is installed above.
#
# The MODEL WEIGHTS are a separate concern (same spirit as the Phase 1 note in
# docker/paddle_printed/Dockerfile about files that must be present at runtime):
#   - Not baked in here, the container downloads them from Hugging Face on the
#     FIRST query. That needs outbound network at runtime + a writable
#     HF cache dir. On a locked-down/offline Fargate task the download fails and
#     evidence_retriever degrades gracefully to KEYWORD-ONLY retrieval (the
#     hybrid feature silently turns off — answers still work via the existing
#     bounded-BFS path, just without the vector rescue).
#   - To guarantee the feature is ON in the container, pre-bake the weights by
#     uncommenting the line below (adds ~90MB to the image but removes the
#     first-query latency spike and the network dependency):
# RUN python -c "from sentence_transformers import SentenceTransformer; \
#     SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', \
#     revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41')"
#
# NOTE: the OCR Lambdas (docker/paddle_*) and the Layer 1 router Lambda do NOT
# import query_service, so they need NEITHER this library NOR the weights.
# ----------------------------------------------------------------------------

# Copy complete project codebase and frontend assets.
# The Layer 1.5 probe model ships here too: the observer resolves it at
# /app/src/ai/layer1_probe/models/lipika_script.onnx (shipped by this COPY;
# .dockerignore does not exclude *.onnx / *.json). IDP_ENABLE_PROBE is OFF by
# default, so onnxruntime/opencv sit idle until the flag is turned on. No extra COPY.
COPY . /app

# Ensure standard runtime directories exist
RUN mkdir -p dataset dataset_output schema_registry logs schema_chatbot_v2/logs data schema_chatbot_v2/data

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

WORKDIR /app/schema_chatbot_v2

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
