# ==============================================================================
# Dockerfile for IDP Layer 2 — Docling Digital Conversion Engine (AWS Lambda)
# ==============================================================================

FROM public.ecr.aws/lambda/python:3.12

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/huggingface_cache

# Install essential system build libraries
RUN dnf install -y mesa-libGL glib2 libgomp && \
    dnf clean all

WORKDIR ${LAMBDA_TASK_ROOT}

# Install PyTorch CPU-only first to prevent downloading 3+ GB of CUDA GPU packages
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy engine and handler
COPY engine.py ${LAMBDA_TASK_ROOT}/
COPY handler.py ${LAMBDA_TASK_ROOT}/

CMD [ "handler.lambda_handler" ]
