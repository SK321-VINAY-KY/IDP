# ==============================================================================
# Dockerfile for IDP Layer 2 — PaddleOCR Printed Engine (AWS Lambda)
# ==============================================================================

FROM public.ecr.aws/lambda/python:3.12

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PADDLEOCR_HOME=/opt/paddleocr \
    PADDLE_HOME=/opt/paddle

# Install system dependencies required for OpenCV and PaddlePaddle
RUN dnf install -y mesa-libGL glib2 libgomp && \
    dnf clean all

WORKDIR ${LAMBDA_TASK_ROOT}

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download and cache model weights into the container image
RUN mkdir -p /opt/paddleocr /opt/paddle /opt/paddlex && \
    python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='en', device='cpu', enable_mkldnn=False, use_textline_orientation=True)" && \
    (cp -r /tmp/.paddlex/* /opt/paddlex/ 2>/dev/null || true) && \
    (cp -r /tmp/.paddleocr/* /opt/paddleocr/ 2>/dev/null || true) && \
    chmod -R 755 /opt/paddleocr /opt/paddle /opt/paddlex

# Copy engine and handler
COPY engine.py ${LAMBDA_TASK_ROOT}/
COPY handler.py ${LAMBDA_TASK_ROOT}/

CMD [ "handler.lambda_handler" ]
