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
COPY docker/paddle_printed/requirements.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download and cache model weights into the container image
RUN mkdir -p /opt/paddleocr /opt/paddle /opt/paddlex && \
    python -c "from paddleocr import PaddleOCR; PaddleOCR(ocr_version='PP-OCRv6', lang='en', device='cpu', enable_mkldnn=False, use_doc_unwarping=False, use_doc_orientation_classify=False, use_textline_orientation=False)" && \
    (cp -r /tmp/.paddlex/* /opt/paddlex/ 2>/dev/null || true) && \
    (cp -r /tmp/.paddleocr/* /opt/paddleocr/ 2>/dev/null || true) && \
    chmod -R 755 /opt/paddleocr /opt/paddle /opt/paddlex

# Copy engine, handler, and shared schema dependencies.
# document_ir.py and paddle_block_rollup.py must be present at the Lambda
# task root because engine.py imports them via the ImportError fallback
# ("from document_ir import ..." / "from paddle_block_rollup import ...").
# They are NOT auto-included by the build context; omitting either causes
# an ImportError at container cold-start.
#
# NOTE: This Dockerfile must be built with the REPO ROOT as the build context:
#   docker build --provenance=false --platform linux/amd64 \
#     -f docker/paddle_printed/Dockerfile \
#     -t <tag> .
# The deploy_paddle_lambdas.py and deploy_layer2.py scripts have been updated
# to pass the repo root as the build context.
COPY docker/paddle_printed/engine.py ${LAMBDA_TASK_ROOT}/
COPY docker/paddle_printed/handler.py ${LAMBDA_TASK_ROOT}/
COPY src/ai/schemas/document_ir.py ${LAMBDA_TASK_ROOT}/
COPY src/ai/layer2_conversion/paddle_block_rollup.py ${LAMBDA_TASK_ROOT}/

CMD [ "handler.lambda_handler" ]
