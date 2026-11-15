# syntax=docker/dockerfile:1
ARG APP_UID=10001
ARG PORT=8000
ARG RELEASE=v1

# ---- builder: resolve dependencies and fetch the backbone weights once ----
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[service]"

ENV TORCH_HOME=/opt/torch
RUN python -c "from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2; wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)"

# ---- runtime ----
FROM python:3.11-slim AS runtime
ARG APP_UID
ARG PORT
ARG RELEASE
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    TORCH_HOME=/opt/torch \
    PORT=${PORT} \
    AD_RELEASE_DIR=/app/releases/${RELEASE} \
    AD_DEVICE=cpu \
    AD_JSON_LOGS=true

RUN groupadd --system --gid ${APP_UID} app \
 && useradd --system --uid ${APP_UID} --gid app --no-create-home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/torch /opt/torch

WORKDIR /app
COPY releases ./releases

USER ${APP_UID}:${APP_UID}
EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os, sys, urllib.request as u; sys.exit(0 if u.urlopen('http://127.0.0.1:%s/readyz' % os.environ['PORT'], timeout=3).status == 200 else 1)"

CMD ["sh", "-c", "exec uvicorn anomaly.service.main:app --host 0.0.0.0 --port ${PORT}"]