# syntax=docker/dockerfile:1.7

# ---------------- builder ----------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# Bake the buffalo_l model so the runtime container starts warm.
# insightface 0.7.3 uses ~/.insightface (ignores INSIGHTFACE_HOME), so we
# pin HOME during the bake and copy that exact tree into the runtime user.
COPY scripts/download_models.py /tmp/download_models.py
RUN HOME=/iface-cache mkdir -p /iface-cache && \
    HOME=/iface-cache PYTHONPATH=/install/lib/python3.11/site-packages \
        python /tmp/download_models.py && \
    ls /iface-cache/.insightface/models/buffalo_l

# ---------------- runtime ----------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        tini \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-hin \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /var/log/aadhaar-verify \
    && chown -R app:app /var/log/aadhaar-verify

ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/

COPY --from=builder /install /usr/local
COPY --from=builder --chown=app:app /iface-cache/.insightface /home/app/.insightface

WORKDIR /app
COPY --chown=app:app . /app

USER app
EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "aav.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers", "--forwarded-allow-ips=*"]
