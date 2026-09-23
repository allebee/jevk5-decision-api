FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY jevk5_api ./jevk5_api
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126 \
    && pip install --no-cache-dir '.[gpu]'

RUN useradd --create-home --uid 10001 app && mkdir -p /models && chown -R app:app /models
USER app
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/health/ready', timeout=4)"
CMD ["jevk5-api", "--host", "0.0.0.0", "--port", "8090"]
