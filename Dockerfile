FROM python:3.12-slim

ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HARNESS_HOST=0.0.0.0 \
    HARNESS_PORT=8000 \
    HARNESS_DB_PATH=/data/harness.db \
    HARNESS_WORKSPACE_ROOT=/data/workspaces \
    HARNESS_SANDBOX_BACKEND=local

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -i "$PIP_INDEX_URL" .

RUN useradd --create-home --uid 10001 harness \
    && mkdir -p /data \
    && chown -R harness:harness /data /app

USER harness
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

CMD ["python", "-m", "harness"]
