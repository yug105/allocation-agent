# Runs on CPU. No GPU, no external services, no database server.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir "uvicorn[standard]"

# Pre-trained model + held-out demo slice. The container never sees the source
# data and never trains on boot.
COPY artifacts ./artifacts

# Hugging Face Spaces expects 7860; everything else honours $PORT.
ENV PORT=7860 ARTIFACTS_DIR=/app/artifacts
EXPOSE 7860
CMD ["sh", "-c", "uvicorn allocation_agent.api:create_app --factory --host 0.0.0.0 --port ${PORT:-7860}"]
