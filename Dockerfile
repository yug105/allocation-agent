# Runs on CPU. No GPU, no external services, no database server.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# LightGBM links against the GNU OpenMP runtime, which the slim image omits.
# Without it the import fails at load with "libgomp.so.1: cannot open shared
# object file" -- at process start, so the container never serves a request.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir "uvicorn[standard]"

# Pre-trained model + held-out demo slice. The container never sees the source
# data and never trains on boot.
COPY artifacts ./artifacts

# Fail fast at build time rather than at deploy time.
RUN python -c "import lightgbm, allocation_agent.api; print('imports ok')"

ENV PORT=7860 ARTIFACTS_DIR=/app/artifacts
# Render injects $PORT; HF and local default to 7860.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn allocation_agent.api:create_app --factory --host 0.0.0.0 --port ${PORT:-7860}"]
