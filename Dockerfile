FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .
RUN mkdir -p data/experiments
RUN useradd -m -r appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000

# Note: Ollama must be accessible from the container.
# Set OLLAMA_BASE_URL=http://host.docker.internal:11434 when using Docker Desktop.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
