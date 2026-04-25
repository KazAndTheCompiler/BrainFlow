# BrainFlow Docker Image
# Multi-stage build for production deployment

FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.12-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd -m -u 1000 brainflow && \
  mkdir -p /app/brain_state && \
  chown -R brainflow:brainflow /app

USER brainflow
WORKDIR /app

# Copy application code
COPY --chown=brainflow:brainflow brain/ ./brain/
COPY --chown=brainflow:brainflow sensory/ ./sensory/
COPY --chown=brainflow:brainflow dashboard/ ./dashboard/
COPY --chown=brainflow:brainflow run.py .

# Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  NEUROLINKED_HOST=0.0.0.0 \
  NEUROLINKED_PORT=8000 \
  NEUROLINKED_REQUIRE_AUTH=true \
  NEUROLINKED_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "run.py"]

# Development stage
FROM production as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-cov black flake8 mypy

USER brainflow

# Development settings
ENV NEUROLINKED_REQUIRE_AUTH=false \
  NEUROLINKED_ENV=development

CMD ["python", "run.py"]
