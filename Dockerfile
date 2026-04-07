# ============================================================
# Stage 1: Build — install dependencies with build tools
# ============================================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends g++ build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first for better layer caching.
# Source code changes won't invalidate the dependency layer.
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ src/
COPY config.json ./

# ============================================================
# Stage 2: Runtime — minimal image without build tools
# ============================================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        procps && \
    rm -rf /var/lib/apt/lists/*

# Run as non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed venv and source from builder
COPY --from=builder /app /app

# Ensure data directories exist and are writable
RUN mkdir -p /app/data /app/logs && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 1995

CMD ["uv", "run", "python", "src/run.py"]
