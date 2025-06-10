FROM python:3.11-slim

# Install curl, build tools, and libpq
RUN apt-get update \
  && apt-get install -y --no-install-recommends curl build-essential libpq-dev \
  && rm -rf /var/lib/apt/lists/*

# Install uv (the ultra-fast Python package manager)
ENV PATH="/root/.local/bin:${PATH}"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# IMPORTANT: Copy all application source code, including pyproject.toml and uv.lock,
# before installing dependencies in editable mode.
# This ensures that 'source/' directory and 'README.md' are present.
COPY . .

# Install dependencies (using either --system or uv venv, as discussed previously)
# Option 1: Using --system (as per your last attempt)
RUN uv pip install -e . --system

# Option 2: Using a virtual environment (more robust for production)
# RUN uv venv
# RUN uv pip install -e .


# Optional: switch to a non-root user
# RUN useradd -m appuser
# USER appuser

# Expose application port
EXPOSE 8000

# Run the server
CMD ["uv", "run", "source/yf_server.py", "--host", "0.0.0.0", "--port", "8000"]