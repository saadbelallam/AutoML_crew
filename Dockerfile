FROM python:3.12-slim

WORKDIR /work

# Install the project (the harness) and its dependencies inside the image
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Run as a normal user, not root
RUN useradd --create-home sandbox
USER sandbox