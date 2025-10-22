# Multi-stage build for enhanced security
# Stage 1: Build dependencies
FROM python:3.12-slim-bookworm AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies (using pre-cached layers)
# build-essential and g++ already included in python:3.12 base image
# Just ensure pip is up to date

# Create virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements-security.txt /tmp/requirements-security.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements-security.txt

# Stage 2: Runtime image
FROM python:3.12-slim-bookworm AS runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:$PATH"

# Runtime environment already includes curl and ca-certificates
# No additional system packages needed

# Create non-root user for security
RUN groupadd -r ots && useradd -r -g ots -d /home/ots -s /bin/bash ots && \
    mkdir -p /home/ots/OpenTextShield && \
    chown -R ots:ots /home/ots

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /home/ots/OpenTextShield

# Copy application code
COPY --chown=ots:ots . /home/ots/OpenTextShield

# Make start scripts executable
RUN chmod +x /home/ots/OpenTextShield/scripts/start.sh && \
    chmod +x /home/ots/OpenTextShield/scripts/start-local.sh

# Switch to non-root user
USER ots

# Expose ports
EXPOSE 8002 8080

# Health check with improved security
HEALTHCHECK --interval=15s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the application
CMD ["bash", "/home/ots/OpenTextShield/scripts/start.sh"]