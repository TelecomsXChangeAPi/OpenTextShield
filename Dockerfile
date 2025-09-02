# Multi-stage build for enhanced security
# Stage 1: Build dependencies
FROM ubuntu:24.04 AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies with security updates
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    build-essential \
    g++ \
    libomp-dev \
    curl \
    ca-certificates \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements-security.txt /tmp/requirements-security.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements-security.txt

# Stage 2: Runtime image
FROM ubuntu:24.04 AS runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:$PATH"

# Install only runtime dependencies with security updates
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    python3.12 \
    python3.12-venv \
    curl \
    ca-certificates \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

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
RUN chmod +x /home/ots/OpenTextShield/start.sh && \
    chmod +x /home/ots/OpenTextShield/start-local.sh

# Switch to non-root user
USER ots

# Expose ports
EXPOSE 8002 8080

# Health check with improved security
HEALTHCHECK --interval=15s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the application
CMD ["bash", "/home/ots/OpenTextShield/start.sh"]