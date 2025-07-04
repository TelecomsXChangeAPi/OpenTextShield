# Use Ubuntu 24.04 LTS with latest security patches
FROM ubuntu:24.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Create necessary directories
RUN mkdir -p /home/ots/OpenTextShield

# Set the working directory
WORKDIR /home/ots/OpenTextShield

# Install system dependencies with security updates
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

# Copy requirements first for better Docker layer caching
COPY requirements-minimal.txt /home/ots/OpenTextShield/requirements-minimal.txt

# Create the Python virtual environment
RUN python3.12 -m venv /home/ots/OpenTextShield/ots

# Install Python dependencies using minimal requirements for faster builds
RUN /home/ots/OpenTextShield/ots/bin/pip install --no-cache-dir --upgrade pip
RUN /home/ots/OpenTextShield/ots/bin/pip install --no-cache-dir -r /home/ots/OpenTextShield/requirements-minimal.txt

# Copy the rest of the app's code into the container
COPY . /home/ots/OpenTextShield

# Make start scripts executable
RUN chmod +x /home/ots/OpenTextShield/start.sh
RUN chmod +x /home/ots/OpenTextShield/start-local.sh

# Expose both API and frontend ports
EXPOSE 8002 8080

# Health check - reduced start period for faster readiness detection
HEALTHCHECK --interval=15s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the start script
CMD ["bash", "/home/ots/OpenTextShield/start.sh"]

