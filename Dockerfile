# Use Ubuntu 22.04 LTS as the base image
FROM ubuntu:22.04

# Set the working directory in the container
WORKDIR /app

# Install Python 3.12, build tools, and required dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    build-essential \
    cmake \
    g++ \
    gcc \
    python3-dev \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Create the ots virtual environment using Python's built-in venv tool
RUN python3.12 -m venv /app/ots

# Copy the requirements.txt file
COPY requirements.txt /app/requirements.txt

# Install the dependencies in the ots environment
RUN /app/ots/bin/pip install --no-cache-dir --upgrade pip
RUN /app/ots/bin/pip install --no-cache-dir -r /app/requirements.txt

# Make the start.sh script executable
RUN chmod +x /app/start.sh

# Expose port 8002
EXPOSE 8002

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run the start script using the ots environment
CMD ["bash", "/app/start.sh"]

