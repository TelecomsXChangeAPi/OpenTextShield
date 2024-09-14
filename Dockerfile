# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install required build tools (including a C++ compiler) and dependencies
RUN apt-get update && apt-get install -y \
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
RUN python -m venv /app/ots

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

