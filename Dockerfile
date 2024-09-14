# Use Ubuntu 24.04 as the base image
FROM ubuntu:24.04

# Create necessary directories
RUN mkdir -p /home/ots/OpenTextShield

# Set the working directory in the container to match the host structure
WORKDIR /home/ots/OpenTextShield

# Copy the installed_packages.txt file from the host to the container
COPY installed_packages.txt /home/ots/OpenTextShield/installed_packages.txt

# Update the package lists, install apt-utils and system-level packages
RUN apt-get update && \
    apt-get install -y apt-utils dselect build-essential software-properties-common cmake && \
    dpkg --set-selections < /home/ots/OpenTextShield/installed_packages.txt && \
    apt-get dselect-upgrade -y && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file from the host to the container
COPY requirements.txt /home/ots/OpenTextShield/requirements.txt

# Install Python 3.12 and create a virtual environment
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Install necessary libraries for fasttext (C++17 support)
RUN apt-get update && \
    apt-get install -y g++ libomp-dev && \
    rm -rf /var/lib/apt/lists/*

# Create the Python virtual environment using Python 3.12
RUN python3.12 -m venv /home/ots/OpenTextShield/ots

# Install Python dependencies from requirements.txt
RUN /home/ots/OpenTextShield/ots/bin/pip install --no-cache-dir --upgrade pip
RUN /home/ots/OpenTextShield/ots/bin/pip install --no-cache-dir -r /home/ots/OpenTextShield/requirements.txt

# Ensure uvicorn is installed
RUN /home/ots/OpenTextShield/ots/bin/pip install --no-cache-dir uvicorn

# Copy the rest of the app's code into the container, preserving the directory structure
COPY . /home/ots/OpenTextShield

# Make the start.sh script executable
RUN chmod +x /home/ots/OpenTextShield/start.sh

# Expose port 8002
EXPOSE 8002

# Set environment variable
ENV PYTHONUNBUFFERED=1

# Run the start script using the ots environment
CMD ["bash", "/home/ots/OpenTextShield/start.sh"]

