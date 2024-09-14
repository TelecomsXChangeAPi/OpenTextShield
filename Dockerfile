# Use Ubuntu as the base image
FROM ubuntu:22.04

# Set the working directory in the container
WORKDIR /app

# Copy the installed_packages.txt file from the host to the container
COPY installed_packages.txt /app/installed_packages.txt

# Update the package lists and install system-level packages from the host
RUN apt-get update && \
    apt-get install -y dselect && \
    dpkg --set-selections < /app/installed_packages.txt && \
    apt-get dselect-upgrade -y && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file from the host
COPY requirements.txt /app/requirements.txt

# Install Python 3.12 and create a virtual environment
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Create the Python virtual environment using Python 3.12
RUN python3.12 -m venv /app/ots

# Install Python dependencies from requirements.txt
RUN /app/ots/bin/pip install --no-cache-dir --upgrade pip
RUN /app/ots/bin/pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the app's code into the container
COPY . /app

# Make the start.sh script executable
RUN chmod +x /app/start.sh

# Expose port 8002
EXPOSE 8002

# Set environment variable
ENV PYTHONUNBUFFERED=1

# Run the start script using the ots environment
CMD ["bash", "/app/start.sh"]

