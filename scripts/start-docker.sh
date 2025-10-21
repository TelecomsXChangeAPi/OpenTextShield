#!/bin/bash

# OpenTextShield API - Docker Environment Startup Script
# This script is specifically for Docker containers

set -e

echo "🐳 OpenTextShield API - Docker Mode"
echo "==================================="
echo ""

# Docker environment paths
VENV_PATH="/home/ots/OpenTextShield/ots"
REQUIREMENTS_PATH="/home/ots/OpenTextShield/requirements.txt"

# Detect Python version
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "❌ Python 3 not found in Docker container"
    exit 1
fi

echo "🐍 Using Python: $PYTHON_CMD ($($PYTHON_CMD --version))"

# Setup virtual environment
if [ ! -d "$VENV_PATH" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_PATH"
    echo "✅ Virtual environment created"
fi

echo "🔄 Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Install dependencies
if [ -f "$REQUIREMENTS_PATH" ]; then
    echo "📥 Installing dependencies..."
    pip install --upgrade pip
    pip install -r "$REQUIREMENTS_PATH"
    echo "✅ Dependencies installed"
else
    echo "❌ requirements.txt not found at: $REQUIREMENTS_PATH"
    exit 1
fi

# Set working directory
cd "/home/ots/OpenTextShield"

echo ""
echo "🚀 Starting OpenTextShield API (Docker)..."
echo "📡 Server: http://0.0.0.0:8002"
echo ""

# Start server for production (no reload in Docker)
uvicorn src.api_interface.main:app --host 0.0.0.0 --port 8002 --log-level info