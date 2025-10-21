#!/bin/bash

# OpenTextShield API - Main Startup Script
# Exit the script if any command fails
set -e

echo "🚀 OpenTextShield API - Starting..."
echo "=================================="
echo ""

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Local environment paths
VENV_PATH="$PROJECT_ROOT/ots"
REQUIREMENTS_PATH="$PROJECT_ROOT/requirements.txt"

# Detect Python version
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "❌ Python 3 not found. Please install Python 3.8+"
    echo "💡 macOS: brew install python3"
    echo "💡 Ubuntu: sudo apt-get install python3"
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

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip

# Try minimal requirements first if available
if [ -f "$PROJECT_ROOT/requirements-minimal.txt" ]; then
    echo "📦 Using minimal requirements (avoids conflicts)..."
    pip install -r "$PROJECT_ROOT/requirements-minimal.txt"
elif [ -f "$REQUIREMENTS_PATH" ]; then
    echo "📦 Using full requirements.txt..."
    pip install -r "$REQUIREMENTS_PATH" || {
        echo "⚠️  Dependency conflicts detected, falling back to minimal install..."
        pip install fastapi uvicorn pydantic pydantic-settings torch transformers numpy requests
    }
else
    echo "📦 Installing essential packages..."
    pip install fastapi uvicorn pydantic pydantic-settings torch transformers numpy requests
fi

# Verify installation
if ! command -v uvicorn &> /dev/null; then
    echo "❌ uvicorn installation failed"
    exit 1
fi

echo "✅ Dependencies ready"
echo ""

# Set working directory to project root
cd "$PROJECT_ROOT"

# Function to start frontend server
start_frontend() {
    if [ -d "$PROJECT_ROOT/frontend" ]; then
        echo "🌐 Starting frontend server..."
        echo "📱 Frontend will be available at: http://localhost:8080"
        cd "$PROJECT_ROOT/frontend"
        $PYTHON_CMD -m http.server 8080 > /dev/null 2>&1 &
        FRONTEND_PID=$!
        cd "$PROJECT_ROOT"
        echo "✅ Frontend server started (PID: $FRONTEND_PID)"
    else
        echo "⚠️  Frontend directory not found, skipping frontend server"
    fi
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
        echo "🌐 Frontend server stopped"
    fi
    echo "✅ Cleanup completed"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start frontend server
start_frontend

echo ""
echo "💡 To test the API, use this curl command:"
echo 'curl -X POST "http://localhost:8002/predict/" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"text\":\"Your SMS content here\",\"model\":\"ots-mbert\"}"'
echo ""

echo "🚀 Starting OpenTextShield API..."
echo "🌟 Both servers will be running:"
echo "📡 API Server: http://localhost:8002"
echo "📚 API Docs: http://localhost:8002/docs"
echo "❤️  Health: http://localhost:8002/health"
echo "🌐 Frontend: http://localhost:8080"
echo ""
echo "🛑 Press Ctrl+C to stop both servers"
echo ""

# Start the API service using Uvicorn
# Set PYTHONPATH to ensure src package is found
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"
python -m uvicorn src.api_interface.main:app --host 0.0.0.0 --port 8002

