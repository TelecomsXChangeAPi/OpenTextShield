#!/bin/bash

# OpenTextShield API - Docker Environment Startup Script
# This script uses the pre-installed /opt/venv from the Docker build stage.
# It does NOT install dependencies at runtime — they are baked into the image.

set -e

echo "🐳 OpenTextShield API - Docker Mode"
echo "==================================="
echo ""

# Use the pre-built virtual environment from the Docker image
VENV_PATH="/opt/venv"
PROJECT_ROOT="/home/ots/OpenTextShield"

if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "   This script is for Docker containers with pre-built dependencies."
    echo "   For local development, use: ./scripts/start.sh"
    exit 1
fi

echo "🔄 Activating pre-built virtual environment..."
source "$VENV_PATH/bin/activate"

# Verify critical packages are available
python -c "import fastapi; import torch; import transformers; print('✅ Core packages verified')" || {
    echo "❌ Required packages missing from virtual environment"
    exit 1
}

# Set working directory
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Start frontend server if available
if [ -d "$PROJECT_ROOT/frontend" ]; then
    echo "🌐 Starting frontend server..."
    cd "$PROJECT_ROOT/frontend"
    python -m http.server 8080 > /dev/null 2>&1 &
    FRONTEND_PID=$!
    cd "$PROJECT_ROOT"
    echo "✅ Frontend server started (PID: $FRONTEND_PID)"
    echo "📱 Frontend: http://localhost:8080"
fi

# Cleanup on exit
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
trap cleanup SIGINT SIGTERM

echo ""
echo "🚀 Starting OpenTextShield API..."
echo "📡 API Server: http://0.0.0.0:8002"
echo "📚 API Docs: http://0.0.0.0:8002/docs"
echo "❤️  Health: http://0.0.0.0:8002/health"
echo ""

# Start the API service
exec python -m uvicorn src.api_interface.main:app --host 0.0.0.0 --port 8002 --log-level info
