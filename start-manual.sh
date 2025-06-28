#!/bin/bash

# OpenTextShield API - Manual/Direct Startup Script
# For users who want to start the API directly without virtual environment management

echo "⚡ OpenTextShield API - Manual Start"
echo "===================================="
echo ""

# Check if we're in the right directory
if [ ! -f "src/api_interface/main.py" ]; then
    echo "❌ Please run this script from the OpenTextShield project root directory"
    echo "💡 Expected file: src/api_interface/main.py"
    exit 1
fi

# Check Python availability
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check required packages
echo "🔍 Checking required packages..."
python3 -c "
import sys
missing = []
try:
    import fastapi
    print('✅ FastAPI found')
except ImportError:
    missing.append('fastapi')

try:
    import uvicorn
    print('✅ Uvicorn found')
except ImportError:
    missing.append('uvicorn')

try:
    import pydantic
    print('✅ Pydantic found')
except ImportError:
    missing.append('pydantic')

if missing:
    print(f'❌ Missing packages: {missing}')
    print('📦 Install with: pip install ' + ' '.join(missing))
    sys.exit(1)
else:
    print('✅ All required packages found')
" || exit 1

echo ""
echo "🚀 Starting OpenTextShield API..."
echo "📡 Server: http://localhost:8002"
echo "📚 Docs: http://localhost:8002/docs"
echo "❤️  Health: http://localhost:8002/health"
echo ""
echo "🛑 Press Ctrl+C to stop"
echo ""

# Start the server directly
python3 -m uvicorn src.api_interface.main:app --host 0.0.0.0 --port 8002 --reload