#!/bin/bash

# Test script for OpenTextShield frontend setup
set -e

# Ensure we're running from the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "📂 Working directory: $PROJECT_ROOT"

echo "🧪 Testing OpenTextShield Frontend Setup"
echo "========================================"
echo ""

# Check if frontend directory exists
if [ -d "frontend" ]; then
    echo "✅ Frontend directory exists"
else
    echo "❌ Frontend directory missing"
    exit 1
fi

# Check if frontend HTML file exists
if [ -f "frontend/index.html" ]; then
    echo "✅ Frontend HTML file exists"
else
    echo "❌ Frontend HTML file missing"
    exit 1
fi

# Check if start.sh is executable
if [ -x "start.sh" ]; then
    echo "✅ start.sh is executable"
else
    echo "❌ start.sh is not executable"
    exit 1
fi

# Check if frontend HTML contains required elements
if grep -q "OpenTextShield Frontend" frontend/index.html; then
    echo "✅ Frontend contains proper branding"
else
    echo "❌ Frontend missing proper branding"
    exit 1
fi

if grep -q "AI-Powered Text Security Analysis Interface" frontend/index.html; then
    echo "✅ Frontend contains professional subtitle"
else
    echo "❌ Frontend missing professional subtitle"
    exit 1
fi

if grep -q "localhost:8002" frontend/index.html; then
    echo "✅ Frontend configured for correct API endpoint"
else
    echo "❌ Frontend has incorrect API endpoint"
    exit 1
fi

# Check Docker configuration
if grep -q "8080:8080" docker-compose.yml; then
    echo "✅ Docker Compose configured for frontend port"
else
    echo "❌ Docker Compose missing frontend port mapping"
    exit 1
fi

if grep -q "EXPOSE.*8080" Dockerfile; then
    echo "✅ Dockerfile exposes frontend port"
else
    echo "❌ Dockerfile missing frontend port exposure"
    exit 1
fi

# Check CLAUDE.md documentation
if grep -q "Frontend Interface" CLAUDE.md; then
    echo "✅ Documentation updated with frontend information"
else
    echo "❌ Documentation missing frontend information"
    exit 1
fi

echo ""
# Check that old model directories are removed
if [ -d "src/FastText" ]; then
    echo "❌ FastText directory still exists"
    exit 1
else
    echo "✅ FastText directory successfully removed"
fi

if [ -d "src/BERT" ]; then
    echo "❌ BERT directory still exists"
    exit 1
else
    echo "✅ BERT directory successfully removed"
fi

# Check that only mBERT remains
if [ -d "src/mBERT" ]; then
    echo "✅ mBERT directory exists"
else
    echo "❌ mBERT directory missing"
    exit 1
fi

# Check frontend uses OTS-branded mbert
if grep -q "ots-mbert" frontend/index.html; then
    echo "✅ Frontend configured for OTS-mBERT"
else
    echo "❌ Frontend missing OTS-mBERT configuration"
    exit 1
fi

if grep -q "OpenTextShield mBERT" frontend/index.html; then
    echo "✅ Frontend shows OpenTextShield branding"
else
    echo "❌ Frontend missing OpenTextShield branding in model display"
    exit 1
fi

if grep -E "^fasttext" requirements.txt requirements-minimal.txt 2>/dev/null; then
    echo "❌ Requirements files still contain active FastText"
    exit 1
else
    echo "✅ Requirements files cleaned of active FastText"
fi

echo "🎉 All tests passed! OTS-mBERT system ready with OpenTextShield branding!"
echo ""
echo "To start the application:"
echo "  ./start.sh"
echo ""
echo "Access points:"
echo "  🌐 Frontend: http://localhost:8080"
echo "  📡 API: http://localhost:8002"
echo "  📚 API Docs: http://localhost:8002/docs"