# 🚀 OpenTextShield API Startup Guide

This guide provides multiple ways to start the OpenTextShield API, supporting both Docker and local development environments.

## 📋 Quick Start Options

### Option 1: Auto-Detection (Recommended)
```bash
./scripts/start.sh
```
**Best for:** Most users - automatically detects your environment and configures accordingly.

### Option 2: Local Development
```bash
./scripts/start-local.sh
```
**Best for:** Local development with hot reload and development features.

### Option 3: Manual Start
```bash
./scripts/start-manual.sh
```
**Best for:** Users who already have dependencies installed and want direct control.

### Option 4: Docker
```bash
# Build locally (includes 679MB mBERT model)
docker build -t opentextshield .
docker run -d -p 8002:8002 -p 8080:8080 opentextshield

# Alternative if port 8080 is in use
docker run -d -p 8002:8002 -p 8081:8080 opentextshield

# Using pre-built image
docker pull telecomsxchange/opentextshield:latest
docker run -d -p 8002:8002 -p 8080:8080 telecomsxchange/opentextshield:latest
```
**Access:** API at http://localhost:8002, Frontend at http://localhost:8080 (or 8081)

## 🔧 Prerequisites

### For Local Development:
- **Python:** 3.8+ (3.10+ recommended)
- **Operating System:** macOS, Linux, or Windows WSL
- **Memory:** 2GB+ RAM recommended
- **Storage:** 1GB+ free space

### For Docker:
- **Docker:** Version 20.0+
- **Memory:** 4GB+ RAM recommended
- **Storage:** 2GB+ free space

## 📊 What Each Script Does

| Script | Environment | Virtual Env | Dependencies | Hot Reload | Best For |
|--------|-------------|-------------|--------------|------------|----------|
| `scripts/start.sh` | Auto-detect | ✅ Creates | ✅ Installs | ✅ Yes | General use |
| `scripts/start-local.sh` | Local only | ✅ Creates | ✅ Installs | ✅ Yes | Development |
| `scripts/start-manual.sh` | Any | ❌ Uses current | ❌ Must exist | ✅ Yes | Advanced users |
| Docker | Container | ✅ Built-in | ✅ Pre-installed | ❌ No | Production |

## 🖥️ Platform-Specific Instructions

### macOS
```bash
# Install Python if needed
brew install python3

# Start the API
./scripts/start.sh
```

### Ubuntu/Debian
```bash
# Install Python if needed
sudo apt-get update
sudo apt-get install python3 python3-venv python3-pip

# Start the API
./scripts/start.sh
```

### Windows (WSL)
```bash
# In WSL terminal
sudo apt-get update
sudo apt-get install python3 python3-venv python3-pip

# Start the API
./scripts/start.sh
```

## 🔍 Verification

Once started, verify the API is working:

### Health Check
```bash
curl http://localhost:8002/health
```

### API Documentation
Open in browser: http://localhost:8002/docs

### Test Prediction
```bash
curl -X POST "http://localhost:8002/predict/" \
  -H "Content-Type: application/json" \
  -d '{"text":"Free money! Click here!","model":"bert"}'
```

## ⚙️ Configuration

### Environment Variables
```bash
# API Configuration
export OTS_API_HOST=0.0.0.0
export OTS_API_PORT=8002
export OTS_LOG_LEVEL=INFO

# Security
export OTS_ALLOWED_IPS="127.0.0.1,10.0.0.0/8"
export OTS_CORS_ORIGINS="https://yourdomain.com"

# Models
export OTS_DEFAULT_MODEL=bert
export OTS_MAX_TEXT_LENGTH=512

# Then start
./scripts/start.sh
```

### Configuration File (.env)
Create a `.env` file in the project root:
```env
OTS_API_PORT=8003
OTS_LOG_LEVEL=DEBUG
OTS_ALLOWED_IPS=ANY
```

## 🛠️ Troubleshooting

### Common Issues

**❌ "Python 3 not found"**
```bash
# macOS
brew install python3

# Ubuntu
sudo apt-get install python3 python3-pip

# Check version
python3 --version
```

**❌ "Virtual environment creation failed"**
```bash
# Install venv module
sudo apt-get install python3-venv  # Ubuntu
# or
python3 -m pip install virtualenv  # Any platform
```

**❌ "Port 8002 already in use"**
```bash
# Find what's using the port
lsof -i :8002

# Kill the process (replace PID)
kill <PID>

# Or use different port
export OTS_API_PORT=8003
./scripts/start.sh
```

**❌ "uvicorn not found"**
```bash
# Install manually
pip install uvicorn

# Or reinstall dependencies
rm -rf ots/  # Remove virtual env
./scripts/start.sh   # Recreate and install
```

### Debug Mode
```bash
# Enable debug logging
export OTS_LOG_LEVEL=DEBUG
./scripts/start.sh
```

### Reset Environment
```bash
# Remove virtual environment and start fresh
rm -rf ots/
./scripts/start.sh
```

## 🐳 Docker-Specific

### Build Custom Image
```bash
# Build with custom tag
docker build -t my-opentextshield:latest .

# Run with custom settings
docker run -d \
  -p 8003:8002 \
  -e OTS_LOG_LEVEL=DEBUG \
  my-opentextshield:latest
```

### Docker Compose
Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  opentextshield:
    image: telecomsxchange/opentextshield:latest
    ports:
      - "8002:8002"
    environment:
      - OTS_LOG_LEVEL=INFO
      - OTS_ALLOWED_IPS=ANY
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

## 🎯 Production Deployment

### Recommended Production Setup
```bash
# Set production environment
export OTS_LOG_LEVEL=WARNING
export OTS_ALLOWED_IPS="10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
export OTS_CORS_ORIGINS="https://yourdomain.com"

# Start with production script
./scripts/start.sh
```

### Behind Reverse Proxy (nginx)
```nginx
server {
    listen 80;
    server_name your-api-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8002;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📞 Support

If you encounter issues:

1. **Check the logs:** Look for error messages in the terminal output
2. **Verify dependencies:** Ensure Python 3.8+ is installed
3. **Test basic functionality:** Try the health check endpoint
4. **Reset environment:** Remove `ots/` directory and restart
5. **Report issues:** Open an issue on GitHub with full error output

## 🔗 Quick Links

- **API Documentation:** http://localhost:8002/docs
- **Health Check:** http://localhost:8002/health
- **Repository:** https://github.com/TelecomsXChangeAPi/OpenTextShield
- **Issues:** https://github.com/TelecomsXChangeAPi/OpenTextShield/issues