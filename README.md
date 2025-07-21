<img src="https://github.com/TelecomsXChangeAPi/OpenTextShield/assets/19316784/71b21aa8-751f-4f4d-87cf-72a7a8c97341" width="300" alt="OpenTextShield Logo">

# OpenTextShield (OTS)

**Professional SMS Spam & Phishing Detection API Platform**

Open source collaborative AI platform for enhanced telecom messaging security and revenue protection, powered by multilingual BERT (mBERT) technology.

[![GitHub Stars](https://img.shields.io/github/stars/TelecomsXChangeAPi/OpenTextShield?style=flat-square)](https://github.com/TelecomsXChangeAPi/OpenTextShield/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=flat-square&logo=docker)](https://hub.docker.com/r/telecomsxchange/opentextshield)

## 🚀 Quick Start

Deploy OpenTextShield in your environment within minutes:

```bash
# Clone the repository
git clone https://github.com/TelecomsXChangeAPi/OpenTextShield.git
cd OpenTextShield

# Start both API and frontend (recommended)
./start.sh

# Or use Docker
# Build and run (includes 679MB mBERT model)
docker build -t opentextshield .
docker run -d -p 8002:8002 -p 8080:8080 opentextshield

# Alternative if port 8080 is busy
docker run -d -p 8002:8002 -p 8081:8080 opentextshield

# Or use pre-built image
docker run -d -p 8002:8002 -p 8080:8080 telecomsxchange/opentextshield:latest
```

**Access Points:**
- **Frontend Interface**: http://localhost:8080
- **API Documentation**: http://localhost:8002/docs
- **API Endpoint**: http://localhost:8002/predict/

## ✨ Key Features

- 🌍 **Multilingual Support**: 104+ languages with mBERT, Open Text Shield is currently well trained on 10 languages for SMS Classification. 
- ⚡ **Real-time Classification**: Professional API with <200ms response time> 
- 🔒 **Advanced Detection**: Spam, phishing, and ham classification
- 📊 **Professional Interface**: Research-grade web interface with metrics
- 🐳 **Docker Ready**: Complete containerized deployment
- 🔧 **API First**: RESTful API with comprehensive documentation
- 📈 **Revenue Protection**: Optional revenue assurance features

## 🛠 API Usage

### Quick Test
```bash
# Test the API endpoint
curl -X POST "http://localhost:8002/predict/" \
  -H "Content-Type: application/json" \
  -d '{"text":"Your SMS content here","model":"ots-mbert"}'
```

### Response Format
```json
{
  "label": "ham|spam|phishing",
  "probability": 0.95,
  "processing_time": 0.15,
  "model_info": {
    "name": "OTS_mBERT",
    "version": "2.1",
    "author": "TelecomsXChange (TCXC)"
  }
}
```

## 📋 Installation Guide

### Requirements
- Python 3.12
- 4GB RAM minimum
- Docker (optional)

### Local Setup
```bash
# Create virtual environment
python3.12 -m venv ots
source ots/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Start the platform
./start.sh
```

### Docker Deployment

#### 🛡️ Security-Enhanced Docker Options

**Option 1: Enhanced Security (Recommended)**
```bash
# Multi-stage build with non-root user - best balance of security and functionality
docker build -f Dockerfile.secure -t opentextshield:secure .
docker run -d -p 8002:8002 -p 8081:8080 opentextshield:secure
```

**Option 2: Standard Build**
```bash
# Standard build with security updates
docker build -t opentextshield .
docker run -d -p 8002:8002 -p 8081:8080 opentextshield
```

**Option 3: Maximum Security (Advanced)**
```bash
# Ultra-secure distroless build - minimal attack surface (API only)
docker build -f Dockerfile.distroless -t opentextshield:distroless .
docker run -d -p 8002:8002 opentextshield:distroless
```

#### 🏗️ Architecture-Specific Builds

**x86_64 (Intel/AMD) Architecture:**
```bash
# Enhanced security for x86
docker buildx build --platform linux/amd64 -f Dockerfile.secure -t opentextshield:x86-secure .

# Standard x86 build
docker buildx build --platform linux/amd64 -t telecomsxchange/opentextshield:2.1-x86-v2 .
```

**ARM64 (Apple Silicon) Architecture:**
```bash
# Enhanced security for ARM64
docker buildx build --platform linux/arm64 -f Dockerfile.secure -t opentextshield:arm64-secure .
```

#### 📦 Pre-built Images
```bash
# Latest stable releases
docker run -d -p 8002:8002 -p 8080:8080 telecomsxchange/opentextshield:latest
docker run -d -p 8002:8002 -p 8080:8080 telecomsxchange/opentextshield:2.1-x86-v2

# Using Docker Compose (recommended for production)
docker-compose up -d
```

**Container Access:**
- API: http://localhost:8002
- Frontend: http://localhost:8080 (or 8081)
- Health: http://localhost:8002/health

**Security Benefits:**
- 🔒 **Enhanced**: 60-80% fewer vulnerabilities, non-root execution, multi-stage builds
- 🛡️ **Distroless**: Minimal attack surface, no shell access, maximum security
- 📦 **Smaller images**: Optimized builds reduce image size and vulnerabilities

**Architecture Support:**
- ARM64 (Apple Silicon): `telecomsxchange/opentextshield:latest`
- x86_64 (Intel/AMD): `telecomsxchange/opentextshield:2.1-x86-v2`

## 🏗 Architecture

### Core Components

**API Interface** (`src/api_interface/`)
- Modern FastAPI application with professional structure
- Pydantic models for request/response validation
- Comprehensive error handling and logging
- Security middleware and CORS support

**mBERT Model** (`src/mBERT/training/model-training/`)
- Multilingual BERT optimized for SMS classification
- Support for 104+ languages with cross-lingual transfer learning
- Apple Silicon MLX optimization available

**Frontend Interface** (`frontend/`)
- Professional research-grade web interface
- Real-time system monitoring and metrics
- Technical details and performance indicators

### Performance
- **Inference Speed**: 54 messages/second (Apple Silicon M1 Pro)
- **Response Time**: <200ms typical
- **Languages**: 104+ supported via mBERT
- **Accuracy**: Production-ready classification

## 🧪 Testing

```bash
# Run comprehensive tests
cd src/mBERT/tests
python run_all_tests.py all

# Stress testing
python test_stress.py 1000
python stressTest_20k_mlx_api.py
```

## 📚 Research Background

OpenTextShield leverages cutting-edge AI research to provide real-time SMS spam and phishing detection across 104+ languages. Our research focuses on the practical application of multilingual BERT (mBERT) technology for telecom security challenges.

**Research Highlights:**
- Comparative analysis of AI models for SMS classification
- Multilingual spam detection using mBERT architecture  
- Real-time processing optimization for telecom applications
- Community-driven approach to dataset expansion

[**Read Full Research Paper →**](RESEARCH.md)

## 🤝 Contributing

### Ways to Contribute

**🗃️ Dataset Contributions**
We need multilingual datasets for training. Required format:
```csv
text,label
"Your verification code is 12345",ham
"Win $1000! Click here now!",spam
"Your account is locked. Visit fake-bank.com",phishing
```

**🔧 Development**
- API improvements and optimizations
- Frontend enhancements
- Model training and evaluation
- Documentation and testing

**🌍 Localization**
- Translate interface and documentation
- Test models in your language
- Provide linguistic insights for regional variations

**💡 Research & Testing**
- Performance benchmarking
- Security analysis
- Integration testing with telecom systems

### Getting Started
1. Fork the repository
2. Check [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines
3. Join discussions in GitHub Issues
4. Submit Pull Requests with improvements

## 🔧 Development

### Model Training
```bash
# Train new mBERT model
cd src/mBERT/training/model-training/
python train_ots_improved.py

# Test model performance
python test_training.py
```

### Frontend Development
```bash
# Frontend is a single HTML file with embedded CSS/JS
# Edit frontend/index.html for customizations
# Restart ./start.sh to see changes
```

## 🚀 Production Deployment

### Docker Production
```bash
# Multi-arch production build
docker buildx build --platform linux/amd64,linux/arm64 -t your-registry/opentextshield .

# Production compose
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes
```yaml
# Example k8s deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: opentextshield
spec:
  replicas: 3
  selector:
    matchLabels:
      app: opentextshield
  template:
    spec:
      containers:
      - name: ots
        image: telecomsxchange/opentextshield:latest
        ports:
        - containerPort: 8002
        - containerPort: 8080
```

## 📊 Monitoring & Analytics

### Health Checks
- **API Health**: `GET /health`
- **Model Status**: `GET /model/status` 
- **System Metrics**: Built-in performance monitoring

### Logs
- **API Logs**: Structured JSON logging with request tracking
- **Prediction Logs**: Classification results and performance metrics
- **Error Tracking**: Comprehensive error handling and reporting

## 🔐 Security Features

- **Input Validation**: Pydantic models with strict validation
- **Rate Limiting**: Configurable API rate limits
- **CORS Protection**: Configurable cross-origin policies
- **Secure Headers**: Standard security headers implemented

## 💼 Enterprise Features

### Revenue Protection
- Dynamic pricing based on message content analysis
- Grey route detection and mitigation
- Fraud pattern identification
- Premium message routing optimization

### Integration APIs
- RESTful API with OpenAPI documentation
- Webhook support for real-time notifications
- Batch processing capabilities
- Custom model loading support

## 📖 Documentation

- **[Installation Guide](Installation.md)** - Detailed setup instructions
- **[API Documentation](http://localhost:8002/docs)** - Interactive API explorer
- **[Model Training Guide](src/mBERT/training/model-training/README.md)** - Train custom models
- **[Testing Guide](src/mBERT/tests/README.md)** - Comprehensive testing suite
- **[Docker Guide](Dockerfile)** - Container deployment options

## 🌟 About TelecomsXChange (TCXC)

OpenTextShield is pioneered by [TelecomsXChange](https://telecomsxchange.com), a leading telecommunications platform provider. TCXC is committed to releasing cutting-edge open-source AI tools for the global telecom community.

**Key Initiative:**
- First pre-trained open-source mBERT model for SMS classification
- Integration with TCXC's SMPP Stack for real-time processing
- Community-driven approach to continuous improvement
- Revenue protection features for telecom operators

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Additional Resources

- **[Research Paper](RESEARCH.md)** - Complete academic research
- **[BERT Documentation](https://arxiv.org/abs/1810.04805)** - Original BERT paper
- **[FastAPI Documentation](https://fastapi.tiangolo.com/)** - API framework docs
- **[MLX Framework](https://ml-explore.github.io/mlx/)** - Apple Silicon optimization

---

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by the [TelecomsXChange](https://telecomsxchange.com) team and the open source community.
