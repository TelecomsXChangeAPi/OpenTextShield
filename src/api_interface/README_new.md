# OpenTextShield API Interface

Professional SMS spam and phishing detection API built with FastAPI, supporting multiple AI models including BERT, mBERT, and FastText.

## 🚀 Features

- **Multi-Model Support**: BERT, mBERT (multilingual), and FastText models
- **Professional Architecture**: Modular design with proper separation of concerns
- **Comprehensive Error Handling**: Detailed error responses with proper HTTP status codes
- **Security**: IP-based access control and CORS support
- **Logging**: Structured logging for monitoring and debugging
- **Configuration Management**: Environment-based configuration
- **Health Checks**: Built-in health and status endpoints
- **Feedback System**: User feedback collection and CSV export
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation

## 📁 Project Structure

```
src/api-interface/
├── main.py                    # FastAPI application entry point
├── config/
│   └── settings.py           # Configuration management
├── models/
│   ├── request_models.py     # Pydantic request models
│   └── response_models.py    # Pydantic response models
├── services/
│   ├── model_loader.py       # ML model management
│   ├── prediction_service.py # Business logic for predictions
│   └── feedback_service.py   # Feedback handling
├── middleware/
│   └── security.py          # Security middleware
├── utils/
│   ├── logging.py           # Logging configuration
│   └── exceptions.py        # Custom exceptions
└── routers/
    ├── health.py            # Health check endpoints
    ├── prediction.py        # Prediction endpoints
    └── feedback.py          # Feedback endpoints
```

## 🛠 Installation & Setup

### Prerequisites
- Python 3.12+
- PyTorch
- FastAPI and dependencies (see requirements.txt)

### Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Start the API server**:
```bash
# Using the start script (recommended)
./start.sh

# Or manually
uvicorn src.api-interface.main:app --host 0.0.0.0 --port 8002
```

3. **Access the API**:
- API: http://localhost:8002
- Documentation: http://localhost:8002/docs
- Health Check: http://localhost:8002/health

## 🔧 Configuration

Configure the API using environment variables with the `OTS_` prefix:

```bash
# API Configuration
OTS_API_HOST=0.0.0.0
OTS_API_PORT=8002
OTS_LOG_LEVEL=INFO

# Security
OTS_ALLOWED_IPS=ANY,127.0.0.1,localhost
OTS_CORS_ORIGINS=*

# Models
OTS_DEFAULT_MODEL=bert
OTS_DEFAULT_BERT_VERSION=bert-base-multilingual-cased
```

Or create a `.env` file in the project root.

## 📡 API Endpoints

### Prediction
**POST** `/predict/`

Classify text as ham, spam, or phishing.

```json
{
  "text": "Congratulations! You've won $1000. Click here to claim!",
  "model": "bert",
  "bert_version": "bert-base-multilingual-cased"
}
```

**Response:**
```json
{
  "label": "spam",
  "probability": 0.95,
  "processing_time": 0.15,
  "model_info": {
    "name": "OTS_mBERT",
    "version": "bert-base-multilingual-cased",
    "author": "TelecomsXChange (TCXC)",
    "last_training": "2024-03-20"
  }
}
```

### Health Check
**GET** `/health`

Check API and model status.

### Feedback
**POST** `/feedback/`

Submit feedback about classification results.

**GET** `/feedback/download/{model_name}`

Download feedback CSV file.

## 🧪 Testing

### Basic API Test
```bash
curl -X POST "http://localhost:8002/predict/" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Free money! Click here now!",
    "model": "bert",
    "bert_version": "bert-base-multilingual-cased"
  }'
```

### Health Check Test
```bash
curl http://localhost:8002/health
```

## 📊 Supported Models

### BERT Models
- `bert-base-multilingual-cased`: Multilingual BERT supporting 104+ languages

### FastText
- Lightweight model for fast inference

### Model Selection
```json
{
  "model": "bert",  // or "fasttext"
  "bert_version": "bert-base-multilingual-cased"  // only for BERT
}
```

## 🔒 Security Features

- **IP Whitelisting**: Configurable IP access control
- **CORS**: Cross-Origin Resource Sharing support
- **Input Validation**: Pydantic-based request validation
- **Error Handling**: Secure error responses without sensitive data exposure

## 📝 Logging

Structured logging with configurable levels:
- Request/response logging
- Model loading status
- Error tracking
- Performance metrics

## 🔄 Migration from Legacy API

### Key Changes
1. **New entry point**: Use `main.py` instead of `api_2_mBERT.py`
2. **Modular structure**: Code separated into logical modules
3. **Enhanced error handling**: Detailed error responses
4. **Configuration management**: Environment-based settings
5. **Improved logging**: Structured logging system

### Backward Compatibility
The API endpoints remain compatible with existing clients. Update your startup command:

```bash
# Old
uvicorn src.api-interface.api_2_mBERT:app --host 0.0.0.0 --port 8002

# New
uvicorn src.api-interface.main:app --host 0.0.0.0 --port 8002
```

## 🚀 Production Deployment

### Docker
```bash
# Build
docker build -t opentextshield .

# Run
docker run -d -p 8002:8002 opentextshield
```

### Environment Variables for Production
```bash
OTS_LOG_LEVEL=WARNING
OTS_ALLOWED_IPS=10.0.0.0/8,172.16.0.0/12,192.168.0.0/16
OTS_CORS_ORIGINS=https://your-domain.com
```

## 🤝 Contributing

1. Follow the established project structure
2. Add comprehensive error handling
3. Include proper logging
4. Write tests for new features
5. Update documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.