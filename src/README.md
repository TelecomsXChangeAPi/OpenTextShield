# OpenTextShield Source Code

This directory contains the core source code for the OpenTextShield platform.

## 📁 Directory Structure

```
src/
├── api_interface/           # Modern FastAPI application
│   ├── config/             # Configuration management
│   ├── models/             # Pydantic request/response models
│   ├── routers/            # API endpoint definitions
│   ├── services/           # Business logic and model loading
│   ├── middleware/         # Security and CORS middleware
│   └── utils/              # Logging and exception handling
│
└── mBERT/                  # Multilingual BERT implementation
    ├── training/
    │   └── model-training/ # Modern training suite with MLX optimization
    └── tests/              # Comprehensive test suite
```

## 🚀 Getting Started

### API Interface
The `api_interface/` directory contains the modern FastAPI application:

```bash
# Start the API server
cd /path/to/OpenTextShield
./scripts/start.sh
```

### Model Training
The `mBERT/training/model-training/` directory contains training tools:

```bash
# Train a new model
cd src/mBERT/training/model-training/
python train_ots_improved.py
```

### Testing
The `mBERT/tests/` directory contains comprehensive tests:

```bash
# Run all tests
cd src/mBERT/tests/
python run_all_tests.py all
```

## 🏗 Architecture

### API Interface (`api_interface/`)
- **Professional FastAPI Structure**: Modular, maintainable code organization
- **Pydantic Models**: Strict request/response validation
- **Comprehensive Logging**: Structured logging for monitoring and debugging
- **Security Middleware**: CORS, rate limiting, and input validation
- **Health Checks**: Built-in health monitoring endpoints

### mBERT Training (`mBERT/training/model-training/`)
- **Apple Silicon Optimization**: Native MLX framework support
- **Multi-device Support**: Automatic device detection (MPS, CUDA, CPU)
- **Modern Training Pipeline**: Validation splits, early stopping, comprehensive metrics
- **Dataset Management**: Organized dataset handling and validation

### Testing Suite (`mBERT/tests/`)
- **Comprehensive Coverage**: Unit tests, integration tests, stress tests
- **Professional Test Runner**: Command-line interface with detailed reporting
- **Performance Benchmarking**: Stress testing with configurable parameters
- **API Testing**: Full API endpoint validation

## 📊 Key Features

- **Real-time Classification**: <200ms response time
- **Multilingual Support**: 104+ languages via mBERT
- **Production Ready**: Professional error handling and monitoring
- **Scalable Architecture**: Docker and Kubernetes ready
- **Security First**: Input validation, rate limiting, secure headers

For detailed documentation, see the component-specific README files in each directory.
