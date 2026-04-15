# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
```bash
# Create virtual environment and install dependencies
python3.12 -m venv ots
source ots/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Running the API Server
```bash
# Start both API and frontend servers (recommended)
./scripts/start.sh

# Manual start (alternative - API only)
source ots/bin/activate
uvicorn src.api_interface.main:app --host 0.0.0.0 --port 9000
```

### Frontend Interface
The OpenTextShield Research Platform provides a professional web-based interface for testing the API:
- **Frontend URL**: http://localhost:8080
- **API URL**: http://localhost:8002
- **API Documentation**: http://localhost:8002/docs
- **OpenAPI Spec**: http://localhost:8002/openapi.json

Features:
- Professional AI research lab aesthetic
- Real-time system status monitoring with visual indicators
- Advanced text analysis interface with technical terminology
- OpenTextShield mBERT model with detailed architecture information
- Comprehensive results display with performance metrics
- Technical details panel showing model architecture and processing info
- Sample message functionality for quick testing
- Responsive design optimized for research environments

### Testing the API
```bash
# Test the API endpoint
curl -X POST "http://localhost:8002/predict/" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"text":"Your SMS content here","model":"ots-mbert"}'
```

### Running the SMPP Interface
```bash
# Requires OTS API running on port 8002 first
cd src/smpp_interface
npm install
node ots_smpp_proxy.js config.json

# Or use the start script
./start.sh
```

**SMPP Interface Services:**
- **SMPP Server**: port 2775 (accepts client binds)
- **Upstream**: connects to configured upstream SMSC (connection pool)
- **Classification**: calls OTS API at http://localhost:8002/predict/

**Testing the SMPP Interface:**
```bash
cd src/smpp_interface
node test_smpp.js       # Basic tests (47 tests)
node test_advanced.js   # Advanced tests: UDH, emoji, async, DLR, benchmarks (32 tests)
```

### Running Model Tests
```bash
# Test mBERT model
python src/mBERT/tests/test_sms.py

# Stress testing
python src/mBERT/tests/stressTest_500.py
python src/mBERT/tests/stressTest_1000_mlx.py
python src/mBERT/tests/stressTest_20k_mlx_api.py
```

### Training Models
```bash
# Install training dependencies first
cd src/mBERT/training/model-training/
source ../../../../ots/bin/activate

# Option 1: Minimal dependencies (recommended)
pip install -r requirements-minimal.txt

# Option 2: Full dependencies (includes all packages)
pip install -r requirements.txt

# Train mBERT model (original)
python train_ots.py

# Train mBERT model (improved version with better logging and validation)
python train_ots_improved.py
```

**Note**: The `requirements-minimal.txt` file contains only the essential dependencies needed for training, while `requirements.txt` contains the full development environment. Use minimal for faster setup.

### Docker Deployment

**🛡️ Secure Builds (Default):**
```bash
# Default secure build with non-root user and multi-stage optimization
docker build -t opentextshield .
docker run -d -p 8002:8002 -p 8081:8080 opentextshield

# Use docker-compose (recommended)
docker-compose up -d

# Ultra-secure distroless build (API only, minimal attack surface)
docker build -f Dockerfile.distroless -t opentextshield:distroless .
docker run -d -p 8002:8002 opentextshield:distroless
```

**Alternative Builds:**
```bash
# Legacy insecure build (NOT recommended for production)
docker build -f Dockerfile.insecure -t opentextshield:insecure .
docker run -d -p 8002:8002 -p 8081:8080 opentextshield:insecure

# Use pre-built image
docker pull telecomsxchange/opentextshield:latest
docker run -d -p 8002:8002 -p 8080:8080 telecomsxchange/opentextshield:latest
```

**Container Services:**
- **API**: http://localhost:8002 (with Swagger docs at /docs)
- **Frontend**: http://localhost:8080 (or 8081 if using alternative port)
- **Health Check**: http://localhost:8002/health

**Security Benefits:**
- **Default Dockerfile**: 60-80% fewer vulnerabilities, non-root execution, multi-stage builds
- **Dockerfile.distroless**: Maximum security with minimal attack surface  
- **All secure builds**: Enhanced security posture suitable for production deployment

## Architecture Overview

### mBERT-Powered Text Classification System
OpenTextShield implements a REST API powered by mBERT for global SMS spam/phishing detection:

- **mBERT (Multilingual BERT)**: Advanced transformer model supporting 104+ languages with cross-lingual transfer learning, optimized for Apple Silicon MLX

### Core Components

**API Interface** (`src/api-interface/`) - Professional modular structure:
- `main.py`: FastAPI application entry point with lifespan management
- `config/settings.py`: Environment-based configuration management
- `models/`: Pydantic request/response models with validation
- `services/`: Business logic (model_loader, prediction_service, feedback_service)
- `routers/`: API endpoints (health, prediction, feedback)
- `middleware/`: Security middleware (IP verification, CORS)
- `utils/`: Logging and custom exceptions
- Comprehensive error handling with proper HTTP status codes
- Structured logging for monitoring and debugging

**SMPP Interface** (`src/smpp_interface/`) - SMPP-to-SMPP classification proxy:
- `ots_smpp_proxy.js`: Main proxy — accepts client SMPP binds, classifies via OTS API, forwards to upstream SMSC
- `config.json`: All SMPP settings, upstream connection pool, classification rules
- `node-smpp/`: Forked SMPP protocol library (PDU encoding/decoding, session management)
- Supports configurable upstream connection pooling, UDH/multipart SMS, DLR relay, auto-reconnect
- Classification via HTTP to OTS API with round-robin load balancing across multiple API instances (`api_urls[]`)
- See `src/smpp_interface/README.md` for full documentation

**Model Training** (`src/mBERT/training/`)
- Dedicated mBERT training scripts and datasets
- Apple Silicon MLX optimization (`model-training/`)
- Dataset format: CSV with `text,label` columns where labels are `ham`, `spam`, or `phishing`

**Model Loading Path**
- mBERT: `/home/ots/OpenTextShield/src/mBERT/training/model-training/mbert_ots_model_2.5.pth`
- Model is loaded at application startup and cached in memory

### Classification Labels
- `ham`: Legitimate messages
- `spam`: Spam messages
- `phishing`: Phishing attempts

### Request/Response Format
```json
// Request
{
  "text": "Message to classify",
  "model": "ots-mbert"
}

// Response
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

### TMForum API (TMF922 - AI Inference Job Management)

OpenTextShield also provides TMForum-compliant API endpoints for enterprise integrations:

#### Create Inference Job
```bash
curl -X POST "http://localhost:8002/tmf-api/aiInferenceJob" \
  -H "Content-Type: application/json" \
  -d '{
    "priority": "normal",
    "input": {
      "inputType": "text",
      "inputFormat": "plain",
      "inputData": {"text": "Message to classify"}
    },
    "model": {
      "id": "ots-mbert",
      "name": "OpenTextShield mBERT",
      "version": "2.1",
      "type": "bert",
      "capabilities": ["text-classification", "multilingual"]
    }
  }'
```

#### Check Job Status
```bash
curl "http://localhost:8002/tmf-api/aiInferenceJob/{job_id}"
```

#### List Jobs
```bash
curl "http://localhost:8002/tmf-api/aiInferenceJob"
```

**Benefits:**
- TMForum standard compliance (TMF922)
- Async job processing with status tracking
- Enterprise-grade API for telecom operators
- Full backward compatibility with legacy API

## Key Technical Details

### Hardware Requirements
- Python 3.12 required
- Apple Silicon MLX optimization available for mBERT
- CPU inference supported for mBERT model
- Docker deployment uses Ubuntu 24.04 base

### Dependencies
- **FastAPI**: REST API framework
- **PyTorch**: Deep learning framework for mBERT model
- **Transformers**: Hugging Face library for pre-trained models
- **MLX**: Apple's machine learning framework (mBERT optimization)

### Performance Considerations
- mBERT: Variable processing time (tens to hundreds of milliseconds depending on hardware)
- Apple Silicon MLX: Optimized performance on M1/M2/M3 chips
- Model pre-loaded at startup to avoid initialization overhead
- Single high-accuracy multilingual model for consistent global results

### Dynamic Batching (v2.9+)
Concurrent prediction requests are coalesced into padded batches by the
`DynamicBatcher` in `src/api_interface/services/batching_service.py`. This
raises GPU utilisation dramatically — a T4 handles a batch of 32 SMS in
roughly the same wall-time as a single message, so per-message throughput
scales near-linearly with batch size up to saturation.

Tuning knobs (all overridable via `OTS_` env vars):

| Setting | Default | Description |
|---|---|---|
| `batching_enabled` | `true` | Master switch. Disable for single-request debugging. |
| `max_batch_size` | `32` | Maximum requests per forward pass. |
| `batch_wait_ms` | `15` | Max time to wait collecting a partial batch. |
| `max_text_length` | `96` | Token truncation (was 512; typical SMS = 20–60 tokens). |
| `use_fp16` | `true` | FP16 weights on CUDA. Ignored on CPU/MPS. |

Example:
```bash
OTS_MAX_BATCH_SIZE=16 OTS_BATCH_WAIT_MS=20 \
  uvicorn src.api_interface.main:app --host 0.0.0.0 --port 8002
```

### Observability: /metrics
Prometheus-compatible metrics are exposed at `GET /metrics` (no extra
dependency). Useful series:

- `ots_requests_total` / `ots_batches_total` — throughput counters
- `ots_inference_seconds_total` — total GPU time
- `ots_queue_depth` — current batcher backlog (adaptive-concurrency signal)
- `ots_last_batch_size` / `ots_batch_size_bucket{le="N"}` — batch efficiency
- `ots_api_info{device,fp16,max_text_length,version}` — build info

The `ots-bridge` can scrape this to drive adaptive concurrency limits.

### Development Workflow
1. Dataset preparation in CSV format with `text,label` columns
2. Model training using framework-specific training scripts
3. Model integration via API interface model loading
4. Testing with provided test scripts and stress tests
5. Docker containerization for deployment

### File Paths and Structure
- Model files use absolute paths starting with `/home/ots/OpenTextShield/`
- Training datasets stored in respective `training/dataset/` directories
- Model weights stored as `.pth` files (PyTorch)
- Configuration files and tokenizers stored alongside model weights