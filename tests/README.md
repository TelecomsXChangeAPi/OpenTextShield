# OpenTextShield Test Suite

This directory contains comprehensive tests for the OpenTextShield platform.

## Directory Structure

```
tests/
├── api/                    # API endpoint tests
│   ├── test_api_complete.py      # Complete API functionality test
│   ├── test_api_full.py          # Full API test suite
│   ├── test_api_simple.py        # Basic API tests
│   ├── test_api_structure.py     # API structure validation
│   ├── test_feedback_api.py      # Feedback API tests
│   └── test_server.py            # Server functionality tests
├── integration/            # Integration and end-to-end tests
│   ├── test-docker.sh           # Docker container testing
│   ├── test-frontend.sh         # Frontend interface testing
│   └── test_feedback_curl.sh    # Curl-based feedback testing
├── data/                   # Test data and fixtures
│   ├── test_obvious_ham.json     # Ham message test data
│   ├── test_obvious_phishing.json # Phishing test data
│   ├── test_obvious_spam.json    # Spam test data
│   ├── test_phishing.json        # Additional phishing samples
│   └── test_spam.json            # Additional spam samples
└── README.md              # This file
```

## Running Tests

### API Tests
```bash
# Run all API tests
python -m pytest tests/api/ -v

# Run specific test
python tests/api/test_api_simple.py
```

### Integration Tests
```bash
# Test Docker container
bash tests/integration/test-docker.sh

# Test frontend interface
bash tests/integration/test-frontend.sh

# Test feedback API with curl
bash tests/integration/test_feedback_curl.sh
```

### Model Tests (Located in src/mBERT/tests/)
```bash
# Run mBERT model tests
python src/mBERT/tests/test_sms.py

# Run stress tests
python src/mBERT/tests/stressTest_500.py
```

## Test Categories

### 1. API Tests (`tests/api/`)
- **Purpose**: Validate API endpoints, request/response formats, error handling
- **Coverage**: Prediction API, feedback API, health checks, model loading
- **Tools**: Python requests, pytest framework

### 2. Integration Tests (`tests/integration/`)
- **Purpose**: End-to-end testing of complete system functionality
- **Coverage**: Docker containers, frontend interface, API integration
- **Tools**: Bash scripts, curl commands, container testing

### 3. Test Data (`tests/data/`)
- **Purpose**: Standardized test messages for consistent testing
- **Categories**: Ham (legitimate), spam, phishing messages
- **Format**: JSON files with message content and expected classifications

### 4. Model Tests (`src/mBERT/tests/`)
- **Purpose**: mBERT model functionality and performance testing
- **Coverage**: Model loading, prediction accuracy, stress testing
- **Tools**: PyTorch, transformers, MLX (Apple Silicon)

## Test Data Format

Test data files follow this JSON structure:
```json
{
  "text": "Test message content",
  "expected_label": "ham|spam|phishing",
  "category": "obvious|subtle|edge_case"
}
```

## Adding New Tests

### API Tests
1. Create new test file in `tests/api/`
2. Follow existing naming convention: `test_[feature]_[type].py`
3. Use pytest framework for assertions
4. Include setup/teardown for API client

### Integration Tests
1. Create new shell script in `tests/integration/`
2. Follow naming convention: `test-[component].sh`
3. Include error handling and cleanup
4. Provide clear success/failure indicators

### Test Data
1. Add new JSON files to `tests/data/`
2. Use descriptive filenames indicating content type
3. Include diverse message types and edge cases
4. Validate JSON format before committing

## CI/CD Integration

These tests are designed to be integrated into continuous integration pipelines:

```yaml
# Example GitHub Actions step
- name: Run API Tests
  run: |
    python -m pytest tests/api/ --verbose --tb=short
    
- name: Run Integration Tests
  run: |
    bash tests/integration/test-docker.sh
    bash tests/integration/test-frontend.sh
```

## Performance Testing

For performance and stress testing, see:
- `src/mBERT/tests/stressTest_*.py` - Load testing scripts
- Model-specific performance benchmarks
- Container resource usage monitoring

## Security Testing

Security-related tests include:
- API input validation and sanitization
- Authentication and authorization (when implemented)
- Container security scanning
- Dependency vulnerability testing