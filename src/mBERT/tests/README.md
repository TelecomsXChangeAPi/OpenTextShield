# OpenTextShield mBERT Tests

Modern, maintainable test suite for the OpenTextShield mBERT API with shared utilities and comprehensive coverage.

## Architecture

### Core Components
- **`test_utils.py`** - Shared utilities and base classes
  - `OTSAPIClient` - Centralized API communication
  - `TestDataGenerator` - Test data generation
  - `TestLogger` - Unified logging
  - `TestRunner` - Base test runner with common functionality

### Test Suites
- **`test_basic.py`** - Basic functionality tests
  - Single message classification
  - API response time measurement
  - Basic error handling

- **`test_stress.py`** - Comprehensive stress testing
  - Sequential stress tests (500 messages)
  - Concurrent stress tests (1000 messages) 
  - Massive stress tests (20,000 messages)
  - Configurable worker pools and progress tracking

- **`pytest_tests.py`** - Professional pytest suite
  - Unit tests for core functionality
  - Performance benchmarks
  - Error condition testing
  - Pytest fixtures and markers

### Test Orchestration
- **`run_all_tests.py`** - Modern test runner
  - Command-line interface
  - Prerequisite checking
  - Comprehensive reporting
  - Timeout handling

## Usage

### Prerequisites
Ensure the OpenTextShield API is running:
```bash
cd /Users/ameedjamous/programming/OpenTextShield
./start.sh
```

### Running Tests

#### Quick Commands
```bash
cd src/mBERT/tests

# List available tests
python run_all_tests.py list

# Run all tests
python run_all_tests.py all

# Run basic functionality tests
python run_all_tests.py basic

# Run specific stress test
python run_all_tests.py stress 500
python run_all_tests.py stress 1000  
python run_all_tests.py stress 20k

# Run all stress tests
python run_all_tests.py stress all

# Run pytest suite
python run_all_tests.py pytest
```

#### Direct Execution
```bash
# Run individual test files directly
python test_basic.py
python test_stress.py 500
python -m pytest pytest_tests.py -v
```

## API Configuration

All tests are configured to use:
- **API Endpoint**: `http://localhost:8002/predict/`
- **Model**: `ots-mbert` (OpenTextShield mBERT)
- **Expected Response Format**:
  ```json
  {
    "label": "ham|spam|phishing",
    "probability": 0.95,
    "processing_time": 0.15
  }
  ```

## Test Results

### Logs
- **`ots_api_test_logs.log`** - Detailed request/response logs from concurrent tests
- **`prediction_logs.log`** - Legacy log file (may be present from old tests)

### Performance Expectations
- **Basic test**: < 1 second response time
- **500 messages**: Depends on API performance
- **1000 concurrent**: Tests concurrent handling capability  
- **20k messages**: Long-running test (several minutes)

## Troubleshooting

### Common Issues

1. **API Not Running**
   ```
   ❌ OpenTextShield API is not running!
   ```
   **Solution**: Start the API with `./start.sh`

2. **Connection Refused**
   ```
   Error: Connection refused
   ```
   **Solution**: Verify API is running on port 8002

3. **Test Timeouts**
   **Solution**: Tests have a 5-minute timeout; for slower systems, modify the timeout in `run_tests.py`

### Debugging
- Check API health: `curl http://localhost:8002/health`
- View API logs in the terminal where `./start.sh` is running
- Review test logs in `ots_api_test_logs.log`

## Development

### Adding New Tests
1. Create new test file following the naming pattern `test_*.py` or `stressTest_*.py`
2. Use the `predict_via_api()` function pattern for consistency
3. Add the test file to the `tests` list in `run_tests.py`

### Test Structure
All tests follow this pattern:
```python
def predict_via_api(text, url='http://localhost:8002/predict/'):
    """Make prediction via OpenTextShield API."""
    data = {"text": text, "model": "ots-mbert"}
    # ... API call logic
    
def main():
    """Test description and execution."""
    # ... test logic
```