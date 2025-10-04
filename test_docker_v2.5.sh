#!/bin/bash
# Test script for OpenTextShield v2.5 Docker container

echo "🧪 Testing OpenTextShield v2.5 Docker Configuration"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test cases
TEST_CASES=(
    "Transfer 500 euros to account DE1234567890 at Deutsche Bank:spam"
    "Your ING Bank account has been credited with 250 EUR:spam"
    "Bonjour, votre colis DHL numéro FR123456789 est arrivé:ham"
    "Free iPhone giveaway! Click here to claim:http://spam-link.com:spam"
    "Meeting scheduled for tomorrow at 3 PM in conference room A:ham"
)

# Function to test API
test_api() {
    local text="$1"
    local expected="$2"

    echo -n "Testing: \"$text\" (expected: $expected) ... "

    response=$(curl -s -X POST "http://localhost:8002/predict/" \
        -H "accept: application/json" \
        -H "Content-Type: application/json" \
        -d "{\"text\":\"$text\",\"model\":\"ots-mbert\"}")

    if [ $? -ne 0 ]; then
        echo -e "${RED}FAILED${NC} - API not responding"
        return 1
    fi

    predicted=$(echo "$response" | jq -r '.label' 2>/dev/null)
    confidence=$(echo "$response" | jq -r '.probability' 2>/dev/null)
    processing_time=$(echo "$response" | jq -r '.processing_time' 2>/dev/null)

    if [ "$predicted" = "$expected" ]; then
        echo -e "${GREEN}PASSED${NC} (confidence: $(printf "%.1f" $(echo "$confidence * 100" | bc -l))%, time: $(printf "%.2f" $processing_time)s)"
        return 0
    else
        echo -e "${RED}FAILED${NC} (predicted: $predicted, expected: $expected)"
        return 1
    fi
}

# Test health endpoint
echo -n "Testing health endpoint ... "
health_response=$(curl -s http://localhost:8002/health)
if echo "$health_response" | jq -e '.status == "healthy"' >/dev/null 2>&1; then
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${RED}FAILED${NC} - Health check failed"
    exit 1
fi

# Test model loading
echo -n "Testing model loading ... "
if echo "$health_response" | jq -e '.models_loaded.mbert_multilingual == true' >/dev/null 2>&1; then
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${RED}FAILED${NC} - Model not loaded"
    exit 1
fi

echo ""
echo "🧪 Running classification tests..."
echo "==================================="

passed=0
total=0

for test_case in "${TEST_CASES[@]}"; do
    text="${test_case%%:*}"
    expected="${test_case##*:}"

    if test_api "$text" "$expected"; then
        ((passed++))
    fi
    ((total++))
done

echo ""
echo "📊 Test Results Summary"
echo "======================="
echo "Passed: $passed/$total tests"

if [ $passed -eq $total ]; then
    echo -e "${GREEN}🎉 All tests passed! Docker container is working correctly.${NC}"
    echo ""
    echo "🚀 Docker Deployment Commands:"
    echo "docker build -t opentextshield:v2.5 ."
    echo "docker run -d -p 8002:8002 -p 8080:8080 opentextshield:v2.5"
    echo ""
    echo "🔗 API Endpoints:"
    echo "- Health: http://localhost:8002/health"
    echo "- Predict: http://localhost:8002/predict/"
    echo "- Docs: http://localhost:8002/docs"
    echo "- Frontend: http://localhost:8080"
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Please check the configuration.${NC}"
    exit 1
fi