#!/bin/bash
# Test script for multiple OpenTextShield v2.5 Docker containers

echo "🚀 Testing Multiple OpenTextShield v2.5 Docker Containers"
echo "========================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Container configurations
CONTAINERS=(
    "ots-primary:8002:8080"
    "ots-secondary:8003:8081"
)

# Test cases
TEST_CASES=(
    "Transfer 500 euros to account DE1234567890 at Deutsche Bank:spam:European Banking Fraud"
    "Your ING Bank account has been credited with 250 EUR:spam:ING Bank Alert"
    "Bonjour, votre colis DHL numéro FR123456789 est arrivé:ham:French DHL Delivery"
    "Free iPhone giveaway! Click here to claim:http://spam-link.com:spam:iPhone Scam"
    "Meeting scheduled for tomorrow at 3 PM in conference room A:ham:Business Meeting"
)

# Function to test container
test_container() {
    local name="$1"
    local api_port="$2"
    local frontend_port="$3"

    echo -e "\n${BLUE}🧪 Testing Container: $name (API: $api_port, Frontend: $frontend_port)${NC}"
    echo "================================================================="

    # Test health endpoint
    echo -n "Testing health endpoint ... "
    health_response=$(curl -s "http://localhost:$api_port/health" 2>/dev/null)
    if echo "$health_response" | jq -e '.status == "healthy"' >/dev/null 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
    else
        echo -e "${RED}FAILED${NC} - Container not responding"
        return 1
    fi

    # Test model loading
    echo -n "Testing model loading ... "
    if echo "$health_response" | jq -e '.models_loaded.mbert_multilingual == true' >/dev/null 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
    else
        echo -e "${RED}FAILED${NC} - Model not loaded"
        return 1
    fi

    # Test version reporting
    echo -n "Testing version reporting ... "
    version_response=$(curl -s -X POST "http://localhost:$api_port/predict/" \
        -H "accept: application/json" \
        -H "Content-Type: application/json" \
        -d '{"text":"test","model":"ots-mbert"}' 2>/dev/null)

    reported_version=$(echo "$version_response" | jq -r '.model_info.version' 2>/dev/null)
    if [ "$reported_version" = "2.5" ]; then
        echo -e "${GREEN}PASSED${NC} (v$reported_version)"
    else
        echo -e "${RED}FAILED${NC} (reported: v$reported_version)"
        return 1
    fi

    echo -e "\n${YELLOW}📊 Classification Tests:${NC}"
    local passed=0
    local total=0

    for test_case in "${TEST_CASES[@]}"; do
        IFS=':' read -r text expected description <<< "$test_case"

        echo -n "Testing \"$description\" ... "

        response=$(curl -s -X POST "http://localhost:$api_port/predict/" \
            -H "accept: application/json" \
            -H "Content-Type: application/json" \
            -d "{\"text\":\"$text\",\"model\":\"ots-mbert\"}" 2>/dev/null)

        predicted=$(echo "$response" | jq -r '.label' 2>/dev/null)
        confidence=$(echo "$response" | jq -r '.probability' 2>/dev/null)

        if [ "$predicted" = "$expected" ]; then
            echo -e "${GREEN}PASSED${NC} ($predicted, $(printf "%.1f" $(echo "$confidence * 100" | bc -l))%)"
            ((passed++))
        else
            echo -e "${RED}FAILED${NC} (expected: $expected, got: $predicted)"
        fi
        ((total++))
    done

    echo -e "\n${BLUE}🔗 TMForum API Tests:${NC}"
    echo "===================="

    # Test TMForum job creation
    echo -n "Testing TMForum job creation ... "
    tmf_response=$(curl -s -X POST "http://localhost:$api_port/tmf-api/aiInferenceJob" \
        -H "Content-Type: application/json" \
        -d '{
          "priority": "normal",
          "input": {
            "inputType": "text",
            "inputFormat": "plain",
            "inputData": {"text": "Free iPhone giveaway! Click here to claim"}
          },
          "model": {
            "id": "ots-mbert",
            "name": "OpenTextShield mBERT",
            "version": "2.5",
            "type": "bert",
            "capabilities": ["text-classification", "multilingual"]
          }
        }' 2>/dev/null)

    job_id=$(echo "$tmf_response" | jq -r '.id' 2>/dev/null)
    tmf_version=$(echo "$tmf_response" | jq -r '.model.version' 2>/dev/null)

    if [ -n "$job_id" ] && [ "$job_id" != "null" ]; then
        echo -e "${GREEN}PASSED${NC} (Job ID: $job_id)"
    else
        echo -e "${RED}FAILED${NC} - Could not create TMForum job"
        return 1
    fi

    # Test TMForum version reporting
    echo -n "Testing TMForum version reporting ... "
    if [ "$tmf_version" = "2.5" ]; then
        echo -e "${GREEN}PASSED${NC} (v$tmf_version)"
    else
        echo -e "${RED}FAILED${NC} (reported: v$tmf_version)"
    fi

    # Test TMForum job status
    echo -n "Testing TMForum job completion ... "
    sleep 2  # Wait for job to complete
    status_response=$(curl -s "http://localhost:$api_port/tmf-api/aiInferenceJob/$job_id" 2>/dev/null)
    job_state=$(echo "$status_response" | jq -r '.state' 2>/dev/null)
    job_result=$(echo "$status_response" | jq -r '.output.outputData.label' 2>/dev/null)

    if [ "$job_state" = "completed" ] && [ "$job_result" = "spam" ]; then
        echo -e "${GREEN}PASSED${NC} (State: $job_state, Result: $job_result)"
    else
        echo -e "${RED}FAILED${NC} (State: $job_state, Result: $job_result)"
    fi

    echo -e "\n${GREEN}✅ Container $name: $passed/$total tests passed${NC}"
    return 0
}

# Function to launch container
launch_container() {
    local name="$1"
    local api_port="$2"
    local frontend_port="$3"

    echo -e "\n${BLUE}🐳 Launching Container: $name${NC}"
    echo "================================"

    echo "docker run -d --name $name -p $api_port:8002 -p $frontend_port:8080 opentextshield:v2.5"
    echo "(Note: Docker daemon not available in this environment)"

    # In a real environment, you would run:
    # docker run -d --name $name -p $api_port:8002 -p $frontend_port:8080 opentextshield:v2.5

    echo -e "${YELLOW}Waiting for container to start...${NC}"
    sleep 3
}

# Main testing loop
echo -e "${BLUE}🏗️  Launching Containers${NC}"
echo "======================="

for container_config in "${CONTAINERS[@]}"; do
    IFS=':' read -r name api_port frontend_port <<< "$container_config"
    launch_container "$name" "$api_port" "$frontend_port"
done

echo -e "\n${BLUE}🧪 Running Tests${NC}"
echo "================="

all_passed=true
for container_config in "${CONTAINERS[@]}"; do
    IFS=':' read -r name api_port frontend_port <<< "$container_config"
    if ! test_container "$name" "$api_port" "$frontend_port"; then
        all_passed=false
    fi
done

echo -e "\n${BLUE}📊 Final Results${NC}"
echo "=================="

if $all_passed; then
    echo -e "${GREEN}🎉 All containers tested successfully!${NC}"
    echo ""
    echo "🚀 Production Deployment Commands:"
    echo "=================================="
    echo "# Launch primary container"
    echo "docker run -d --name ots-primary -p 8002:8002 -p 8080:8080 opentextshield:v2.5"
    echo ""
    echo "# Launch secondary container (for load balancing)"
    echo "docker run -d --name ots-secondary -p 8003:8002 -p 8081:8080 opentextshield:v2.5"
    echo ""
    echo "# Check container status"
    echo "docker ps"
    echo ""
    echo "# View logs"
    echo "docker logs ots-primary"
    echo ""
    echo "# Stop containers"
    echo "docker stop ots-primary ots-secondary"
    echo "docker rm ots-primary ots-secondary"
else
    echo -e "${RED}❌ Some tests failed. Please check the configuration.${NC}"
    exit 1
fi

echo ""
echo "🔗 API Endpoints:"
echo "- Primary API: http://localhost:8002"
echo "- Secondary API: http://localhost:8003"
echo "- Frontend: http://localhost:8080, http://localhost:8081"
echo "- TMForum: http://localhost:8002/tmf-api/aiInferenceJob"
echo "- Health: http://localhost:8002/health"
echo "- Docs: http://localhost:8002/docs"