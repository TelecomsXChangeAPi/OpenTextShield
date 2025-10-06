#!/bin/bash
# Verify Docker setup for OpenTextShield v2.5

echo "🔍 Verifying Docker Setup for OpenTextShield v2.5"
echo "================================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if Docker files exist
echo -n "Checking Docker files... "
files=("Dockerfile" "docker-compose.yml" ".dockerignore")
missing=()
for file in "${files[@]}"; do
    if [ ! -f "$file" ]; then
        missing+=("$file")
    fi
done

if [ ${#missing[@]} -eq 0 ]; then
    echo -e "${GREEN}PASSED${NC} - All main Docker files present"
else
    echo -e "${RED}FAILED${NC} - Missing files: ${missing[*]}"
    exit 1
fi

# Check model and dataset files
echo -n "Checking model and dataset files... "
if [ -f "src/mBERT/training/model-training/mbert_ots_model_2.5.pth" ] && [ -f "src/mBERT/training/model-training/dataset/sms_spam_phishing_dataset_v2.4_combined.csv" ]; then
    echo -e "${GREEN}PASSED${NC} - v2.5 model and combined dataset present"
else
    echo -e "${RED}FAILED${NC} - Model or dataset files missing"
    exit 1
fi

# Check .dockerignore includes correct files
echo -n "Checking .dockerignore configuration... "
if grep -q "mbert_ots_model_2.5.pth" .dockerignore && grep -q "sms_spam_phishing_dataset_v2.4_combined.csv" .dockerignore; then
    echo -e "${GREEN}PASSED${NC} - .dockerignore includes v2.5 files"
else
    echo -e "${RED}FAILED${NC} - .dockerignore missing v2.5 references"
    exit 1
fi

# Check API configuration
echo -n "Checking API configuration... "
if grep -q "mbert_ots_model_2.5.pth" src/api_interface/config/settings.py; then
    echo -e "${GREEN}PASSED${NC} - API configured for v2.5 model"
else
    echo -e "${RED}FAILED${NC} - API not configured for v2.5"
    exit 1
fi

# Check alternative Docker files
echo -n "Checking alternative Docker files... "
alt_files=("Dockerfile.dev" "Dockerfile.distroless" "Dockerfile.secure" "Dockerfile.simple")
alt_missing=()
for file in "${alt_files[@]}"; do
    if [ ! -f "$file" ]; then
        alt_missing+=("$file")
    fi
done

if [ ${#alt_missing[@]} -eq 0 ]; then
    echo -e "${GREEN}PASSED${NC} - All alternative Docker files present"
else
    echo -e "${YELLOW}WARNING${NC} - Missing alternative files: ${alt_missing[*]}"
fi

# Check file sizes
echo -n "Checking file sizes... "
model_size=$(stat -f%z "src/mBERT/training/model-training/mbert_ots_model_2.5.pth" 2>/dev/null || stat -c%s "src/mBERT/training/model-training/mbert_ots_model_2.5.pth" 2>/dev/null)
dataset_size=$(stat -f%z "src/mBERT/training/model-training/dataset/sms_spam_phishing_dataset_v2.4_combined.csv" 2>/dev/null || stat -c%s "src/mBERT/training/model-training/dataset/sms_spam_phishing_dataset_v2.4_combined.csv" 2>/dev/null)

if [ "$model_size" -gt 100000000 ] && [ "$dataset_size" -gt 10000000 ]; then  # 100MB and 10MB
    echo -e "${GREEN}PASSED${NC} - Files are correct size"
else
    echo -e "${RED}FAILED${NC} - File sizes incorrect"
    exit 1
fi

echo ""
echo "🎉 Docker Setup Verification Complete!"
echo "======================================"
echo -e "${GREEN}✅ All checks passed! Docker is ready for deployment.${NC}"
echo ""
echo "🚀 Deployment Commands:"
echo "======================="
echo "# Build and run single container"
echo "docker build -t opentextshield:v2.5 ."
echo "docker run -d -p 8002:8002 -p 8080:8080 opentextshield:v2.5"
echo ""
echo "# Or use docker-compose"
echo "docker-compose up -d"
echo ""
echo "# Test the deployment"
echo "curl http://localhost:8002/health"
echo "curl -X POST 'http://localhost:8002/predict/' -H 'Content-Type: application/json' -d '{\"text\":\"test message\",\"model\":\"ots-mbert\"}'"
echo ""
echo "🔗 Available Endpoints:"
echo "- API: http://localhost:8002"
echo "- Frontend: http://localhost:8080"
echo "- Health: http://localhost:8002/health"
echo "- Docs: http://localhost:8002/docs"
echo "- TMForum: http://localhost:8002/tmf-api/aiInferenceJob"