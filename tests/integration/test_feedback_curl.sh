#!/bin/bash

# Test OpenTextShield Feedback API with cURL
set -e

# Ensure we're running from the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "📂 Working directory: $PROJECT_ROOT"
echo "🧪 Testing OpenTextShield Feedback API with cURL"
echo "=================================================="
echo ""

API_URL="http://localhost:8002"

# Test 1: Health check
echo "1. Testing health endpoint..."
curl -s "$API_URL/health" | jq '.' || echo "Health check failed"
echo ""

# Test 2: Make a prediction first
echo "2. Making a prediction..."
PREDICTION_RESPONSE=$(curl -s -X POST "$API_URL/predict/" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "URGENT! Your bank account has been compromised. Click here to verify: http://fake-bank.com",
    "model": "ots-mbert"
  }')

echo "Prediction response:"
echo "$PREDICTION_RESPONSE" | jq '.' || echo "Prediction failed"
echo ""

# Test 3: Submit feedback
echo "3. Submitting feedback..."
FEEDBACK_RESPONSE=$(curl -s -X POST "$API_URL/feedback/" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "URGENT! Your bank account has been compromised. Click here to verify: http://fake-bank.com",
    "feedback": "This was correctly identified as phishing. Great job!",
    "thumbs_up": true,
    "thumbs_down": false,
    "user_id": "test_user_curl",
    "model": "ots-mbert"
  }')

echo "Feedback response:"
echo "$FEEDBACK_RESPONSE" | jq '.' || echo "Feedback submission failed"
echo ""

# Test 4: Try to download feedback file
echo "4. Attempting to download feedback file..."
curl -s "$API_URL/feedback/download/ots-mbert" -o /tmp/feedback_test.csv
if [ $? -eq 0 ] && [ -s /tmp/feedback_test.csv ]; then
    echo "✅ Feedback file downloaded successfully"
    echo "File contents:"
    head -5 /tmp/feedback_test.csv
    rm -f /tmp/feedback_test.csv
else
    echo "❌ Feedback file download failed or file is empty"
fi
echo ""

# Test 5: Test invalid model download
echo "5. Testing invalid model download..."
INVALID_RESPONSE=$(curl -s "$API_URL/feedback/download/invalid-model")
echo "Invalid model response:"
echo "$INVALID_RESPONSE" | jq '.' || echo "Invalid model test failed"
echo ""

echo "🎯 Feedback API tests completed!"