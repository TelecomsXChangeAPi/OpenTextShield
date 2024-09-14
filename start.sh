#!/bin/bash

# Display the CURL command to the user
echo "To test the OpenTextShield API, you can run the following curl command:"
echo 'curl -X POST "http://localhost:8002/predict/" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"text\":\"Your SMS content here\",\"model\":\"bert\"}"'

# Activate the ots virtual environment (correct path for local testing)
source /home/ots/ots/bin/activate

# Start the API service using Uvicorn (correct module path)
uvicorn src.api-interface.api_2_mBERT:app --host 0.0.0.0 --port 8002

