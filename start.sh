#!/bin/bash

# Exit the script if any command fails
set -e

# Display the CURL command to the user
echo "To test the OpenTextShield API, you can run the following curl command:"
echo 'curl -X POST "http://localhost:8002/predict/" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"text\":\"Your SMS content here\",\"model\":\"bert\"}"'

# Set the virtual environment path
VENV_PATH="/home/ots/OpenTextShield/ots"

# Check if the virtual environment exists, and if not, create it
if [ ! -d "$VENV_PATH" ]; then
  echo "Virtual environment not found. Creating the virtual environment..."
  python3.12 -m venv $VENV_PATH
  
  echo "Virtual environment created. Installing dependencies..."
  source $VENV_PATH/bin/activate
  pip install --upgrade pip
  pip install -r /home/ots/OpenTextShield/requirements.txt
else
  echo "Virtual environment found."
  # Activate the ots virtual environment
  source $VENV_PATH/bin/activate
fi

# Start the API service using Uvicorn (correct module path)
echo "Starting the OpenTextShield API..."
uvicorn src.api-interface.api_2_mBERT:app --host 0.0.0.0 --port 8002

