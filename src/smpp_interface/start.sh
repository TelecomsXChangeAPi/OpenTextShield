#!/bin/bash
# OpenTextShield SMPP Interface - Start Script
#
# Usage:
#   ./start.sh                    # Use default config.json
#   ./start.sh /path/to/config    # Use custom config

cd "$(dirname "$0")"

# Install deps if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Config file
CONFIG=${1:-"./config.json"}

echo "Starting OTS SMPP Interface..."
echo "Config: $CONFIG"

node ots_smpp_proxy.js "$CONFIG"
