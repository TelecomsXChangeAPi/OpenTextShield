#!/bin/bash
# Restarts the dummy upstream SMSC on a schedule so the proxy's reconnect path
# gets exercised during the soak. Logs each event.
#   chaos.sh <run_dir> <upstream_port> <interval_seconds>
RUN_DIR=$1; PORT=${2:-2776}; INTERVAL=${3:-21600}
PROXY_DIR="$(cd "$(dirname "$0")/../../src/smpp_interface" && pwd)"
LOG="$RUN_DIR/chaos.log"
while true; do
  sleep "$INTERVAL"
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  pid=$(pgrep -f "node dummy_upstream.js $PORT" | head -1)
  echo "$ts killing upstream pid=$pid" >> "$LOG"
  [ -n "$pid" ] && kill "$pid"
  sleep 20
  (cd "$PROXY_DIR" && nohup node dummy_upstream.js "$PORT" >> "$RUN_DIR/upstream.log" 2>&1 &)
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) upstream restarted" >> "$LOG"
done
