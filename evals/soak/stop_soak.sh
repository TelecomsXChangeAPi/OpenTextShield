#!/bin/bash
# Stop a soak run cleanly: client first (it flushes a final summary), then the
# host processes, then the container. Results stay in the run directory.
RUN_DIR=$1
pkill -TERM -f 'evals/soak/soak_client.js' 2>/dev/null; sleep 3
pkill -f 'evals/soak/monitor.sh' 2>/dev/null
pkill -f 'evals/soak/chaos.sh' 2>/dev/null
pkill -f 'node ots_smpp_proxy.js' 2>/dev/null
pkill -f 'node dummy_upstream.js 2776' 2>/dev/null
docker rm -f ots-soak >/dev/null 2>&1
echo "stopped; results in $RUN_DIR"
