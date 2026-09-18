#!/bin/bash
# Launch a detached soak run of a release-candidate image through the SMPP proxy.
#
#   evals/soak/run_soak.sh <image> <run_dir> [hours] [base_rate] [burst_rate]
#
# Starts: the container (restart unless-stopped, audit ON inside the container),
# the dummy upstream, the proxy (log in run_dir), the per-minute monitor, the
# 6-hourly upstream restart, and the traffic client. Everything is nohup'd so it
# survives the launching shell. Stop with: evals/soak/stop_soak.sh <run_dir>
set -euo pipefail
IMAGE=$1; RUN_DIR=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); HOURS=${3:-48}; BASE=${4:-3}; BURST=${5:-12}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PROXY_DIR="$REPO/src/smpp_interface"
CONTAINER=ots-soak
mkdir -p "$RUN_DIR"

python3 - "$PROXY_DIR/config.json" "$RUN_DIR" <<'EOF'
import json, sys
src, run = sys.argv[1:]
c = json.load(open(src))
c["logging"]["file"] = f"{run}/ots_smpp.log"
json.dump(c, open(f"{run}/config.json", "w"), indent=2)
EOF

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  docker run -d --name "$CONTAINER" --restart unless-stopped -p 8002:8002 -p 8080:8080 "$IMAGE" >/dev/null
fi
for i in $(seq 1 90); do curl -s --max-time 5 http://127.0.0.1:8002/health | grep -q healthy && break; sleep 3; done
curl -s --max-time 5 http://127.0.0.1:8002/health | cut -c1-140; echo

pgrep -f 'node dummy_upstream.js 2776' >/dev/null || (cd "$PROXY_DIR" && nohup node dummy_upstream.js 2776 >> "$RUN_DIR/upstream.log" 2>&1 < /dev/null &)
pkill -f 'node ots_smpp_proxy.js' 2>/dev/null || true; sleep 1
(cd "$PROXY_DIR" && nohup node ots_smpp_proxy.js "$RUN_DIR/config.json" >> "$RUN_DIR/proxy.stdout" 2>&1 < /dev/null &)
for i in $(seq 1 30); do nc -z 127.0.0.1 2775 2>/dev/null && break; sleep 1; done

nohup bash "$HERE/monitor.sh" "$RUN_DIR" "$CONTAINER" >> "$RUN_DIR/monitor.stderr" 2>&1 < /dev/null &
nohup bash "$HERE/chaos.sh" "$RUN_DIR" 2776 21600 >> "$RUN_DIR/chaos.stderr" 2>&1 < /dev/null &
(cd "$PROXY_DIR" && NODE_PATH="$PROXY_DIR/node_modules" nohup node "$HERE/soak_client.js" \
   --corpus "$(dirname "$RUN_DIR")/corpus.jsonl" --out "$RUN_DIR" \
   --base-rate "$BASE" --burst-rate "$BURST" --hours "$HOURS" >> "$RUN_DIR/client.stdout" 2>&1 < /dev/null &)

cat > "$RUN_DIR/RUN.md" <<EOF
image: $IMAGE
started: $(date -u +%Y-%m-%dT%H:%M:%SZ)
hours: $HOURS  base_rate: $BASE  burst_rate: $BURST
report: python3 $HERE/soak_report.py $RUN_DIR
stop:   bash $HERE/stop_soak.sh $RUN_DIR
EOF
echo "soak started in $RUN_DIR"
