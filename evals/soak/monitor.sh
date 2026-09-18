#!/bin/bash
# Samples service health every minute during a soak run. One JSON line per sample.
#   monitor.sh <run_dir> <container_name> [api_url]
RUN_DIR=$1; CONTAINER=$2; API=${3:-http://127.0.0.1:8002}
OUT="$RUN_DIR/monitor.jsonl"
while true; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  t0=$(python3 -c 'import time;print(time.time())')
  health=$(curl -s --max-time 10 "$API/health" 2>/dev/null)
  t1=$(python3 -c 'import time;print(time.time())')
  status=$(echo "$health" | python3 -c 'import json,sys
try: d=json.load(sys.stdin); print(d.get("status","?"))
except Exception: print("unreachable")' 2>/dev/null)
  metrics=$(curl -s --max-time 10 "$API/metrics" 2>/dev/null)
  req=$(echo "$metrics" | awk '/^ots_requests_total/ {print $2}' | head -1)
  batches=$(echo "$metrics" | awk '/^ots_batches_total/ {print $2}' | head -1)
  inf=$(echo "$metrics" | awk '/^ots_inference_seconds_total/ {print $2}' | head -1)
  queue=$(echo "$metrics" | awk '/^ots_queue_depth/ {print $2}' | head -1)
  dstat=$(docker stats --no-stream --format '{{.CPUPerc}} {{.MemUsage}}' "$CONTAINER" 2>/dev/null)
  cpu=$(echo "$dstat" | awk '{print $1}'); mem=$(echo "$dstat" | awk '{print $2}')
  crestarts=$(docker inspect -f '{{.RestartCount}}' "$CONTAINER" 2>/dev/null)
  cstate=$(docker inspect -f '{{.State.Status}}/{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null)
  audit_kb=$(docker exec "$CONTAINER" sh -c 'du -sk /home/ots/OpenTextShield/audit_logs 2>/dev/null | cut -f1' 2>/dev/null)
  proxy_rss=$(ps -eo rss=,command= | grep '[o]ts_smpp_proxy.js' | grep node | sort -n | tail -1 | awk '{print $1}')
  proxy_log_kb=$(du -sk "$RUN_DIR/ots_smpp.log" 2>/dev/null | cut -f1)
  python3 - "$OUT" "$ts" "$status" "$t0" "$t1" "$req" "$batches" "$inf" "$queue" "$cpu" "$mem" "$crestarts" "$cstate" "$audit_kb" "$proxy_rss" "$proxy_log_kb" <<'EOF'
import json, sys
out, ts, status, t0, t1, req, batches, inf, queue, cpu, mem, crestarts, cstate, audit_kb, proxy_rss, proxy_log_kb = sys.argv[1:]
def num(x):
    try: return float(x)
    except Exception: return None
rec = {"ts": ts, "health": status, "health_ms": round((float(t1) - float(t0)) * 1000, 1),
       "requests_total": num(req), "batches_total": num(batches), "inference_seconds_total": num(inf),
       "queue_depth": num(queue), "container_cpu": cpu or None, "container_mem": mem or None,
       "container_restarts": num(crestarts), "container_state": cstate or None,
       "audit_dir_kb": num(audit_kb), "proxy_rss_kb": num(proxy_rss), "proxy_log_kb": num(proxy_log_kb)}
with open(out, "a") as f:
    f.write(json.dumps(rec) + "\n")
EOF
  sleep 60
done
