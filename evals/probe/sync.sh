#!/usr/bin/env bash
# Pull the probe logs and report from the host into the repo, or push the code there.
#   evals/probe/sync.sh          # host -> repo (logs/, REPORT.md)
#   evals/probe/sync.sh --push   # repo -> host (probe.py, LABELING_GUIDE.md, units)
set -euo pipefail
cd "$(dirname "$0")"
HOST=${OTS_HOST:-ubuntu@3.99.29.158}
KEY=${OTS_KEY:-$HOME/.ssh/ots-deploy-key.pem}
if [ "${1:-}" = "--push" ]; then
  mkdir -p _stage/docs && cp probe.py _stage/ && cp ../../docs/LABELING_GUIDE.md _stage/docs/
  scp -q -i "$KEY" -r _stage/. "$HOST:/opt/ots-probe/"
  scp -q -i "$KEY" ots-probe.service ots-probe.timer "$HOST:/tmp/"
  ssh -i "$KEY" "$HOST" 'sudo mv /tmp/ots-probe.service /tmp/ots-probe.timer /etc/systemd/system/ && sudo systemctl daemon-reload && systemctl list-timers ots-probe.timer --no-pager | head -3'
  rm -rf _stage
else
  mkdir -p logs
  rsync -az -e "ssh -i $KEY" "$HOST:/opt/ots-probe/logs/" logs/
  rsync -az -e "ssh -i $KEY" "$HOST:/opt/ots-probe/REPORT.md" REPORT.md 2>/dev/null || true
  echo "synced: $(ls logs/*.jsonl 2>/dev/null | wc -l | tr -d ' ') day files, $(wc -l < logs/improvements.log 2>/dev/null || echo 0) improvement lines"
fi
