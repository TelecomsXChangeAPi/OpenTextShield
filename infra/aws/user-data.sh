#!/bin/bash
set -euo pipefail
exec > /var/log/user-data.log 2>&1

echo "=== OpenTextShield EC2 bootstrap ==="
echo "Started at: $(date)"

# --- Swap (2GB) to cushion the mBERT load spike on t3.medium ---
if [ ! -f /swapfile ]; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# --- System updates + docker ---
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get upgrade -y
apt-get install -y curl ca-certificates python3

if ! command -v docker > /dev/null; then
  curl -fsSL https://get.docker.com | sh
  systemctl enable docker
  systemctl start docker
  usermod -aG docker ubuntu
fi

# --- Pre-pull OTS image ---
docker pull ${docker_image}

# --- Extract & patch files the published image currently ships broken ---
# Frontend has a hardcoded localhost:8002 API URL.
# settings.py silently falls back to a no-op BaseSettings when
# pydantic-settings isn't installed, so env-var overrides are ignored.
# We bind-mount fixed copies into the container so the image can be
# upgraded without losing these patches.

mkdir -p /opt/ots
TMPC=$(docker create ${docker_image})
docker cp "$TMPC":/home/ots/OpenTextShield/frontend/index.html /opt/ots/index.html
docker cp "$TMPC":/home/ots/OpenTextShield/src/api_interface/config/settings.py /opt/ots/settings.py
docker rm "$TMPC" >/dev/null

python3 <<'PYEOF'
import re
from pathlib import Path

# 1) Frontend: derive API URL from the browser's location
fp = Path('/opt/ots/index.html')
h = fp.read_text()
old = "const API_BASE_URL = 'http://localhost:8002';"
new = "const API_BASE_URL = `$${window.location.protocol}//$${window.location.hostname}:8002`;"
if old in h:
    fp.write_text(h.replace(old, new))
    print("index.html: patched API_BASE_URL")
elif "window.location.hostname" in h:
    print("index.html: already fixed upstream, nothing to do")
else:
    print("index.html: WARNING unknown state, leaving as-is")

# 2) Settings: force CORS to wildcard. Regex is whitespace-tolerant so the
#    patch survives minor formatting changes in the image's settings.py.
#    Becomes a no-op once the image ships with pydantic-settings + wildcard.
sp = Path('/opt/ots/settings.py')
s = sp.read_text()
pattern = re.compile(r'cors_origins\s*:\s*List\[str\]\s*=\s*\[[^\]]*\]')
if re.search(r'cors_origins\s*:\s*List\[str\]\s*=\s*\[\s*"\*"\s*\]', s):
    print("settings.py: already fixed upstream, nothing to do")
elif pattern.search(s):
    sp.write_text(pattern.sub('cors_origins: List[str] = ["*"]', s, count=1))
    print("settings.py: patched cors_origins -> ['*']")
else:
    print("settings.py: WARNING cors pattern not found, leaving as-is")
PYEOF

chmod 644 /opt/ots/index.html /opt/ots/settings.py

# --- systemd unit with bind-mounts ---
cat > /etc/systemd/system/ots.service <<'UNIT'
[Unit]
Description=OpenTextShield API
After=docker.service network-online.target
Requires=docker.service
Wants=network-online.target

[Service]
Restart=always
RestartSec=10
TimeoutStartSec=0
ExecStartPre=-/usr/bin/docker rm -f ots
ExecStart=/usr/bin/docker run --rm --name ots \
  -p 8002:8002 -p 8080:8080 \
  -e OTS_MAX_BATCH_SIZE=16 \
  -e OTS_BATCH_WAIT_MS=30 \
  -v /opt/ots/settings.py:/home/ots/OpenTextShield/src/api_interface/config/settings.py:ro \
  -v /opt/ots/index.html:/home/ots/OpenTextShield/frontend/index.html:ro \
  ${docker_image}
ExecStop=/usr/bin/docker stop ots

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable ots.service
systemctl restart ots.service

echo "=== Waiting for API to become healthy ==="
for i in $(seq 1 60); do
  if curl -fs http://127.0.0.1:8002/health > /dev/null; then
    echo "API healthy after $i checks ($(date))"
    curl -fs http://127.0.0.1:8002/health
    echo
    exit 0
  fi
  sleep 5
done

echo "API not healthy after 5 minutes — debug: journalctl -u ots -n 200"
exit 0
