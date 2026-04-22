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

# --- System packages + Docker + Caddy ---
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get upgrade -y
apt-get install -y curl ca-certificates python3 debian-keyring debian-archive-keyring apt-transport-https gnupg

# Docker
if ! command -v docker > /dev/null; then
  curl -fsSL https://get.docker.com | sh
  systemctl enable docker
  systemctl start docker
  usermod -aG docker ubuntu
fi

# Caddy (official repo — gives us a systemd unit + automatic LE)
if ! command -v caddy > /dev/null; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -y
  apt-get install -y caddy
fi

# --- Pull OTS image + extract files for the two in-image bug workarounds ---
docker pull ${docker_image}

mkdir -p /opt/ots
TMPC=$(docker create ${docker_image})
docker cp "$TMPC":/home/ots/OpenTextShield/frontend/index.html /opt/ots/index.html
docker cp "$TMPC":/home/ots/OpenTextShield/src/api_interface/config/settings.py /opt/ots/settings.py
docker rm "$TMPC" >/dev/null

python3 <<'PYEOF'
import re
from pathlib import Path

# 1) Frontend: smart URL + rewrite the example curl block on page load so the
#    example reflects the real API URL rather than "http://localhost:8002".
#    Idempotent; skips if image is already fixed.
fp = Path('/opt/ots/index.html')
h_original = fp.read_text()
h = h_original

old_const = "const API_BASE_URL = 'http://localhost:8002';"
new_block = (
    "const API_BASE_URL = (!window.location.port || "
    "['80','443'].includes(window.location.port)) "
    "? window.location.origin "
    ": `$${window.location.protocol}//$${window.location.hostname}:8002`;\n"
    "\n"
    "        (function initCurlSample() {\n"
    "            const el = document.getElementById('curlRequest');\n"
    "            if (!el) return;\n"
    "            el.textContent = `curl -X POST \"$${API_BASE_URL}/predict/\" \\\\\n"
    "  -H \"accept: application/json\" \\\\\n"
    "  -H \"Content-Type: application/json\" \\\\\n"
    "  -d '{\"text\":\"Your message here\",\"model\":\"ots-mbert\"}'`;\n"
    "        })();"
)
h = h.replace(old_const, new_block)
# Replace every hardcoded curl URL (static <pre> + post-classification JS template)
h = h.replace('"http://localhost:8002/predict/"', '"$${API_BASE_URL}/predict/"')

# Fix the post-classification curl payload so apostrophes in user input don't
# break the bash single-quoted JSON. Uses JSON.stringify + '\''-style bash escape.
old_payload_block = (
    "            const curlRequest = document.getElementById('curlRequest');\n"
    "            if (curlRequest) {\n"
    "                const curlText = `curl -X POST \"$${API_BASE_URL}/predict/\" \\\\\n"
    "  -H \"accept: application/json\" \\\\\n"
    "  -H \"Content-Type: application/json\" \\\\\n"
    "  -d '{\"text\":\"$${originalText.replace(/\"/g, '\\\\\"')}\",\"model\":\"$${model}\"}'`;\n"
    "                curlRequest.textContent = curlText;\n"
    "            }"
)
new_payload_block = (
    "            const curlRequest = document.getElementById('curlRequest');\n"
    "            if (curlRequest) {\n"
    "                const payload = JSON.stringify({text: originalText, model: model})\n"
    "                    .replace(/'/g, \"'\\\\''\");\n"
    "                const curlText = `curl -X POST \"$${API_BASE_URL}/predict/\" \\\\\n"
    "  -H \"accept: application/json\" \\\\\n"
    "  -H \"Content-Type: application/json\" \\\\\n"
    "  -d '$${payload}'`;\n"
    "                curlRequest.textContent = curlText;\n"
    "            }"
)
if old_payload_block in h:
    h = h.replace(old_payload_block, new_payload_block)

# Inject mobile + a11y performance CSS right before </style>.
# Addresses: iOS Safari scroll jank from background-attachment:fixed,
# backdrop-filter GPU cost on phones, stuck-hover on touch, and
# prefers-reduced-motion accessibility. No-op if already present
# (so a rebuilt image with these baked in won't be double-injected).
if 'prefers-reduced-motion' not in h:
    mobile_perf_css = (
        "\n"
        "        /* Mobile + a11y perf overrides (injected at bootstrap) */\n"
        "        @media (prefers-reduced-motion: reduce) {\n"
        "            *, *::before, *::after {\n"
        "                animation-duration: 0.01ms !important;\n"
        "                animation-iteration-count: 1 !important;\n"
        "                transition-duration: 0.01ms !important;\n"
        "                scroll-behavior: auto !important;\n"
        "            }\n"
        "        }\n"
        "        @media (hover: none) {\n"
        "            .form-textarea:hover,\n"
        "            .analysis-panel:hover,\n"
        "            .results-panel:hover,\n"
        "            .sample-btn:hover,\n"
        "            .metric-card:hover,\n"
        "            .curl-code:hover {\n"
        "                box-shadow: inherit;\n"
        "                background: inherit;\n"
        "                border-color: inherit;\n"
        "                transform: none;\n"
        "            }\n"
        "        }\n"
        "        @media (max-width: 768px) {\n"
        "            body { background-attachment: scroll; }\n"
        "            .analysis-panel,\n"
        "            .results-panel {\n"
        "                backdrop-filter: none;\n"
        "                -webkit-backdrop-filter: none;\n"
        "                box-shadow: var(--shadow-sm);\n"
        "            }\n"
        "            .results-panel { min-height: auto; }\n"
        "            .metric-card,\n"
        "            .sample-btn { box-shadow: none; }\n"
        "        }\n"
        "    "
    )
    h = h.replace('    </style>\n</head>', mobile_perf_css + '</style>\n</head>', 1)

# Also bump the textarea font-size to 16px to stop iOS zoom-on-focus.
h = h.replace(
    '.form-textarea {\n            width: 100%;\n            min-height: 140px;\n            padding: 1rem;\n            border: 1px solid var(--rh-border);\n            border-radius: 10px;\n            font-size: 0.875rem;',
    '.form-textarea {\n            width: 100%;\n            min-height: 140px;\n            padding: 1rem;\n            border: 1px solid var(--rh-border);\n            border-radius: 10px;\n            font-size: 16px;',
    1,
)

# Preconnect to font CDNs for faster first paint on mobile networks.
if 'rel="preconnect" href="https://fonts.gstatic.com"' not in h:
    h = h.replace(
        '<link href="https://fonts.googleapis.com/css2',
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '    <link href="https://fonts.googleapis.com/css2',
        1,
    )

if h != h_original:
    fp.write_text(h)
    print("index.html: patched (URL + IIFE + curl + payload + mobile perf)")
else:
    print("index.html: unchanged (likely already fixed upstream)")

# 2) Settings: force CORS to wildcard (regex — tolerates whitespace drift).
#    No-op once the image ships with pydantic-settings + wildcard.
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

# --- systemd unit: OTS listens only on loopback; Caddy fronts it ---
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
  -p 127.0.0.1:8002:8002 -p 127.0.0.1:8080:8080 \
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

# --- Caddy: API routes go to uvicorn, everything else to the frontend ---
cat > /etc/caddy/Caddyfile <<'CADDY'
{
    email ${tls_email}
}

${domain} {
    encode gzip zstd

    # API paths -> uvicorn on 127.0.0.1:8002
    @api path /predict* /health /docs /openapi.json /metrics /tmf-api/*
    reverse_proxy @api 127.0.0.1:8002

    # Everything else -> static frontend on 127.0.0.1:8080
    reverse_proxy 127.0.0.1:8080
}
CADDY

systemctl enable caddy
systemctl restart caddy

echo "=== Waiting for OTS API to become healthy (via loopback) ==="
for i in $(seq 1 60); do
  if curl -fs http://127.0.0.1:8002/health > /dev/null; then
    echo "OTS healthy after $i checks ($(date))"
    break
  fi
  sleep 5
done

echo "=== Bootstrap complete ==="
echo "DNS must point ${domain} -> this EIP before first HTTPS request."
echo "Caddy will auto-issue a Let's Encrypt cert on the first HTTPS hit."
