# OpenTextShield SMPP Interface

**Version 1.0** | **Node.js** | **SMPP v3.4 / v5.0**

---

## Overview

The OpenTextShield (OTS) SMPP Interface is an inline SMS classification proxy. It accepts SMPP connections from client SMSCs, classifies each message using the OTS mBERT AI model, and either forwards legitimate traffic to an upstream SMSC or blocks spam and phishing in real time.

**Primary use case:** Clients whose existing SMSC infrastructure cannot perform HTTP-based classification lookups. The SMPP interface provides a transparent, protocol-native integration point — the client's SMSC sends traffic via standard SMPP, and classified traffic exits via standard SMPP.

---

## System Architecture

### High-Level Data Flow

```
    CLIENT SIDE                        OTS SMPP PROXY                         UPSTREAM SIDE
  ┌─────────────┐                 ┌─────────────────────┐                  ┌──────────────┐
  │             │   bind_trx      │                     │   bind_trx       │              │
  │   Client    │ ───────────────>│    SMPP Server      │ ────────────────>│   Upstream   │
  │   SMSC      │                 │    (port 2775)      │   (N pooled      │   SMSC       │
  │             │   submit_sm     │         │           │    connections)  │              │
  │             │ ───────────────>│         ▼           │                  │              │
  │             │                 │  ┌─────────────┐    │                  │              │
  │             │                 │  │  Extract     │    │                  │              │
  │             │                 │  │  Message     │    │                  │              │
  │             │                 │  │  Text        │    │                  │              │
  │             │                 │  └──────┬──────┘    │                  │              │
  │             │                 │         │           │                  │              │
  │             │                 │         ▼           │                  │              │
  │             │                 │  ┌─────────────┐    │                  │              │
  │             │                 │  │  OTS mBERT  │    │                  │              │
  │             │                 │  │  Classify   │◄───┼──── HTTP POST    │              │
  │             │                 │  │  (API call) │    │    :8002/predict/ │              │
  │             │                 │  └──────┬──────┘    │                  │              │
  │             │                 │         │           │                  │              │
  │             │                 │    ┌────┼────┐      │                  │              │
  │             │                 │    ▼    ▼    ▼      │                  │              │
  │             │                 │  HAM  SPAM PHISH    │                  │              │
  │             │                 │   │    │     │      │                  │              │
  │             │   submit_sm_resp│   │    │     │      │   submit_sm      │              │
  │             │ <───────────────│   │  REJECT  │      │ ────────────────>│              │
  │             │   (error 0x45)  │   │          │      │                  │              │
  │             │                 │   │          │      │   submit_sm_resp │              │
  │             │   submit_sm_resp│   ▼          │      │ <────────────────│              │
  │             │ <───────────────│ FORWARD      │      │   (message_id)   │              │
  │             │   (message_id)  │              │      │                  │              │
  │             │                 │              │      │                  │              │
  │             │   deliver_sm    │    DLR RELAY         │   deliver_sm     │              │
  │             │ <───────────────│ <────────────────────┼──────────────── │              │
  │             │   (DLR)         │  (correlate msg_id) │   (DLR)          │              │
  └─────────────┘                 └─────────────────────┘                  └──────────────┘
```

### Classification Decision Flow

```
  ┌──────────────────────┐
  │  Incoming submit_sm  │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐     YES    ┌────────────────┐
  │  Is message empty?   │ ─────────> │ Forward as-is  │
  └──────────┬───────────┘            └────────────────┘
             │ NO
             ▼
  ┌──────────────────────┐     FAIL   ┌────────────────┐
  │  Call OTS API        │ ─────────> │ Forward as ham  │
  │  POST /predict/      │           │ (fail-open)     │
  └──────────┬───────────┘            └────────────────┘
             │ OK
             ▼
  ┌──────────────────────┐     YES    ┌────────────────┐
  │  Probability below   │ ─────────> │ Treat as ham   │
  │  threshold (0.7)?    │            │ → Forward       │
  └──────────┬───────────┘            └────────────────┘
             │ NO
             ▼
  ┌──────────────────────┐
  │  Apply config rule   │
  │  for detected label  │
  └──────────┬───────────┘
             │
     ┌───────┼───────┐
     ▼       ▼       ▼
  ┌──────┐┌──────┐┌──────┐
  │FORWARD││REJECT││ DROP │
  │      ││      ││      │
  │Send  ││Error ││Fake  │
  │to    ││back  ││OK    │
  │SMSC  ││0x45  ││reply │
  └──────┘└──────┘└──────┘
```

### Upstream Connection Pool

```
                OTS SMPP Proxy                          Upstream SMSC
           ┌──────────────────────┐              ┌───────────────────┐
           │                      │   Session 0   │                   │
           │  ┌────────────────┐  │ ════════════> │                   │
           │  │  Pool Manager  │  │   Session 1   │                   │
           │  │                │  │ ════════════> │    Upstream       │
           │  │  Random select │  │   Session 2   │    SMSC           │
           │  │  per message   │  │ ════════════> │                   │
           │  │                │  │   Session N   │                   │
           │  │  Auto-reconnect│  │ ════════════> │                   │
           │  └────────────────┘  │               │                   │
           └──────────────────────┘              └───────────────────┘

  Config: "connections": 4  →  Opens 4 parallel SMPP binds
  Load balancing: Random session selection per submit_sm
  Resilience: Auto-reconnect on session drop with configurable delay
```

### Scaling Architecture

```
                              ┌─────────────────────┐
                              │  OTS API Instance 1  │
  ┌─────────────┐            │  :8002 (4 workers)   │
  │   Client    │            └──────────┬────────────┘
  │   SMSC      │                       │
  │             │   SMPP    ┌───────┐   │  HTTP (round-robin)
  │             │ ─────────>│  OTS  │───┤
  │             │           │ SMPP  │   │  ┌─────────────────────┐
  │             │ <─────────│ Proxy │───┼──│  OTS API Instance 2  │
  └─────────────┘           │       │   │  │  :8003 (4 workers)   │
                            └───────┘   │  └─────────────────────┘
                                        │
                                        │  ┌─────────────────────┐
                                        └──│  OTS API Instance 3  │
                                           │  :8004 (4 workers)   │
                                           └─────────────────────┘

  Single proxy  →  Multiple API backends  →  Linear TPS scaling
  0ms overhead     ~57ms each (CPU)          3 instances = ~180 TPS
```

---

## Quick Start

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Node.js | >= 14 | SMPP proxy runtime |
| OTS API | Running on `:8002` | mBERT classification endpoint |

### Installation and Startup

```bash
cd src/smpp_interface

# 1. Install dependencies
npm install

# 2. Configure (set upstream host, client credentials, rules)
vi config.json

# 3. Start the proxy
./start.sh
```

### Verify It's Running

```bash
# Check process
cat ots_smpp_proxy.pid

# Check log
tail -f logs/ots_smpp.log

# Check OTS API health
curl http://localhost:8002/health
```

---

## Configuration Reference

All settings live in `config.json`, read once at startup.

### Server Settings

Controls the SMPP server that accepts client connections.

```
  ┌──────────────────────────────────────────────────────────┐
  │  server                                                  │
  │                                                          │
  │  host ─────────── Listen address         (0.0.0.0)       │
  │  port ─────────── Listen port            (2775)          │
  │  system_id ────── Identity in bind_resp  (OTS_SMSC)      │
  │  clients ──────── Allowed credentials    (see below)     │
  │  session_timeout ─ Idle timeout, sec     (120)           │
  │  enquire_link_interval ─ Keepalive, sec  (30)            │
  └──────────────────────────────────────────────────────────┘
```

**Client credentials** are defined as a map of `system_id` to `password`:

```json
"clients": {
    "carrier_alpha": { "password": "s3cure_p@ss" },
    "carrier_beta":  { "password": "an0ther_key" }
}
```

Clients bind using the `system_id` as their username and the corresponding password. Unrecognized credentials receive `ESME_RBINDFAIL` (status 0x0D).

### Upstream Settings

Controls the outbound SMPP connection(s) to the upstream SMSC.

```
  ┌──────────────────────────────────────────────────────────┐
  │  upstream                                                │
  │                                                          │
  │  host ──────────── Upstream SMSC address                 │
  │  port ──────────── Upstream SMSC port    (2775)          │
  │  system_id ─────── Bind username                         │
  │  password ──────── Bind password                         │
  │  connections ───── Pool size (N binds)   (2)             │
  │  interface_version  SMPP version         (5.0 or 3.4)    │
  │  connect_timeout ── TCP + bind wait, sec (5)             │
  │  submit_sm_timeout  Response wait, sec   (30)            │
  │  enquire_link_interval ── Keepalive, sec (30)            │
  │  session_timeout ── Max idle time, sec   (120)           │
  │  reconnect_delay ── Retry wait, sec      (5)             │
  └──────────────────────────────────────────────────────────┘
```

**Key behaviors:**

- **Connection pooling:** The proxy opens `connections` parallel SMPP sessions to the upstream. Each outbound `submit_sm` is routed through a randomly selected session for load distribution.
- **Auto-reconnect:** When a session drops (network error, timeout, or unbind), the proxy automatically reconnects after `reconnect_delay` seconds and re-adds the session to the pool.
- **Keepalive:** The proxy sends `enquire_link` to the upstream every `enquire_link_interval` seconds. If no successful response is received within `session_timeout`, the session is closed and reconnected.

### Classification Settings

Controls the integration with the OTS mBERT classification API.

```
  ┌──────────────────────────────────────────────────────────┐
  │  classification                                          │
  │                                                          │
  │  api_url ────────── Single API endpoint                  │
  │  api_urls ───────── Multiple endpoints (round-robin)     │
  │  model ──────────── Model identifier     (ots-mbert)     │
  │  confidence_threshold ── Min probability (0.7)           │
  │  timeout ────────── HTTP timeout, ms     (5000)          │
  │  rules ──────────── Per-label actions    (see below)     │
  └──────────────────────────────────────────────────────────┘
```

Use `api_url` for a single API instance, or `api_urls` (array) to load-balance across multiple instances. When `api_urls` is set, it takes precedence.

**Classification rules** define what happens to each detected label:

```
  Label        Action     What Happens
  ─────────    ────────   ──────────────────────────────────────────────
  ham          forward    Message is sent to the upstream SMSC.
                          Client receives the upstream's message_id.

  spam         reject     Message is blocked. Client receives
                          submit_sm_resp with the configured
                          error_status (default: 0x45 / 69).

  phishing     reject     Same as spam — blocked with error response.

  (any)        drop       Message is silently discarded. Client
                          receives a fake success response with
                          a synthetic message_id (OTS-XXXXXXXX).
```

**Confidence threshold:** If the model's probability is below `confidence_threshold`, the message is treated as `ham` regardless of the predicted label. This prevents low-confidence false positives from blocking legitimate traffic.

**Fail-open design:** If the OTS API is unreachable or returns an error, the message is forwarded as ham. SMS delivery is never blocked by a classification outage.

### Logging Settings

```
  ┌──────────────────────────────────────────────────────────┐
  │  logging                                                 │
  │                                                          │
  │  file ─── Log file path    (./logs/ots_smpp.log)         │
  │  level ── Verbosity        (info | debug)                │
  └──────────────────────────────────────────────────────────┘
```

Set `level` to `"debug"` for verbose console output during development. Send `SIGHUP` to rotate the log file without restarting.

---

## SMPP Protocol Support

### Supported Operations

```
  PDU                    Direction              Notes
  ─────────────────────  ─────────────────────  ──────────────────────────────
  bind_transceiver       Client → Proxy         Credential-based authentication
  bind_transceiver_resp  Proxy → Client         Returns OTS system_id

  submit_sm              Client → Proxy         Classified, then forwarded/blocked
  submit_sm              Proxy → Upstream       Forwarded ham messages
  submit_sm_resp         Upstream → Proxy       message_id passed to client
  submit_sm_resp         Proxy → Client         Upstream or error response

  deliver_sm (DLR)       Upstream → Proxy       Correlated by message_id
  deliver_sm (DLR)       Proxy → Client         Relayed to originating session

  enquire_link           Bidirectional          Keepalive (both sides)
  unbind                 Bidirectional          Graceful disconnect
```

### Character Encoding Support

```
  data_coding   Encoding       Classification Handling
  ───────────   ────────────   ────────────────────────────────────
  0             GSM 7-bit      Decoded to ASCII, classified as text
  1             ASCII          Direct text extraction
  3             Latin-1        Direct text extraction
  4             Binary         Skipped (forwarded without classification)
  8             UCS-2/UTF-16   Decoded with UCS-2, supports all scripts
```

All encodings are preserved exactly when forwarding to the upstream SMSC. The proxy only decodes text for classification purposes.

### Multipart SMS (UDH)

The proxy fully supports User Data Headers for concatenated SMS:

```
  UDH Format             IE Type   Support
  ─────────────────────  ────────  ──────────────────────────────────
  8-bit concat ref       IE 0x00   Standard multipart (up to 255 parts)
  16-bit concat ref      IE 0x08   Extended multipart (up to 65535 parts)
  UCS-2 + UDH            —         Arabic, CJK, emoji multipart
  Object format {udh}    —         node-smpp native UDH representation
```

**How it works:**

- Each PDU segment is classified independently based on its text content
- The UDH bytes are preserved intact when forwarding — the proxy never modifies UDH
- The `esm_class` UDHI bit (0x40) is passed through transparently

### Delivery Receipt (DLR) Relay

```
  1. Client sends submit_sm with registered_delivery=1
                   │
  2. Proxy forwards to upstream, receives message_id
                   │
  3. Proxy stores mapping:  message_id → client session
                   │
  4. Upstream sends deliver_sm (esm_class=4) with DLR text:
     "id:A1B2C3D4 sub:001 dlvrd:001 ... stat:DELIVRD"
                   │
  5. Proxy extracts message_id from DLR, looks up client session
                   │
  6. Proxy relays deliver_sm to original client
```

**Cross-vendor compatibility:** Some SMSCs return `message_id` in hexadecimal format in `submit_sm_resp` but decimal in the DLR text (or vice versa). The proxy handles this by attempting hex/decimal conversion when a direct lookup fails.

**Cleanup:** Stale message store entries older than 24 hours are automatically purged every hour.

---

## Performance and Scaling

### Baseline Measurements

Measured end-to-end on a single machine (CPU inference, single API worker):

```
  Component                    Avg Latency    Share of Pipeline
  ───────────────────────────  ────────────   ─────────────────
  OTS mBERT Classification    56.7 ms        ~100%
  Upstream SMSC Round-Trip      0.3 ms          ~1%
  SMPP Proxy Overhead           0.0 ms          ~0%
  ─────────────────────────────────────────────────────────────
  Full Pipeline (end-to-end)   55.1 ms        ~18 TPS
```

**The proxy adds zero measurable overhead.** All latency comes from the mBERT model's forward pass. Scaling TPS means scaling the classification API.

### Scaling Strategies

```
  Strategy                        Expected TPS   Configuration Change
  ──────────────────────────────  ────────────   ─────────────────────────────────
  1. More uvicorn workers          ~65 TPS       uvicorn ... --workers 4
  2. Multiple API instances       ~180 TPS       api_urls[] with 3 instances
  3. GPU inference (NVIDIA)       ~300 TPS       Install CUDA-enabled PyTorch
  4. Multiple GPU instances       ~800+ TPS      api_urls[] + GPU workers
```

**Strategy 1** is the simplest — each uvicorn worker runs a separate Python process with its own model copy, bypassing the GIL. Cost: ~440 MB RAM per worker.

**Strategy 2** uses the `api_urls` config to distribute classification requests across multiple OTS API instances via round-robin:

```json
"classification": {
    "api_urls": [
        "http://localhost:8002/predict/",
        "http://localhost:8003/predict/",
        "http://localhost:8004/predict/"
    ]
}
```

**The SMPP proxy does not need to be scaled.** A single Node.js process handles thousands of concurrent SMPP sessions with sub-millisecond overhead.

---

## Operations Guide

### Starting the Proxy

```bash
./start.sh                              # Default config
./start.sh /path/to/custom-config.json  # Custom config
node ots_smpp_proxy.js config.json      # Direct invocation
```

The proxy writes its PID to `ots_smpp_proxy.pid` on startup.

### Stopping the Proxy

```bash
kill $(cat ots_smpp_proxy.pid)          # Graceful shutdown (SIGTERM)
```

### Log Rotation

```bash
kill -HUP $(cat ots_smpp_proxy.pid)     # Close and reopen log file
```

### Monitoring

The proxy writes stats to the log file every 60 seconds:

```
  Metric                  What It Tells You
  ──────────────────────  ──────────────────────────────────────────
  messages_received       Total submit_sm PDUs received from clients
  messages_classified     Successfully classified by OTS API
  messages_forwarded      Sent to upstream (classified as ham)
  messages_rejected       Blocked and returned error to client
  messages_dropped        Silently discarded (drop rule)
  dlrs_forwarded          Delivery receipts relayed to clients
  classification_errors   OTS API failures (fail-open → forwarded)
  active_sessions         Current client SMPP sessions
  upstream_pool_size      Current upstream connections in pool
```

### Health Checks

| Component | How to Check |
|-----------|-------------|
| SMPP Proxy | Attempt `bind_transceiver` to port 2775 |
| OTS API | `curl http://localhost:8002/health` |
| Upstream Pool | Check `upstream_pool_size` in stats log |

---

## Testing

### Test Environment Setup

```
  ┌──────────┐       ┌──────────┐       ┌───────────────┐
  │  Test     │ SMPP  │  OTS     │ HTTP  │  OTS API      │
  │  Client   │──────>│  SMPP    │──────>│  :8002        │
  │  (tests)  │ :2775 │  Proxy   │       │  (mBERT)      │
  └──────────┘       │          │ SMPP  ┌───────────────┐
                      │          │──────>│  Dummy        │
                      └──────────┘ :2776 │  Upstream     │
                                         └───────────────┘
```

```bash
# 1. Start OTS API (from project root)
./scripts/start.sh

# 2. Start dummy upstream
node dummy_upstream.js 2776

# 3. Update config.json upstream to 127.0.0.1:2776

# 4. Start proxy
node ots_smpp_proxy.js config.json

# 5. Run tests
node test_smpp.js           # Basic suite
node test_advanced.js       # Advanced suite
```

### Basic Test Suite — `test_smpp.js` (47 tests)

```
  Category                    Tests   What It Covers
  ──────────────────────────  ─────   ──────────────────────────────────────
  Bind Authentication           4     Valid/invalid credentials, empty creds
  Message Classification        5     Ham, spam, phishing, borderline
  Edge Cases                   11     Empty, long, unicode, binary, injection,
                                      null bytes, long addresses, alphanumeric
                                      sender, Arabic/RTL, single character
  Multiple Concurrent Sessions  2     5 simultaneous sessions
  Enquire Link                  2     Single and rapid keepalive
  Rapid Connect/Disconnect      1     10 rapid cycles
  Sequential Benchmark          1     50 messages, TPS measurement
  Concurrent Burst Benchmark    1     20 simultaneous messages
  Multi-Session Stress          1     3 sessions x 10 messages
  Malformed / Adversarial       4     Garbage bytes, truncated PDU,
                                      zero-length, connection flood
  Multilingual Classification  10     English, Spanish, French, German,
                                      Chinese, Russian, Arabic, Japanese
```

### Advanced Test Suite — `test_advanced.js` (32 tests)

```
  Category                    Tests   What It Covers
  ──────────────────────────  ─────   ──────────────────────────────────────
  UDH / Multipart SMS          7     8-bit concat, 16-bit concat, object
                                      format, UCS-2 + UDH, spam in UDH part
  Emoji & Special Unicode       7     Basic, emoji-only, emoji spam, flags,
                                      skin tones, ZWJ sequences, mixed scripts
  Async Correctness             7     Concurrent classification accuracy,
                                      50-message rapid fire, interleaved
                                      sessions, message_id uniqueness
  DLR Relay                     5     Single DLR, batch DLR, esm_class
                                      verification, delivery status content
  Isolated Benchmarks           6     OTS API only, upstream only, full
                                      pipeline, concurrent burst, component
                                      latency breakdown with percentages
```

---

## File Reference

```
  src/smpp_interface/
  │
  ├── ots_smpp_proxy.js ·········· Main proxy application
  ├── config.json ················ Configuration file
  ├── package.json ··············· Node.js project manifest
  ├── start.sh ··················· Startup script
  │
  ├── node-smpp/ ················· SMPP protocol library (forked)
  │   └── lib/
  │       ├── smpp.js ············ Session, Server, Client classes
  │       ├── pdu.js ············· PDU encoding/decoding
  │       └── defs.js ············ Constants, UDH codec, GSM codec
  │
  ├── logs/ ······················ Log output directory
  │   └── ots_smpp.log ··········· Application log
  │
  ├── test_smpp.js ··············· Basic test suite (47 tests)
  ├── test_advanced.js ··········· Advanced test suite (32 tests)
  └── dummy_upstream.js ·········· Simulated upstream SMSC for testing
```

---

## Production Configuration Example

```json
{
    "server": {
        "host": "0.0.0.0",
        "port": 2775,
        "system_id": "OTS_SMSC",
        "clients": {
            "carrier_alpha": { "password": "prod_secret_1" },
            "carrier_beta":  { "password": "prod_secret_2" }
        },
        "session_timeout": 120,
        "enquire_link_interval": 30
    },
    "upstream": {
        "host": "smsc.carrier.com",
        "port": 2775,
        "system_id": "ots_user",
        "password": "upstream_secret",
        "connections": 4,
        "interface_version": "5.0",
        "connect_timeout": 10,
        "submit_sm_timeout": 30,
        "enquire_link_interval": 30,
        "session_timeout": 120,
        "reconnect_delay": 5
    },
    "classification": {
        "api_urls": [
            "http://localhost:8002/predict/",
            "http://localhost:8003/predict/"
        ],
        "model": "ots-mbert",
        "confidence_threshold": 0.7,
        "timeout": 5000,
        "rules": {
            "ham":      { "action": "forward" },
            "spam":     { "action": "reject", "error_status": 69 },
            "phishing": { "action": "reject", "error_status": 69 }
        }
    },
    "logging": {
        "file": "./logs/ots_smpp.log",
        "level": "info"
    }
}
```
