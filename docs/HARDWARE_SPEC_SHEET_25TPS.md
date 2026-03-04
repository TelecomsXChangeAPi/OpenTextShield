# OpenTextShield - Hardware Specification Sheet

**Product:** OpenTextShield AI-Powered SMS Classification Engine
**Version:** 2.5 (mBERT Multilingual Model)
**Date:** March 3, 2026
**Prepared by:** TelecomsXChange (TCXC)

---

## Architecture Overview

OpenTextShield uses an **async inference pipeline** that offloads model inference to a dedicated thread pool (`ThreadPoolExecutor`). This keeps the FastAPI event loop responsive and enables concurrent request processing within a single worker process.

**Per-request processing pipeline:**

| Stage | Operation | Time |
|---|---|---|
| 1 | Enhanced preprocessing (Unicode, homoglyph, URL features) | ~2–5 ms |
| 2 | BERT tokenization (encode_plus, max_length=512) | ~1–3 ms |
| 3 | Model inference (12-layer BERT, torch.no_grad) | ~40–150 ms |
| 4 | Post-processing (softmax, label mapping) | < 1 ms |
| 5 | Audit logging (fire-and-forget, non-blocking) | 0 ms* |

*Audit logging runs asynchronously after the response is sent — it does not add to client-perceived latency.*

**Concurrency model:**

| Device | GIL Released? | Concurrent Inferences per Worker | Effective TPS per Worker |
|---|---|---|---|
| NVIDIA CUDA (T4/L4) | Yes | Up to 4 (thread pool) | ~30–50 |
| Apple MPS (benchmarked) | Partial | Up to 4 (thread pool) | ~17 |
| CPU | No (GIL held) | 1 effective | ~7–10 |

**Benchmark reference:** On Apple MPS (M-series), a single worker achieved 17.4 TPS with 10 concurrent requests (1.8x improvement over sequential). CUDA GPUs (T4/L4) release the GIL more completely during kernel execution, so the concurrency multiplier is expected to be higher (~2–3x). Health checks remained responsive at 6.2ms average during concurrent load. On CPU, the GIL limits parallelism so multiple worker processes are needed instead.

---

## 1. Deployment Target

| Parameter | Requirement |
|---|---|
| Throughput | 25 transactions per second (TPS) sustained |
| Maximum Response Latency | < 500 ms per request |
| Daily Lookup Capacity | **2,160,000 lookups/day** (25 TPS × 86,400 sec) |
| Availability | 24/7 continuous operation |
| Virtualization Platform | VMware vSphere / ESXi |

---

## 2. Model Specifications

| Parameter | Value |
|---|---|
| Model Architecture | BERT-base-multilingual-cased (mBERT) |
| Total Parameters | ~110 million |
| Model File Size | 679 MB (.pth format) |
| Runtime Memory per Instance | ~2 GB |
| Supported Languages | 104+ languages |
| Classification Labels | ham, spam, phishing |
| Max Input Length | 512 tokens |
| Framework | PyTorch 2.7.1 |
| GPU Acceleration | NVIDIA CUDA (recommended), CPU fallback supported |

---

## 3. Hardware Configurations

### Option A — Single VM with GPU (Recommended)

Best for: Lowest latency, simplest architecture, lowest resource footprint.

| Resource | Specification |
|---|---|
| **vCPUs** | 4 vCPUs (Intel Xeon Scalable 3rd Gen+ or AMD EPYC 7003+) |
| **RAM** | 8 GB DDR4 ECC |
| **GPU** | 1× NVIDIA T4 (16 GB VRAM) or NVIDIA L4 (24 GB VRAM) |
| **Storage** | 40 GB SSD (NVMe preferred) |
| **Network** | 1 Gbps virtual NIC (vmxnet3) |
| **OS** | Ubuntu 24.04 LTS (Server) |
| **VMware Requirement** | GPU Passthrough (DirectPath I/O) or NVIDIA vGPU |

**Performance Profile:**

| Metric | Expected Value |
|---|---|
| Per-request latency (end-to-end) | 45 – 85 ms |
| Sustained throughput | 30 – 50 TPS |
| Headroom above 25 TPS target | ~20 – 100% |
| Uvicorn workers | 1 (use 2 for additional headroom) |
| Inference threads per worker | 4 (ThreadPoolExecutor) |
| Model memory in VRAM | ~2 GB of 16 GB available |

**Why 1 worker is sufficient:** CUDA releases Python's GIL during GPU kernel execution. The 4-thread inference pool allows concurrent forward passes to overlap on the GPU. Benchmarked at 17.4 TPS on Apple MPS (1.8x over sequential); CUDA T4/L4 expected to achieve 30–50 TPS due to more complete GIL release. For production headroom, use `--workers 2`.

**VMware GPU Notes:**
- Requires ESXi 7.0+ with GPU passthrough enabled
- Compatible GPUs: NVIDIA T4, L4, A2, A10, A30
- NVIDIA vGPU licensing is an alternative to full passthrough
- Install NVIDIA CUDA drivers inside guest VM

---

### Option B — Single VM, CPU-Only (No GPU Required)

Best for: Environments without GPU passthrough capability.

| Resource | Specification |
|---|---|
| **vCPUs** | 16 vCPUs (Intel Xeon Scalable 3rd Gen+ or AMD EPYC 7003+) |
| **RAM** | 24 GB DDR4 ECC |
| **GPU** | None required |
| **Storage** | 40 GB SSD (NVMe preferred) |
| **Network** | 1 Gbps virtual NIC (vmxnet3) |
| **OS** | Ubuntu 24.04 LTS (Server) |

**Performance Profile:**

| Metric | Expected Value |
|---|---|
| Per-request latency (end-to-end) | 105 – 260 ms |
| Sustained throughput | 28 – 40 TPS |
| Headroom above 25 TPS target | ~12 – 60% |
| Uvicorn workers | 4 |
| Model instances in RAM | 4 (4 × 2 GB = 8 GB model memory) |

**Why 4 workers:** On CPU, Python's GIL prevents true thread-level parallelism for compute-bound inference. Each uvicorn worker is a separate process with its own GIL, so 4 workers = 4 truly parallel inferences. The async thread pool still keeps each worker's event loop responsive for health checks.

**CPU Guidance:**
- Higher single-thread clock speed improves per-request latency
- Minimum recommended: 2.4 GHz base clock
- AVX-512 instruction support beneficial for PyTorch inference
- Hyper-threading should be enabled

---

### Option C — Two VMs with Load Balancer (High Availability)

Best for: Production environments requiring redundancy and failover.

**Application VMs (×2):**

| Resource | Per VM |
|---|---|
| **vCPUs** | 8 vCPUs |
| **RAM** | 12 GB DDR4 ECC |
| **GPU** | None required |
| **Storage** | 30 GB SSD |
| **Network** | 1 Gbps virtual NIC (vmxnet3) |
| **OS** | Ubuntu 24.04 LTS (Server) |

**Load Balancer VM (×1):**

| Resource | Specification |
|---|---|
| **vCPUs** | 2 vCPUs |
| **RAM** | 2 GB DDR4 |
| **Storage** | 10 GB SSD |
| **Network** | 1 Gbps virtual NIC (vmxnet3) |
| **Software** | Nginx (included with OpenTextShield) |
| **Algorithm** | Least Connections |

**Performance Profile:**

| Metric | Expected Value |
|---|---|
| Per-request latency (end-to-end) | 105 – 210 ms |
| Sustained throughput | 28 – 40 TPS |
| Headroom above 25 TPS target | ~12 – 60% |
| Uvicorn workers per VM | 2 |
| Total model instances | 4 (2 per VM) |
| Fault tolerance | 1 VM can fail; remaining VM handles ~14–20 TPS |

**Total Physical Resources (Option C):**

| Resource | Total |
|---|---|
| vCPUs | 18 |
| RAM | 26 GB |
| Storage | 70 GB |
| VMs | 3 |

---

## 4. Configuration Summary

| Option | vCPUs | RAM | GPU | Latency | TPS Capacity | Redundancy | Complexity |
|---|---|---|---|---|---|---|---|
| **A: GPU** | 4 | 8 GB | 1× T4/L4 | 45–85 ms | 30–50 | None | Low |
| **B: CPU** | 16 | 24 GB | None | 105–260 ms | 28–40 | None | Low |
| **C: HA** | 18 | 26 GB | None | 105–210 ms | 28–40 | Yes | Medium |

---

## 5. Software Stack

| Component | Version | Purpose |
|---|---|---|
| Ubuntu Server | 24.04 LTS | Operating system |
| Python | 3.12.x | Runtime |
| PyTorch | 2.7.1 | Model inference engine |
| Transformers | 4.53.0+ | Tokenizer and model loading |
| FastAPI | Latest | REST API framework |
| Uvicorn | Latest | ASGI application server |
| Nginx | Latest | Load balancer (Option C only) |
| Docker | 24.x+ (optional) | Container deployment |
| NVIDIA CUDA | 12.x (Option A only) | GPU acceleration |

---

## 6. Network & Firewall Requirements

| Port | Protocol | Direction | Purpose |
|---|---|---|---|
| 8002 | TCP | Inbound | API endpoint (HTTP REST) |
| 8080 | TCP | Inbound | Web frontend (optional) |
| 443 | TCP | Outbound | Package downloads during setup |
| 80 | TCP | Outbound | Package downloads during setup |

**API Endpoint:** `POST /predict/`
**Health Check:** `GET /health`
**API Documentation:** `GET /docs` (Swagger UI)

---

## 7. Capacity Planning

### Daily Volume at Various TPS Levels

| TPS | Hourly | Daily | Monthly (30d) |
|---|---|---|---|
| 10 | 36,000 | 864,000 | 25,920,000 |
| **25** | **90,000** | **2,160,000** | **64,800,000** |
| 50 | 180,000 | 4,320,000 | 129,600,000 |
| 100 | 360,000 | 8,640,000 | 259,200,000 |

### Scaling Path

**GPU Scaling (per VM with NVIDIA T4/L4):**

| VMs | Workers | Approx. Max TPS | Daily Capacity |
|---|---|---|---|
| 1 | 1 | 30–50 | 2.6M – 4.3M |
| 2 | 2 | 60–100 | 5.2M – 8.6M |
| 4 | 4 | 120–200 | 10.4M – 17.3M |

**CPU-Only Scaling (per VM added, 4 workers each):**

| VMs | Workers | Approx. Max TPS | Daily Capacity |
|---|---|---|---|
| 1 | 4 | 28–40 | 2.4M – 3.5M |
| 2 | 8 | 56–80 | 4.8M – 6.9M |
| 4 | 16 | 112–160 | 9.7M – 13.8M |
| 10 | 40 | 280–400 | 24.2M – 34.6M |

Throughput scales approximately linearly with the number of VM instances.

---

## 8. Deployment Steps (Quick Start)

```bash
# 1. Install system dependencies
sudo apt update && sudo apt install -y python3.12 python3.12-venv git curl

# 2. Clone repository
git clone https://github.com/TelecomsXChangeAPi/OpenTextShield.git
cd OpenTextShield

# 3. Create virtual environment
python3.12 -m venv ots
source ots/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Start the server (adjust --workers based on chosen option)
#    Option A (GPU):  1 worker  (thread pool handles concurrency)
#    Option B (CPU):  4 workers (separate processes for parallelism)
uvicorn src.api_interface.main:app --host 0.0.0.0 --port 8002 --workers 4

# 5. Verify deployment
curl -X POST "http://localhost:8002/predict/" \
  -H "Content-Type: application/json" \
  -d '{"text":"Congratulations! You won a free prize!","model":"ots-mbert"}'
```

**Docker Alternative:**
```bash
docker-compose up -d
```

---

## 9. Monitoring & Health Checks

| Endpoint | Method | Expected Response |
|---|---|---|
| `/health` | GET | `200 OK` with model status |
| `/docs` | GET | Swagger API documentation |
| `/predict/` | POST | Classification result with timing |

**Recommended Monitoring:**
- Response time per request (alert if > 500 ms)
- Requests per second (alert if sustained > 80% of capacity)
- CPU and memory utilization per VM
- GPU utilization % (Option A — alert if sustained > 90%)
- Error rate (alert if > 1%)
- Model load status on application startup
- Thread pool queue depth (alert if inference requests are queuing)

---

## 10. Support & Contact

| Item | Detail |
|---|---|
| Product | OpenTextShield |
| Vendor | TelecomsXChange (TCXC) |
| Repository | https://github.com/TelecomsXChangeAPi/OpenTextShield |
| API Docs | http://<server-ip>:8002/docs |
| Model Version | OTS mBERT v2.5 |

---

*This specification is based on performance benchmarks conducted with OpenTextShield v2.5 using the async inference pipeline. Benchmark data collected on Apple MPS (M-series); CUDA estimates are extrapolated. Actual performance may vary based on VMware host configuration, resource contention, network conditions, and workload characteristics. Benchmark validation on target hardware is recommended prior to production deployment.*
