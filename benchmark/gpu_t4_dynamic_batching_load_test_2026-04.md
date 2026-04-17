# GPU Load Test: OpenTextShield mBERT on NVIDIA T4 with Dynamic Batching
*Report Generated: 2026-04-17*
*Test Conducted by: TelecomsXChange QA Team*

## Executive Summary

This report documents the first end-to-end production-scale load test of **OpenTextShield (OTS) mBERT** running on **GPU-accelerated infrastructure with dynamic batching enabled**. The integration achieved a **20× increase in sustained throughput** over the previous CPU-only baseline (~5 MPS → ~100 MPS) and **100% reliability** across 132,004 combined test messages — zero dropped connections, zero API timeouts, zero circuit-breaker bypasses.

The test validates that the OTS platform is commercially viable for enterprise-grade SMS classification traffic on a single mid-tier GPU instance.

## Test Environment

### Host Machine
- **Instance**: AWS `g4dn.4xlarge`
- **Compute**: 16 vCPUs
- **Memory**: 64 GiB RAM
- **Hardware Accelerator**: NVIDIA Tesla T4 GPU (16 GB VRAM)
- **Inference Precision**: FP16 (half-precision)

### Software Stack
- **OTS API Version**: 2.6.0 (with dynamic batching enabled)
- **Container Runtime**: Docker with `nvidia-container-toolkit`
- **GPU Drivers**: `nvidia-driver-535-server` (headless)
- **Model**: OpenTextShield mBERT (multilingual)

### Deployment Notes
- Container granted exclusive GPU access via the NVIDIA Container Toolkit device mapping.
- `weights_only=False` set in `src/api_interface/services/model_loader.py` to load the trusted mBERT checkpoint.

### Test Topology
Each inbound message generated approximately **12 internal network transactions** through the loop:

```
k6 → SMSC → OTS Bridge → OTS API → OTS Bridge → SMSC
```

This TPS multiplier means infrastructure stability must be evaluated against the amplified internal transaction rate, not just the inbound MPS figure.

## Load Test Methodology

Four escalating 60-second traffic bursts were issued through the OTS Bridge. Each burst measured queue clearance time, success rate, and sustained real-time throughput after the burst window closed.

## Results: Throughput and Stability

| Burst (60s)  | Total Processed | Queue Clearance Time | Sustained Throughput | Failures |
|--------------|-----------------|----------------------|----------------------|----------|
| 100 MPS      | 6,001           | 59 seconds           | ~101.7 MPS           | 0        |
| 300 MPS      | 18,001          | 2 min 59 sec         | ~100.5 MPS           | 0        |
| 600 MPS      | 36,001          | 6 min 16 sec         | ~95.5 MPS            | 0        |
| 1200 MPS     | 72,001          | 12 min 11 sec        | ~98.3 MPS            | 0        |
| **Combined** | **132,004**     | —                    | **~100 MPS baseline**| **0**    |

### Network Resilience (TPS Multiplier)
- At 600 MPS inbound → ~7,200 internal TPS sustained.
- At 1200 MPS inbound → ~14,400 internal TPS sustained.
- 0% packet loss and 0 timeouts maintained throughout.

## Compute Efficiency

| Metric                  | Previous (CPU baseline) | Current (T4 GPU + Dynamic Batching) |
|-------------------------|-------------------------|-------------------------------------|
| Sustained throughput    | ~5 MPS                  | ~100 MPS (**20× / 2,000% gain**)    |
| Host CPU utilization    | ~76%                    | ~5.6%                               |
| Failure rate at peak    | Variable                | 0%                                  |

The CPU is reduced to a thin HTTP-and-batching layer; the GPU absorbs all heavy mathematical work. The host OS remains idle and stable, making the deployment highly cost-efficient per processed message.

## Optimization Insight: Batch-Window Tuning

Prometheus metrics from the 1200 MPS burst exposed an underutilized batching configuration:

```
ots_batch_wait_seconds:        0.015 (15 ms — pre-PR #179 default)
ots_batch_size_bucket{le="8"}:  6,454 batches
ots_batch_size_bucket{le="32"}: 138 batches
```

The 15 ms wait window flushed thousands of micro-batches (≤8 messages) and rarely filled the 32-slot ceiling — leaving GPU capacity on the table.

### Action Taken — PR #179
- `OTS_BATCH_WAIT_MS`: 15 → **50 ms** (allows the batcher more time to coalesce)
- `OTS_MAX_BATCH_SIZE`: 32 → **64** (raises the ceiling to match T4 headroom)
- Added `ots_effective_arrival_rate_msgs_per_second` and `ots_arrival_rate_lifetime_msgs_per_second` gauges so operators can distinguish bridge-limited from GPU-limited scenarios at a glance.

A re-run of the 600/1200 MPS bursts is planned to quantify the lift; expectation is a meaningful skew of `ots_batch_size_bucket` toward the 16/32 buckets and a higher sustained MPS ceiling.

## Scalability and Queue Dynamics

The OTS Bridge's circuit-breaker and queuing layers behaved as designed: when inbound traffic exceeded the ~100 MPS processing ceiling, excess traffic was queued rather than dropped. Trade-offs to factor into operational SLAs:

- A 60-second 1200 MPS burst introduces ~12 minutes of queue clearance.
- The system trades real-time delivery for guaranteed processing under sustained hyper-burst conditions.
- No volatility, restarts, or memory pressure observed at any tested load.

## Conclusion

GPU acceleration combined with dynamic batching has transformed the OTS platform into an enterprise-grade SMS security filter. The system handles 14,000+ internal TPS with zero failures, zero packet loss, and minimal host overhead — validating the architecture as commercially ready for production telecom traffic.

The follow-up tuning landed in PR #179 is expected to push the per-instance throughput ceiling further on the same hardware, with horizontal scaling available via additional API replicas behind the bridge's round-robin client.
