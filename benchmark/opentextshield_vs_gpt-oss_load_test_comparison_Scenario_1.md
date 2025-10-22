# Load Testing Results: OpenTextShield mBERT vs GPT-OSS-20B (Scenario 1)
*Report Generated: 2025-10-21*

## Executive Summary

This report presents a comprehensive performance comparison between **OpenTextShield mBERT v2.5** and **GPT-OSS-20B** for SMS spam and phishing detection. The testing reveals significant performance differences, with OpenTextShield demonstrating production-ready capabilities for high-throughput SMS processing.

## Test Environment

* **Platform**: M4 Mac mini (Apple Silicon)
* **Hardware**:

  * Model Name: Mac mini
  * Model Identifier: Mac16,11
  * Model Number: MCX44VC/A
  * Chip: Apple M4 Pro
  * Total Cores: 12 (8 performance, 4 efficiency)
  * Memory: 24 GB
  * System Firmware Version: 13822.1.2
  * OS Loader Version: 13822.1.2
  * Serial Number (system): GDW0VPQWVP
  * Hardware UUID: 1949F3DF-C8EB-58FB-87F0-8700998B9A1C
  * Provisioning UDID: 00006040-000E08890C00801C
  * Activation Lock Status: Enabled
* **Test Dataset**: 60 SMS samples
* **OpenTextShield**: FastAPI server on `localhost:8002`
* **GPT-OSS**: LM Studio/OpenAI-compatible API on `0.0.0.0:1234`

  * **Runtime**: **LM Studio Version 0.3.30 (0.3.30)**
  * **Model Host**: **GPT-OSS-20B running on LM Studio, no vLLM used**; model **consuming all GPUs and loaded across all GPUs**
* **Test Date**: 2025-10-21

## Performance Comparison

| Metric                           | OpenTextShield mBERT       | GPT-OSS-20B                  | Performance Ratio |
| -------------------------------- | -------------------------- | ---------------------------- | ----------------- |
| **Single Request Response Time** | 0.299s (median)            | 31.79s                       | **gpt is 106× slower**   |
| **Peak Throughput (req/s)**      | 32.55                      | 0.031                        | **gpt is 1047× lower**   |
| **Concurrent Request Handling**  | Excellent (400 concurrent) | Poor (fails at 2 concurrent) |                   |
| **Success Rate**                 | 100%                       | 0% (concurrent)              |                   |
| **Model Size**                   | ~110M parameters           | 20B parameters               | **gpt is 181× larger**   |
| **Resource Efficiency**          | High                       | Low                          |                   |

## Detailed Test Results

### OpenTextShield mBERT Performance

#### Test Scenarios

* **Light Load** (10 concurrent, 100 requests): 17.42 req/s, 0.525s avg response
* **Medium Load** (50 concurrent, 500 requests): 32.55 req/s, 1.003s avg response
* **Heavy Load** (100 concurrent, 1000 requests): 31.44 req/s, 1.818s avg response
* **SMSC Realistic** (100 concurrent, 10s duration): 31.02 req/s, 1.910s avg response
* **SMSC Peak** (400 concurrent, 5s duration): 31.58 req/s, 6.829s avg response

#### Key Metrics

* **Model Processing Time**: ~30–60ms consistently
* **Reliability**: 100% success rate across all scenarios
* **Latency Distribution**:

  * P95 Response Time: 3.137s (heavy load)
  * P99 Response Time: 3.243s (heavy load)
    
* **Resource Utilization**: Efficient CPU usage with minimal overhead

> **Bottleneck Note:** OpenTextShield mBERT reached a CPU-only bottleneck of **~32–34 req/s** due to CPU processing limits on the M4 Mac mini—primarily from the single-threaded nature of mBERT inference and available CPU resources constraining concurrent request handling despite the model’s inherent efficiency.

### GPT-OSS-20B Performance

#### Test Scenarios

* **Single Request**: 31.79s response time, 0.031 req/s throughput
* **Concurrent Requests**: Failed completely (0% success rate at 2 concurrent)

#### Key Metrics

* **Response Time**: 31.79s for single request
* **Throughput**: 0.031 requests/second
* **Concurrency**: Cannot handle multiple simultaneous requests
* **Response Format**: Returns proper JSON but wrapped in markdown code blocks
* **Runtime Context**: Model served via **LM Studio 0.3.30** (no vLLM), configured to use and saturate all available GPUs

## Technical Analysis

### Architectural Differences

#### OpenTextShield mBERT

* **Model Type**: Specialized multilingual BERT for SMS classification
* **Parameters**: ~110M
* **Training**: Domain-specific SMS spam/phishing datasets
* **Optimization**: Apple Silicon MLX acceleration available
* **Architecture**: Efficient transformer optimized for classification

#### GPT-OSS-20B

* **Model Type**: General-purpose large language model
* **Parameters**: 20B
* **Training**: Broad web text, adapted via prompting
* **Optimization**: Standard transformer implementation (generation-focused)
* **Architecture**: Massive model designed for text generation, not classification

### Performance Gap Analysis

* **Response Time**: 0.299s vs 31.79s (**~106× slower** for GPT-OSS)
* **Throughput**: 32.55 req/s vs 0.031 req/s (**~1047× lower** for GPT-OSS)
* **Concurrency**: mBERT stable up to 400 concurrent; GPT-OSS fails at 2 concurrent due to sequential/compute constraints

## Production Implications

### Use Cases

**OpenTextShield mBERT — Recommended For**

* High-throughput SMS filtering in telecom networks
* Real-time spam/phishing detection
* SMSC integration
* Enterprise-grade API deployments
* Cost-sensitive, high-volume processing

**GPT-OSS-20B — Suitable For**

* Research/analysis of suspicious messages
* Low-volume, high-accuracy classification
* Offline processing of flagged messages
* QA and manual review workflows

### Cost-Benefit Analysis

* **Computational Efficiency**: mBERT uses ~181× fewer parameters and yields ~1000× higher throughput on this setup
* **Power Consumption**: mBERT materially more energy-efficient for classification
* **Scalability**: mBERT scales linearly with additional CPU/GPU resources; GPT-OSS constrained by heavy generation costs

## Discoveries & Next Steps

1. **Batching & MLX/ANE Acceleration (Not yet tested):**
   Implement **batch processing** to handle multiple requests per inference and enable **MLX GPU/ANE acceleration** to leverage the M4 Mac mini’s Neural Engine. Expected uplift: **~5–10× throughput** (estimate).

2. **Horizontal Scaling with Containers (Proven pattern available):**
   Run **multiple Docker containers** of OpenTextShield and **load-balance** across them to overcome the single-process bottleneck, enabling near-linear throughput gains up to hardware saturation. Evidence: existing `launch_multiple_containers.sh` in codebase supports multi-container orchestration.

> Together, (1) batching + MLX/ANE and (2) horizontal scaling should push well beyond the current ~32–34 req/s ceiling on this hardware.

## Recommendations

### For Telecom Operators

1. **Primary SMS Filtering:** Deploy OpenTextShield mBERT for real-time classification.
2. **Quality Assurance:** Possible to use GPT-OSS-20B for secondary, investigative analysis of flagged traffic (post-processing)
3. **Hybrid Architecture:** Integrate both for optimal speed and investigative depth.

### For Implementation

1. **OpenTextShield:** Production-ready with Docker; enable autoscaling and per-route circuit-breakers.
2. **Batching/Acceleration:** Add request batching and MLX/ANE acceleration flags; benchmark gains. (for deployment on apple hardware)
3. **Horizontal Scale:** Use `launch_multiple_containers.sh` (or equivalent IaC) with a reverse proxy (e.g., Nginx/Traefik) for load balancing.
4. **Monitoring:** Capture latency histograms, saturation, and error budgets; alert on tail latencies (P95/P99).

## Conclusion

The load testing demonstrates that **OpenTextShield mBERT significantly outperforms GPT-OSS-20B** for SMS classification tasks requiring high throughput and low latency. With ~1000× higher throughput and ~100× lower latency on this setup, OpenTextShield is the clear choice for production SMS filtering. GPT-OSS-20B remains useful as a secondary, investigatory tool rather than for inline, high-volume processing.

## Test Data Files

* OpenTextShield Results: `results_load_test_opentextshield.json`
* GPT-OSS Results: `results_load_test_gpt-oss.json`
* Benchmark Accuracy: `results_opentextshield.json`, `results_gpt_oss.json`

---

*Report Generated: 2025-10-21*
*Testing Platform: M4 Mac mini (Apple M4 Pro, 24 GB)*
*OpenTextShield Version: 2.5.0*
*GPT-OSS Version: 20B*
*Serving Runtime for GPT-OSS: LM Studio 0.3.30 (no vLLM), all GPUs utilized*
