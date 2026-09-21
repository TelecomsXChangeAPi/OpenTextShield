<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/logo-dark.png">
  <img src="docs/assets/brand/logo-light.png" width="360" alt="OpenTextShield">
</picture>

# OpenTextShield (OTS)

**Open-source spam and phishing detection for SMS. Built on a 104-language model and trained on messages in 20 languages.**

OpenTextShield is a compact classifier (a fine-tuned multilingual BERT, about 180M parameters) that labels a text message as `ham`, `spam` or `phishing` in around 150 ms on a small CPU instance. It runs on your own servers as a REST API, an SMPP proxy in front of your SMSC, or both. No third-party AI service is involved.

[![GitHub Stars](https://img.shields.io/github/stars/TelecomsXChangeAPi/OpenTextShield?style=flat-square)](https://github.com/TelecomsXChangeAPi/OpenTextShield/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-telecomsxchange%2Fopentextshield-blue?style=flat-square&logo=docker)](https://hub.docker.com/r/telecomsxchange/opentextshield)

**Try it:** [ots.telecomsxchange.com](https://ots.telecomsxchange.com) runs the current release.

## Quick start

[![Open Text Shield L - Docker Deployment](https://img.youtube.com/vi/HCaTE63lVws/0.jpg)](https://youtu.be/HCaTE63lVws?si=4D4BYAdtUxkX7wcF)

**Docker** (image is multi-arch: linux/amd64 and linux/arm64, model included):

```bash
docker pull telecomsxchange/opentextshield:latest
docker run -d -p 8002:8002 -p 8080:8080 telecomsxchange/opentextshield:latest
```

**From source** (Python 3.12, about 4 GB RAM):

```bash
git clone https://github.com/TelecomsXChangeAPi/OpenTextShield.git
cd OpenTextShield
python3.12 -m venv ots && source ots/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
./scripts/start.sh          # API on :8002, demo page on :8080
```

Then open:

- Demo page: http://localhost:8080
- API reference (Swagger): http://localhost:8002/docs
- Health: http://localhost:8002/health

## Using the API

```bash
curl -X POST "http://localhost:8002/predict/" \
  -H "Content-Type: application/json" \
  -d '{"text":"Your account has been suspended. Verify now at http://secure-login-check.xyz","model":"ots-mbert"}'
```

```json
{
  "label": "phishing",
  "probability": 0.99996,
  "processing_time": 0.145,
  "model_info": {
    "name": "OTS_mBERT",
    "version": "2.7",
    "architecture": "bert-base-multilingual-cased",
    "author": "TelecomsXChange (TCXC)"
  }
}
```

`text` is 1 to 512 characters. Text is normalised before classification, so full-width, look-alike and leetspeak disguises (`Ｐａｙｐａｌ`, `paypa1`) are classified as the text they imitate.

| Endpoint | Purpose |
|---|---|
| `POST /predict/` | Classify one message |
| `GET /health` | Service status, API version and loaded model version |
| `GET /metrics` | Prometheus metrics: throughput, batch sizes, queue depth, inference time |
| `POST /feedback/`, `GET /feedback/download/{model}` | Report a wrong verdict; download collected feedback as CSV ([details](docs/FEEDBACK_API.md)) |
| `GET /audit/info`, `/audit/logs`, `/audit/stats` | Audit log of classifications, when auditing is enabled |
| `POST /tmf-api/aiInferenceJob`, `GET /tmf-api/aiInferenceJob[/{id}]` | TM Forum TMF922 job interface for operators who integrate that way |

Concurrent requests are coalesced into padded batches by a dynamic batcher, so per-message cost falls as load rises; on a GPU one instance handles hundreds of messages per second.

## How well it works

Current model is **2.7**. Numbers below are from [`evals/REPORT.md`](evals/REPORT.md), measured through the same text normalisation the API applies.

| Benchmark | Messages | Block rate | Phishing recall |
|---|---|---|---|
| UCI SMS Spam Collection (classic spam) | 5,574 | 99.5% | n/a (no phishing class) |
| Mishra & Soni SMS phishing | 5,971 | 99.3% | 6.1% |
| IMC 2025 smishing (modern, multilingual) | 8,007 | 72.4% | 45.7% |
| In-house adversarial suite | 127 | 96.9% | 80.6% |

"Block rate" counts a spam or phishing message as blocked whichever of the two labels it received. UCI and Mishra & Soni overlap the training corpus and serve as regression gates; IMC 2025 is the most independent signal. Full method, caveats and the v2.5 to 2.7 comparison are in the report; release notes for the platform are in [`evals/results/`](evals/results/).

## Deploying

- **One host:** `docker compose up -d` (add `--profile production` for the nginx front). [Deployment quickstart](docs/deployment/DEPLOYMENT_QUICKSTART.md).
- **Hardened image:** `docker build -f Dockerfile.secure -t opentextshield:secure .` (multi-stage, non-root).
- **Several API instances behind nginx:** `docker compose -f deploy/docker-compose.2x.yml up -d` or the 10-instance file. [Guide](docs/deployment/DEPLOY_10_SERVERS.md).
- **AWS:** Terraform for a single EC2 host with TLS via Caddy in [`infra/aws/`](infra/aws/README.md).
- **SMPP:** the proxy in [`src/smpp_interface/`](src/smpp_interface/README.md) accepts SMPP binds, classifies each `submit_sm` through the API and forwards or rejects it according to your rules.

Configuration is by environment variables with the `OTS_` prefix. The ones people usually change:

| Variable | Default | Meaning |
|---|---|---|
| `OTS_ALLOWED_IPS` | `ANY` | Comma-separated client allow-list for the API |
| `OTS_CORS_ORIGINS` | localhost origins | JSON list of allowed browser origins, e.g. `["*"]` |
| `OTS_MAX_BATCH_SIZE` / `OTS_BATCH_WAIT_MS` | `64` / `50` | Dynamic batcher limits |
| `OTS_USE_FP16` | `true` | Half precision on CUDA (ignored on CPU) |
| `OTS_AUDIT_ENABLED` | `true` | Write classification audit logs to `audit_logs/` |
| `OTS_MBERT_MODEL_PATH` | bundled 2.7 | Load a different `.pth`, for example to roll back |
| `OTS_LOG_LEVEL` | `INFO` | Logging verbosity |

Sizing guidance for a 25 messages-per-second deployment is in [`docs/HARDWARE_SPEC_SHEET_25TPS.md`](docs/HARDWARE_SPEC_SHEET_25TPS.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/api_interface/` | FastAPI service: routers, batching, model loading, audit, TMF922 |
| `src/smpp_interface/` | SMPP classification proxy (Node.js) |
| `src/mBERT/` | Model weights, training scripts, datasets, model tests |
| `frontend/` | The demo page, one HTML file |
| `deploy/`, `infra/` | Multi-instance compose files and nginx config; AWS Terraform |
| `evals/`, `benchmark/`, `tests/` | Evaluation harness and results; load benchmarks; API and adversarial tests |
| `docs/` | Guides and reports, indexed in [`docs/README.md`](docs/README.md) |

## Testing

```bash
pytest src/api_interface/tests/                 # API unit tests
cd src/smpp_interface && npm test               # SMPP proxy, offline suites
python src/mBERT/tests/test_sms.py              # model smoke test
python evals/run_eval.py --help                 # benchmark evaluation
```

## Training your own model

Datasets are CSV files with `text,label` columns and labels `ham`, `spam` or `phishing`. The training scripts, dataset tools and the [labelling guide](docs/LABELING_GUIDE.md) live under [`src/mBERT/training/model-training/`](src/mBERT/training/model-training/README.md). Contributions of labelled data in more languages are the most useful thing you can send.

## Contributing

Issues and pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md). The research background is in [docs/RESEARCH.md](docs/RESEARCH.md).

## About

OpenTextShield is built by [TelecomsXChange (TCXC)](https://telecomsxchange.com) and released under the [MIT License](LICENSE).
