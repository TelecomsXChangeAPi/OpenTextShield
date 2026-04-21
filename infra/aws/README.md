# OpenTextShield — Minimal AWS Deployment

Single-node t3.medium deployment in `ca-central-1`. Mirrors the
`tcxc-crm/infra/` pattern: auto-generated SSH key, default VPC,
Elastic IP, Ubuntu 22.04.

## What gets created

| Resource | Value |
|---|---|
| EC2 instance | `t3.medium` (2 vCPU, 4 GB RAM) |
| AMI | Ubuntu 22.04 LTS (Canonical, latest) |
| EBS | 30 GB gp3, encrypted |
| Swap | 2 GB file, auto-mounted |
| Elastic IP | persistent public IP |
| Security Group | 22 (SSH), 8002 (API), 8080 (frontend) |

Monthly cost (on-demand, ca-central-1): ~**$33/mo** (compute $30 + disk $2.40).

## First-time deploy

```bash
cd infra/aws
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars if you want to restrict SSH

terraform init
terraform apply
```

Apply takes ~2 min to provision EC2 + EIP. The server then spends a
further ~5–10 min in `user-data.sh` doing: apt upgrade → install Docker
→ pull OTS image → start systemd unit → load mBERT model. Watch with:

```bash
ssh -i deploy-key.pem ubuntu@<EIP> 'tail -f /var/log/user-data.log'
```

## Endpoints (replace `<EIP>` with the output value)

| Purpose | URL |
|---|---|
| **Programmable inference** | `POST http://<EIP>:8002/predict/` |
| Swagger UI | `http://<EIP>:8002/docs` |
| TMF922 async jobs | `POST http://<EIP>:8002/tmf-api/aiInferenceJob` |
| Health check | `GET http://<EIP>:8002/health` |
| Prometheus metrics | `GET http://<EIP>:8002/metrics` |
| Research frontend | `http://<EIP>:8080` |

Example request:
```bash
curl -X POST "http://<EIP>:8002/predict/" \
  -H "Content-Type: application/json" \
  -d '{"text":"Your SMS content here","model":"ots-mbert"}'
```

## DNS

Point an A record to the `server_ip` output value, e.g.:

```
ots.telecomsxchange.com.   300   IN   A   <EIP>
```

The EIP is stable across instance restarts. If you want TLS, put
Caddy/nginx in front (not included in this minimal setup).

## Operations

```bash
# SSH
ssh -i deploy-key.pem ubuntu@<EIP>

# Service status / logs
sudo systemctl status ots
sudo journalctl -u ots -f

# Restart service
sudo systemctl restart ots

# Update to a new image
sudo docker pull telecomsxchange/opentextshield:latest
sudo systemctl restart ots

# Tear down everything
terraform destroy
```

## Sizing notes

`t3.medium` is the practical floor. mBERT weights are 679 MB on disk,
working set is ~1.5–2 GB under load. `t3.small` (2 GB RAM) will OOM
during model load. For higher throughput, bump to `t3.large` (8 GB) or
`c7i.large` (compute-optimised, same RAM but faster CPU).
