# 10-Server Deployment Quick Start Guide

## TL;DR - One Command to Deploy

```bash
# Deploy 10 API servers + nginx load balancer
docker compose -f docker-compose.10x.yml up -d

# Wait 3-5 minutes for models to load, then test:
curl -X POST "http://localhost:8002/predict/" \
  -H "Content-Type: application/json" \
  -d '{"text":"test message","model":"ots-mbert"}'
```

**Expected Result**: 1.74-1.80 seconds for 300 simultaneous messages (vs 16.28s with single server)

---

## What Gets Deployed

```
┌─────────────────────────────────────────────────────────────┐
│                    Nginx Load Balancer                       │
│                    (Port 8002)                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────┐
        │                                             │
    ┌────┴────┬────┬────┬────┬────┬────┬────┬────┬────┴────┐
    │   API    │API │API │API │API │API │API │API │   API  │
    │Server 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │ 8  │ 9-10  │
    │(9001)    │(9002)...(9010)                             │
    └─────────────────────────────────────────────────────────┘
           Each server loads its own mBERT model
              All 10 compete for GPU (MPS)
```

## Ports

| Service | Port | Access |
|---------|------|--------|
| Nginx Load Balancer | 8002 | `http://localhost:8002` |
| API Server 1 | 9001 | `http://localhost:9001` (direct) |
| API Server 2 | 9002 | `http://localhost:9002` (direct) |
| ... | ... | ... |
| API Server 10 | 9010 | `http://localhost:9010` (direct) |

**Use port 8002** for normal operation (load balanced)
**Use 9001-9010** only for debugging individual servers

## Startup Checklist

```
❌ → ⏳ → ✅

Step 1: Start deployment
  docker compose -f docker-compose.10x.yml up -d

Step 2: Wait for startup (~3-5 minutes)
  Watch container logs:
  docker compose -f docker-compose.10x.yml logs -f

  Wait for lines like:
  api-1-1  | INFO:     Uvicorn running on http://0.0.0.0:8002
  api-2-1  | INFO:     Uvicorn running on http://0.0.0.0:8002
  (repeated for all 10 servers)

Step 3: Verify all servers are healthy
  for i in {1..10}; do
    curl -s http://localhost:900$i/health | jq .
  done

  Expected response:
  {"status":"healthy"}

Step 4: Test through load balancer
  curl -X POST "http://localhost:8002/predict/" \
    -H "Content-Type: application/json" \
    -d '{"text":"test","model":"ots-mbert"}'

  Expected: Classification result with latency <100ms
```

## Monitoring

### Check Container Status
```bash
# See all containers and their health
docker compose -f docker-compose.10x.yml ps

# Expected output: all containers "Up" and "(healthy)"
```

### View Logs
```bash
# All containers
docker compose -f docker-compose.10x.yml logs -f

# Specific server
docker compose -f docker-compose.10x.yml logs -f api-1

# Nginx load balancer
docker compose -f docker-compose.10x.yml logs -f load-balancer
```

### Monitor in Real-Time
```bash
# Watch API responses
watch -n 1 'curl -s http://localhost:8002/predict/ -d "{\"text\":\"test\"}" \
  -H "Content-Type: application/json" | jq .'

# Monitor GPU usage
watch -n 1 'nvidia-smi'  # CUDA
# or
watch -n 1 'top -l 1 | grep mps'  # Apple Silicon
```

## Performance Verification

### Quick Test (5 requests)
```bash
for i in {1..5}; do
  time curl -X POST "http://localhost:8002/predict/" \
    -H "Content-Type: application/json" \
    -d '{"text":"test message","model":"ots-mbert"}' \
    -s | jq .
done
```

### Load Test (100 concurrent requests)
```bash
# Install Apache Bench if needed
ab -n 100 -c 10 -p payload.json \
  -T application/json \
  http://localhost:8002/predict/
```

### Expected Latencies
```
✅ Good:    < 100ms
⚠️  Warning: 100-500ms
❌ Problem: > 500ms

With 10 servers, expect: 70-90ms average latency
```

## Troubleshooting

### Servers Not Starting (Taking >5 minutes)
```bash
# Check logs for errors
docker compose -f docker-compose.10x.yml logs api-1

# Common issues:
# 1. Low disk space (PyTorch is 104MB per container × 10)
# 2. Low memory (need ~8GB for 10 models)
# 3. Network issues (downloading PyTorch)

# Solution: Stop, clean up, and restart
docker compose -f docker-compose.10x.yml down
docker system prune -f
docker compose -f docker-compose.10x.yml up -d
```

### Nginx Returning 502 (Bad Gateway)
```bash
# Servers are still starting. Wait a bit longer and retry:
sleep 30
curl http://localhost:8002/health

# If still failing, check if servers are running:
docker compose -f docker-compose.10x.yml ps

# If containers are down, check startup errors:
docker compose -f docker-compose.10x.yml logs api-1 | tail -100
```

### High Latency (>500ms)
```bash
# Check if GPU is available
docker exec opentextshield-api-1-1 python -c "import torch; print(torch.backends.mps.is_available())"

# Check if model is loaded
docker exec opentextshield-api-1-1 python -c "from src.services.model_manager import model_manager; print(model_manager.models)"

# Check memory usage
docker compose -f docker-compose.10x.yml stats
```

### Uneven Load Distribution
```bash
# Monitor nginx load balancer
tail -f /var/log/nginx/access.log | awk '{print $3}' | sort | uniq -c

# Expected: roughly equal requests to each backend
#           api-1:8002 ~30 requests
#           api-2:8002 ~30 requests
#           ...etc
```

## Scaling Up/Down

### Add More Servers (if using Kubernetes)
```bash
# Scale to 20 servers
kubectl scale deployment opentextshield --replicas=20

# Nginx automatically finds new servers
```

### Stop All Servers
```bash
docker compose -f docker-compose.10x.yml down

# Verify they're stopped
docker compose -f docker-compose.10x.yml ps
```

### Stop Individual Server (for maintenance)
```bash
# Stop server 3
docker compose -f docker-compose.10x.yml stop api-3

# Restart it
docker compose -f docker-compose.10x.yml start api-3

# Nginx automatically routes around failed servers
```

## Performance Baseline

After deployment completes and all servers are healthy:

```
Single Message:
  - Latency: ~70ms
  - Success Rate: 100%

300 Simultaneous Messages:
  - Total Duration: 1.74 seconds
  - Throughput: 172 req/s
  - Max Latency: 68.52ms
  - SMSC Timeout Risk: None ✅

vs Single Server (for comparison):
  - Total Duration: 16.28 seconds
  - Throughput: 18.43 req/s
  - Max Latency: 260ms
  - SMSC Timeout Risk: HIGH ❌
```

## Files Reference

| File | Purpose |
|------|---------|
| `docker-compose.10x.yml` | Deploy configuration |
| `nginx.conf` | Load balancer configuration |
| `REAL_WORLD_PERFORMANCE_ANALYSIS.md` | Detailed technical analysis |
| `TEST_vs_REAL_SUMMARY.md` | Simulation vs real comparison |
| `DEPLOY_10_SERVERS.md` | Full deployment guide |
| `burst_test_real_10x_docker.py` | Test script for real deployment |

## Support

If something goes wrong:

1. **Check logs**: `docker compose -f docker-compose.10x.yml logs -f`
2. **Check status**: `docker compose -f docker-compose.10x.yml ps`
3. **Check system resources**: `docker compose -f docker-compose.10x.yml stats`
4. **Restart everything**:
   ```bash
   docker compose -f docker-compose.10x.yml down
   docker system prune -f
   docker compose -f docker-compose.10x.yml up -d
   ```

---

**Ready to deploy?**
```bash
docker compose -f docker-compose.10x.yml up -d
```

See you in 3-5 minutes when your system is ready for production load! 🚀
