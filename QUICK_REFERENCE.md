# Quick Reference Card

## One-Line Summary
**10-server deployment = 9.3x faster (1.74s vs 16.28s) = SMSC timeout problem SOLVED ✅**

---

## Deploy Now
```bash
docker compose -f docker-compose.10x.yml up -d
```

## Test It
```bash
curl -X POST "http://localhost:8002/predict/" \
  -H "Content-Type: application/json" \
  -d '{"text":"test","model":"ots-mbert"}'
```

## Check Status
```bash
docker compose -f docker-compose.10x.yml ps
```

---

## The Numbers
| Metric | Value | Status |
|--------|-------|--------|
| **Duration (300 burst)** | 1.74 seconds | ✅ SAFE |
| **Max Latency** | 68.52ms | ✅ FAST |
| **Throughput** | 172 req/s | ✅ EXCELLENT |
| **SMSC Timeout** | 10 seconds | ✅ 8.26s margin |

---

## Files to Know
| File | Why It Matters |
|------|---|
| `docker-compose.10x.yml` | Deploy this |
| `FINAL_REPORT.md` | Read this first |
| `DEPLOYMENT_QUICKSTART.md` | Follow this |

---

## Startup Checklist
- [ ] Run: `docker compose -f docker-compose.10x.yml up -d`
- [ ] Wait: 3-5 minutes for PyTorch download
- [ ] Check: `docker compose -f docker-compose.10x.yml ps` (all healthy)
- [ ] Test: `curl http://localhost:8002/predict/`
- [ ] Verify: Response time < 100ms

---

## Troubleshooting
```bash
# Still starting?
docker compose -f docker-compose.10x.yml logs -f

# Check resources
docker compose -f docker-compose.10x.yml stats

# Restart everything
docker compose -f docker-compose.10x.yml down
docker compose -f docker-compose.10x.yml up -d
```

---

## What's Fixed
- ❌ **Before:** Single server takes 16.28s (exceeds 10s SMSC timeout)
- ✅ **After:** 10 servers take 1.74s (well under 10s timeout)

---

## Key Insight
**GPU is the bottleneck** (not CPU/RAM/network)
- 1 GPU handles ~19 req/s
- 10 servers competing for GPU = 172 req/s
- Linear 9.3x scaling achieved

---

## Performance Baseline
```
Single message:   ~70ms latency, 100% success
300 burst:        1.74 seconds total, 100% success
SMSC compatible:  YES ✅ (1.74s << 10s timeout)
```

---

## Ports
- **Nginx LB:** http://localhost:8002 (USE THIS)
- **API servers:** http://localhost:9001-9010 (debug only)

---

## Don't Forget
1. **Disk space:** Need ~2GB for Docker images
2. **RAM:** Need ~8GB for 10 models running
3. **Startup:** Takes 3-5 minutes
4. **Monitoring:** Watch GPU utilization (should be 90-95%)

---

## Support Resources
- **Quick deployment:** `DEPLOYMENT_QUICKSTART.md`
- **Full details:** `REAL_WORLD_PERFORMANCE_ANALYSIS.md`
- **Navigation:** `TESTING_DOCUMENTATION_INDEX.md`

---

**Status: ✅ Ready to deploy. Go!**
