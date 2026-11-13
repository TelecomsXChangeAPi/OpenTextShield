#!/usr/bin/env python3
"""Score TypeSafe on evals/datasets/bench100_v1.csv with the labeling-guide prompt.

Writes evals/results/predictions_bench100_v1_typesafe.json in the same shape
as run_eval.py's dumps, so the same message set can be compared head to head.
Needs TYPESAFE_API_KEY.
"""
import asyncio, csv, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_audit import QUESTIONS, TYPESAFE_MODEL, REPO_ROOT  # noqa: E402

BENCH = REPO_ROOT / "evals/datasets/bench100_v1.csv"
OUT = REPO_ROOT / "evals/results/predictions_bench100_v1_typesafe.json"


async def main():
    from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy
    rows = list(csv.DictReader(open(BENCH, newline="", encoding="utf-8")))
    out = []
    async with AsyncTypeSafeClient(retry=RetryPolicy(max_retries=4, timeout=60.0)) as client:
        for r in rows:
            t0 = time.time()
            resp = await client.system_one({"sms": r["text"]}, QUESTIONS, model=TYPESAFE_MODEL)
            ms = (time.time() - t0) * 1000
            ans = resp.answers["label"] if hasattr(resp, "answers") else resp["answers"]["label"]
            choice = getattr(ans, "choice", None) or ans["choice"]
            conf = getattr(ans, "confidence", None) or ans["confidence"]
            out.append({"text": r["text"], "gold": r["label"], "pred": choice, "conf": round(float(conf), 4),
                        "latency_ms": round(ms, 1)})
    n = len(out); acc = sum(o["pred"] == o["gold"] for o in out) / n
    json.dump({"model": TYPESAFE_MODEL, "accuracy": acc, "rows": out}, open(OUT, "w"), indent=1, ensure_ascii=False)
    lat = sorted(o["latency_ms"] for o in out)
    print(f"{TYPESAFE_MODEL}: accuracy {acc:.1%} on {n}, median latency {lat[n//2]:.0f} ms; wrote {OUT.relative_to(REPO_ROOT)}")
    for o in out:
        if o["pred"] != o["gold"]:
            print(f"  MISS gold={o['gold']} pred={o['pred']} conf={o['conf']}: {o['text'][:90]}")

asyncio.run(main())
