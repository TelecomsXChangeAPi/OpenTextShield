#!/usr/bin/env python3
"""Render the 2.9 distillation results: 2.7 baseline vs every 2.9 seed vs TypeSafe on the bench.

    python evals/distill_report.py [--version 2.9] [--baseline v2.7-baseline]

Reads evals/results/summary_<tag>.json for the baseline and each seed, and
evals/results/predictions_bench100_v1_typesafe.json for the teacher. Prints
markdown; every metric is mean over seeds with the min-max spread.
"""
import argparse
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "evals/results"
SETS = [("bench100_v1", "bench100 (hand-written, 100)"), ("hard_legit_a2p_v1", "hard_legit A2P (40)"),
        ("fable5", "fable5 clean (71)"), ("distill_v2.9_heldout", "teacher held-out corpus"),
        ("mishra", "Mishra & Soni (5,971)"), ("uci", "UCI SMS Spam (5,574)"), ("imc25", "IMC25 smishing (8,005)")]


def metrics(m):
    ham = m["per_class"]["ham"]
    return {"acc": m["accuracy_3class"], "block": m["accuracy_binary_block"],
            "phish_recall": m["per_class"]["phishing"]["recall"],
            "false_block": (1 - ham["recall"]) if ham["support"] else None}


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.1%}"
    return f"{sum(vals)/len(vals):.1%} ({min(vals):.1%}–{max(vals):.1%})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="2.9")
    ap.add_argument("--baseline", default="v2.7-baseline")
    args = ap.parse_args()
    base = json.load(open(ROOT / f"summary_{args.baseline}.json"))
    seeds = {}
    for p in sorted(glob.glob(str(ROOT / f"summary_v{args.version}-s*.json"))):
        seeds[re.search(r"-s(\d+)", p).group(1)] = json.load(open(p))
    ts = ROOT / "predictions_bench100_v1_typesafe.json"
    ts = json.load(open(ts)) if ts.exists() else None
    print(f"Seeds: {', '.join(seeds) or 'none'}\n")
    print("| Set | Metric | 2.7 | 2.9 mean (min–max) |")
    print("|---|---|---|---|")
    for key, label in SETS:
        if key not in base:
            continue
        b = metrics(base[key])
        s = [metrics(v[key]) for v in seeds.values() if key in v]
        for mk, ml in (("acc", "accuracy"), ("block", "block accuracy"), ("phish_recall", "phishing recall"), ("false_block", "false block rate")):
            if b[mk] is None or (key == "imc25" and mk == "false_block"):
                continue
            print(f"| {label} | {ml} | {b[mk]:.1%} | {fmt([x[mk] for x in s])} |")
    if ts:
        rows = ts["rows"]
        n = len(rows)
        acc = sum(r["pred"] == r["gold"] for r in rows) / n
        ham = [r for r in rows if r["gold"] == "ham"]
        fb = sum(r["pred"] != "ham" for r in ham) / len(ham)
        thr = [r for r in rows if r["gold"] != "ham"]
        tr = sum(r["pred"] != "ham" for r in thr) / len(thr)
        print(f"\nTypeSafe {ts['model']} on the same bench100: accuracy {acc:.1%}, threat recall {tr:.1%}, false block rate {fb:.1%}")
        for r in rows:
            if r["pred"] != r["gold"]:
                print(f"  - miss gold={r['gold']} pred={r['pred']} conf={r['conf']}: {r['text'][:90]}")


if __name__ == "__main__":
    main()
