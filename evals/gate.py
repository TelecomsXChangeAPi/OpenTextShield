#!/usr/bin/env python3
"""Regression gate for a model checkpoint: score the small eval sets and fail below thresholds.

    python evals/gate.py --model src/mBERT/training/model-training/mbert_ots_model_2.9.pth

Thresholds live in evals/gate_thresholds.json. They are set a little under the
2.9 release numbers so seed noise passes and a real regression does not.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
THRESHOLDS = json.load(open(REPO / "evals/gate_thresholds.json"))
SETS = ["csv:evals/datasets/bench100_v1.csv", "csv:evals/datasets/hard_legit_a2p_v1.csv",
        "fable5:evals/datasets/fable5_adversarial_v1_clean.csv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="gate")
    args = ap.parse_args()
    cmd = [sys.executable, str(REPO / "evals/run_eval.py"), "--model", args.model, "--tag", args.tag]
    for s in SETS:
        cmd += ["--dataset", s]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    summary = json.load(open(REPO / f"evals/results/summary_{args.tag}.json"))
    failed = []
    print(f"{'set':22} {'metric':18} {'value':>7} {'min':>7}")
    for key, checks in THRESHOLDS.items():
        m = summary[key]
        ham = m["per_class"]["ham"]
        values = {"accuracy": m["accuracy_3class"], "block_accuracy": m["accuracy_binary_block"],
                  "ham_recall": ham["recall"] if ham["support"] else 1.0,
                  "phishing_recall": m["per_class"]["phishing"]["recall"]}
        for metric, minimum in checks.items():
            v = values[metric]
            flag = "" if v >= minimum else "  FAIL"
            print(f"{key:22} {metric:18} {v:7.1%} {minimum:7.1%}{flag}")
            if v < minimum:
                failed.append(f"{key}.{metric}={v:.3f} < {minimum}")
    if failed:
        print("\nGATE FAILED: " + "; ".join(failed))
        sys.exit(1)
    print("\nGATE PASSED")


if __name__ == "__main__":
    main()
