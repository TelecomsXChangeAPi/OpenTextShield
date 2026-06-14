#!/usr/bin/env python3
"""
Per-class logit-bias calibration for the OpenTextShield mBERT classifier.

The model collapses much of `phishing` into `spam` (see evals/REPORT.md). A cheap,
training-free fix is to add a small additive bias to each class's logit before the
argmax: nudging the phishing logit up recovers phishing recall, and the spam bias
can be tuned to keep the spam/ham balance. This searches a small grid of (spam,
phishing) biases on a held-out CSV and reports the combination that maximises a
chosen objective, plus the before/after metrics.

The chosen bias vector is written to JSON and can be passed straight to
run_eval.py via --logit-bias to verify the gain on the public benchmarks, or
applied in production by adding it to the logits before argmax.

Objectives:
  macro_f1     - balanced 3-class quality (default)
  block_then_f1 - maximise block accuracy first, break ties by macro-F1
  phishing_recall - maximise phishing recall (use with care; can hurt spam)

Usage:
  python evals/calibrate_thresholds.py \
    --model src/mBERT/training/model-training/mbert_ots_model_2.7.pth \
    --data  evals/datasets/mishra_soni_5971.csv:mishra \
    --out   evals/results/calibration.json
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

REPO_ROOT = Path(__file__).resolve().parent.parent
VOCAB_FILE = REPO_ROOT / "evals/assets/bert-base-multilingual-cased-vocab.txt"
LABELS = ["ham", "spam", "phishing"]
MAX_LEN = 96
csv.field_size_limit(10 * 1024 * 1024)

# Reuse the dataset loaders from the eval harness so calibration and evaluation
# read data identically. Load run_eval by file path via importlib rather than
# mutating sys.path — the latter is import-order-dependent and breaks when this
# module is imported from another working directory.
import importlib.util

_run_eval_spec = importlib.util.spec_from_file_location(
    "ots_run_eval", str(REPO_ROOT / "evals" / "run_eval.py")
)
_run_eval = importlib.util.module_from_spec(_run_eval_spec)
_run_eval_spec.loader.exec_module(_run_eval)
LOADERS = _run_eval.LOADERS


def load_logits(model_path, samples, device="cpu", batch_size=32):
    tok = BertTokenizerFast(vocab_file=str(VOCAB_FILE), do_lower_case=False)
    model = BertForSequenceClassification(BertConfig(vocab_size=119547, num_labels=3))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval().to(device)
    texts = [s["text"] for s in samples]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    logits = [None] * len(texts)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt").to(device)
            out = model(**enc).logits.float().cpu()
            for j, i in enumerate(idx):
                logits[i] = out[j]
    return torch.stack(logits)


def metrics_for(logits, golds, bias):
    preds = (logits + torch.tensor(bias)).argmax(dim=1).tolist()
    n = len(golds)
    acc = sum(p == g for p, g in zip(preds, golds)) / n
    block = sum((g != 0) == (p != 0) for p, g in zip(preds, golds)) / n
    # macro-F1
    f1s = []
    for c in range(3):
        tp = sum(p == c and g == c for p, g in zip(preds, golds))
        fp = sum(p == c and g != c for p, g in zip(preds, golds))
        fn = sum(p != c and g == c for p, g in zip(preds, golds))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    macro_f1 = sum(f1s) / 3
    gold2 = sum(g == 2 for g in golds)
    phish_recall = sum(p == 2 and g == 2 for p, g in zip(preds, golds)) / gold2 if gold2 else 0.0
    return {"accuracy": round(acc, 4), "block_accuracy": round(block, 4),
            "macro_f1": round(macro_f1, 4), "phishing_recall": round(phish_recall, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="path:loader (e.g. evals/datasets/mishra_soni_5971.csv:mishra)")
    ap.add_argument("--out", default=str(REPO_ROOT / "evals/results/calibration.json"))
    ap.add_argument("--objective", choices=["macro_f1", "block_then_f1", "phishing_recall"],
                    default="macro_f1")
    ap.add_argument("--grid", type=float, default=2.0, help="max abs bias to search")
    ap.add_argument("--steps", type=int, default=9, help="grid points per class")
    args = ap.parse_args()

    path, loader = args.data.rsplit(":", 1)
    samples = LOADERS[loader](Path(path))
    golds = [LABELS.index(s["gold"]) for s in samples]
    print(f"Loaded {len(samples)} samples via '{loader}'")

    logits = load_logits(args.model, samples)
    baseline = metrics_for(logits, golds, [0.0, 0.0, 0.0])
    print(f"Baseline (no bias): {baseline}")

    grid = [round(-args.grid + i * (2 * args.grid) / (args.steps - 1), 3) for i in range(args.steps)]
    best = None
    for sb in grid:
        for pb in grid:
            m = metrics_for(logits, golds, [0.0, sb, pb])
            if args.objective == "macro_f1":
                key = (m["macro_f1"],)
            elif args.objective == "block_then_f1":
                key = (m["block_accuracy"], m["macro_f1"])
            else:
                key = (m["phishing_recall"], m["macro_f1"])
            if best is None or key > best["key"]:
                best = {"key": key, "bias": [0.0, sb, pb], "metrics": m}

    print(f"Best bias (ham/spam/phishing): {best['bias']}")
    print(f"Calibrated: {best['metrics']}")
    result = {"model": args.model, "data": args.data, "objective": args.objective,
              "baseline": baseline, "best_bias": best["bias"],
              "calibrated": best["metrics"]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.out}")
    print(f"\nVerify on a benchmark with:\n"
          f"  python evals/run_eval.py --model {args.model} "
          f"--dataset {loader}:{path} --logit-bias {','.join(map(str, best['bias']))}")


if __name__ == "__main__":
    main()
