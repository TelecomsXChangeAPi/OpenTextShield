#!/usr/bin/env python3
"""
OpenTextShield model intelligence evaluation harness.

Loads the mBERT classifier directly (no API server required) and evaluates it
against one or more labelled datasets, producing per-dataset metrics and a
per-sample prediction dump for failure analysis.

Datasets supported:
  - fable5   : evals/datasets/fable5_adversarial_v1.csv (text,label,category,language)
  - uci      : UCI SMS Spam Collection TSV (label<TAB>text), public benchmark
  - imc25    : IMC 2025 smishing dataset CSV (reportsmishing/Smishing-Dataset-IMC25)
  - ots60    : benchmark/test_dataset.json (in-repo curated 60-sample set)
  - csv      : any CSV with text,label columns (e.g. training test_subset.csv)

The harness runs offline: tokenizer is built from the vendored vocab file in
evals/assets/ so no Hugging Face network access is needed.

Usage:
  python evals/run_eval.py --model /path/to/mbert_ots_model_2.5.pth \
      --dataset fable5 --dataset uci:/tmp/ots_eval/uci_sms_spam.tsv \
      --dataset imc25:/tmp/ots_eval/smishing_imc25.csv:4000 \
      --out evals/results
"""

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "src/mBERT/training/model-training/mbert_ots_model_2.5.pth"
VOCAB_FILE = REPO_ROOT / "evals/assets/bert-base-multilingual-cased-vocab.txt"

LABELS = ["ham", "spam", "phishing"]
MAX_LEN = 96  # matches production settings.max_text_length
BATCH_SIZE = 32

csv.field_size_limit(10 * 1024 * 1024)


def load_model(model_path: Path):
    tokenizer = BertTokenizerFast(vocab_file=str(VOCAB_FILE), do_lower_case=False)
    config = BertConfig(vocab_size=119547, num_labels=3)
    model = BertForSequenceClassification(config)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------- dataset loaders

def load_fable5(path):
    samples = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            samples.append({
                "text": row["text"],
                "gold": row["label"],
                "category": row.get("category", ""),
                "language": row.get("language", ""),
            })
    return samples


def load_uci(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("label\t"):
                continue
            label, text = line.split("\t", 1)
            samples.append({"text": text, "gold": label, "category": "uci", "language": "en"})
    return samples


# Deterministic, neutral fill-ins for the IMC25 dataset's redaction
# placeholders so messages read like real SMS instead of containing literal
# "<URL>" tokens the model never saw in training.
_URLS = ["https://bit.ly/3xK9pQz", "http://tinyurl.com/y8m3kq", "https://t.co/8FjqLm2",
         "http://short.link/a82k1", "https://cutt.ly/w92mB"]
_PHONES = ["+31 6 84291023", "+44 7911 203948", "+1 (213) 555-0184", "06 52 48 19 27"]
_DATES = ["12/06", "May 19", "21-05", "03/11"]
_NAMES = ["DPD", "the courier", "our service desk", "the sender"]


def _fill_placeholders(text, seed):
    rng = random.Random(seed)
    for token, pool in (("<URL>", _URLS), ("<PHONE_NUMBER>", _PHONES),
                        ("<DATE_TIME>", _DATES), ("<NAMED_ENTITY>", _NAMES),
                        ("<EMAIL_ADDRESS>", ["info@service-mail.com"]),
                        ("<IBAN_CODE>", ["NL21INGB0001234567"]),
                        ("<PERSON>", ["Alex"]), ("<LOCATION>", ["Main St 12"])):
        while token in text:
            text = text.replace(token, rng.choice(pool), 1)
    return text


def load_imc25(path, sample_n=None):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            gold = "spam" if row.get("scam_type") == "spam" else "phishing"
            rows.append({
                "text": text, "gold": gold,
                "category": row.get("scam_type") or "unknown",
                "language": row.get("language") or "unknown",
            })
    if sample_n and sample_n < len(rows):
        # Stratify by language so small languages survive sampling.
        by_lang = defaultdict(list)
        for r in rows:
            by_lang[r["language"]].append(r)
        rng = random.Random(42)
        frac = sample_n / len(rows)
        sampled = []
        for lang, group in sorted(by_lang.items()):
            k = max(1, round(len(group) * frac))
            sampled.extend(rng.sample(group, min(k, len(group))))
        rows = sampled
    for r in rows:
        seed = int(hashlib.md5(r["text"].encode()).hexdigest()[:8], 16)
        r["text"] = _fill_placeholders(r["text"], seed)
    return rows


def load_mishra(path):
    """Mishra & Soni SMS Phishing Dataset (Dataset_5971.csv): LABEL,TEXT,URL,EMAIL,PHONE.
    Labels: ham / spam|Spam / Smishing|smishing -> ham / spam / phishing."""
    label_map = {"ham": "ham", "spam": "spam", "Spam": "spam",
                 "Smishing": "phishing", "smishing": "phishing"}
    samples = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            gold = label_map.get((row.get("LABEL") or "").strip())
            text = (row.get("TEXT") or "").strip()
            if gold and text:
                samples.append({"text": text, "gold": gold,
                                "category": "mishra", "language": "en"})
    return samples


def load_ots60(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [{"text": s["text"], "gold": s["ground_truth"],
             "category": s.get("category", ""), "language": s.get("language", "")}
            for s in data["samples"]]


def load_generic_csv(path):
    samples = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("text") and row.get("label") in LABELS:
                samples.append({"text": row["text"], "gold": row["label"],
                                "category": row.get("category", ""),
                                "language": row.get("language", "")})
    return samples


LOADERS = {"fable5": load_fable5, "uci": load_uci, "imc25": load_imc25,
           "mishra": load_mishra, "ots60": load_ots60, "csv": load_generic_csv}


# ---------------------------------------------------------------- inference

def predict(model, tokenizer, texts, batch_size=BATCH_SIZE, logit_bias=None):
    """Return (labels, confidences). Sorts by length internally for speed.

    ``logit_bias`` (a 3-vector for ham/spam/phishing) is added to the logits
    before argmax — used to apply a calibrated decision bias from
    calibrate_thresholds.py. Confidence is still the softmax of the biased logits.
    """
    bias = torch.tensor(logit_bias, dtype=torch.float) if logit_bias else None
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    preds = [None] * len(texts)
    confs = [0.0] * len(texts)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                            max_length=MAX_LEN, return_tensors="pt")
            logits = model(**enc).logits.float()
            if bias is not None:
                logits = logits + bias
            probs = torch.softmax(logits, dim=1)
            top = probs.argmax(dim=1)
            for j, i in enumerate(idx):
                preds[i] = LABELS[top[j].item()]
                confs[i] = probs[j, top[j]].item()
    return preds, confs


# ---------------------------------------------------------------- metrics

def evaluate(samples, preds, confs):
    n = len(samples)
    correct = sum(1 for s, p in zip(samples, preds) if s["gold"] == p)
    # Binary firewall view: ham = allow, spam/phishing = block.
    blocked_ok = sum(1 for s, p in zip(samples, preds)
                     if (s["gold"] != "ham") == (p != "ham"))
    confusion = Counter((s["gold"], p) for s, p in zip(samples, preds))

    per_class = {}
    for label in LABELS:
        tp = confusion.get((label, label), 0)
        gold_n = sum(v for (g, _), v in confusion.items() if g == label)
        pred_n = sum(v for (_, p), v in confusion.items() if p == label)
        prec = tp / pred_n if pred_n else 0.0
        rec = tp / gold_n if gold_n else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[label] = {"precision": round(prec, 4), "recall": round(rec, 4),
                            "f1": round(f1, 4), "support": gold_n}

    by_cat = defaultdict(lambda: [0, 0])
    by_lang = defaultdict(lambda: [0, 0])
    for s, p in zip(samples, preds):
        for key, agg in ((s["category"], by_cat), (s["language"], by_lang)):
            agg[key][1] += 1
            if s["gold"] == p:
                agg[key][0] += 1

    return {
        "n": n,
        "accuracy_3class": round(correct / n, 4) if n else 0,
        "accuracy_binary_block": round(blocked_ok / n, 4) if n else 0,
        "per_class": per_class,
        "confusion": {f"{g}->{p}": v for (g, p), v in sorted(confusion.items())},
        "by_category": {k: {"acc": round(c / t, 4), "n": t}
                        for k, (c, t) in sorted(by_cat.items()) if k},
        "by_language": {k: {"acc": round(c / t, 4), "n": t}
                        for k, (c, t) in sorted(by_lang.items()) if k},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--dataset", action="append", required=True,
                    help="name | name:path | name:path:sample_n")
    ap.add_argument("--out", default=str(REPO_ROOT / "evals/results"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--logit-bias", default="",
                    help="comma-separated ham,spam,phishing logit bias from calibrate_thresholds.py")
    args = ap.parse_args()

    logit_bias = None
    if args.logit_bias:
        logit_bias = [float(x) for x in args.logit_bias.split(",")]
        assert len(logit_bias) == 3, "--logit-bias must be 3 comma-separated values"
        print(f"Applying logit bias (ham,spam,phishing): {logit_bias}", file=sys.stderr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}", file=sys.stderr)
    model, tokenizer = load_model(Path(args.model))

    summary = {}
    for spec in args.dataset:
        parts = spec.split(":", 2)
        name = parts[0]
        if name == "fable5" and len(parts) == 1:
            path = REPO_ROOT / "evals/datasets/fable5_adversarial_v1.csv"
        elif name == "ots60" and len(parts) == 1:
            path = REPO_ROOT / "benchmark/test_dataset.json"
        else:
            path = Path(parts[1])
        kwargs = {}
        if name == "imc25" and len(parts) == 3:
            kwargs["sample_n"] = int(parts[2])

        samples = LOADERS[name](path, **kwargs)
        print(f"[{name}] {len(samples)} samples — running inference...", file=sys.stderr)
        t0 = time.time()
        preds, confs = predict(model, tokenizer, [s["text"] for s in samples],
                               logit_bias=logit_bias)
        dt = time.time() - t0
        print(f"[{name}] done in {dt:.1f}s ({len(samples)/dt:.1f} msg/s)", file=sys.stderr)

        metrics = evaluate(samples, preds, confs)
        metrics["inference_seconds"] = round(dt, 1)
        summary[name] = metrics

        dump = [{**s, "pred": p, "conf": round(c, 4)}
                for s, p, c in zip(samples, preds, confs)]
        tag = f"_{args.tag}" if args.tag else ""
        with open(out_dir / f"predictions_{name}{tag}.json", "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False, indent=1)

    tag = f"_{args.tag}" if args.tag else ""
    with open(out_dir / f"summary{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
