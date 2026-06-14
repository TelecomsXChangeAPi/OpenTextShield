#!/usr/bin/env python3
"""
Incremental fine-tune of the OpenTextShield mBERT classifier on the targeted
synthetic dataset, starting from the existing v2.5 weights.

Strategy
--------
We continue training (not from scratch) from mbert_ots_model_2.5.pth so the
strong classic-spam behaviour is retained, then mix in:
  - 100% of the targeted synthetic data (modern smishing archetypes), and
  - a stratified random sample of the original v2.4 combined corpus

The original-data mix is a rehearsal set that prevents catastrophic forgetting:
without it, fine-tuning purely on new attack patterns would erode the 99% UCI
accuracy. A modest learning rate (1e-5) and 2 epochs are enough to absorb the
new patterns on CPU.

Outputs a new checkpoint that run_eval.py can score head-to-head against v2.5.
"""

import argparse
import csv
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

REPO_ROOT = Path(__file__).resolve().parent.parent
VOCAB_FILE = REPO_ROOT / "evals/assets/bert-base-multilingual-cased-vocab.txt"
LABEL2ID = {"ham": 0, "spam": 1, "phishing": 2}
MAX_LEN = 96
csv.field_size_limit(10 * 1024 * 1024)


def read_csv(path, limit_per_label=None):
    buckets = {k: [] for k in LABEL2ID}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = (row.get("text") or "").strip()
            label = row.get("label")
            if text and label in LABEL2ID:
                buckets[label].append(text)
    rng = random.Random(7)
    out = []
    for label, texts in buckets.items():
        if limit_per_label and len(texts) > limit_per_label:
            texts = rng.sample(texts, limit_per_label)
        out.extend((t, LABEL2ID[label]) for t in texts)
    return out


class SmsDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tok = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        text, label = self.rows[i]
        enc = self.tok(text, truncation=True, max_length=MAX_LEN, padding="max_length",
                       return_tensors="pt")
        return {"input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "labels": torch.tensor(label)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--original", required=True)
    ap.add_argument("--orig-per-label", type=int, default=2500,
                    help="rehearsal sample size per class from the original corpus")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers; keep 0 on macOS to avoid tokenizer "
                         "multiprocessing deadlocks (see comment at the loader)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(7)
    torch.manual_seed(7)

    tokenizer = BertTokenizerFast(vocab_file=str(VOCAB_FILE), do_lower_case=False)

    rows = read_csv(args.synthetic)
    rows += read_csv(args.original, limit_per_label=args.orig_per_label)
    random.shuffle(rows)
    from collections import Counter
    print(f"Training rows: {len(rows)} | dist={Counter(l for _, l in rows)}")

    model = BertForSequenceClassification(BertConfig(vocab_size=119547, num_labels=3))
    model.load_state_dict(torch.load(args.base, map_location="cpu", weights_only=True))
    model.train()

    # num_workers defaults to 0: >0 spawns multiprocessing workers that, on
    # macOS with some PyTorch versions, deadlock when __getitem__ calls the
    # tokenizer. Opt into parallel loading explicitly via --num-workers.
    loader = DataLoader(SmsDataset(rows, tokenizer), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(loader) * args.epochs
    sched = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.1,
                                              total_iters=total_steps)

    step = 0
    for epoch in range(args.epochs):
        running = 0.0
        for batch in loader:
            optim.zero_grad()
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"])
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            running += out.loss.item()
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch+1} step {step}/{total_steps} "
                      f"loss {running/50:.4f}", flush=True)
                running = 0.0

    model.eval()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved fine-tuned model to {out_path}")


if __name__ == "__main__":
    main()
