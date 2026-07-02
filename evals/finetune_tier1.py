#!/usr/bin/env python3
"""
Tier-1 fine-tune for the OpenTextShield mBERT classifier.

This is the production-quality successor to finetune_incremental.py. It keeps the
same "continue from v2.5 + rehearsal mix" strategy but adds the Tier-1 training
improvements from evals/REPORT.md §7:

  * Device auto-detection — uses Apple Silicon MPS, CUDA, or CPU automatically.
    On an M1/M2 Mac this runs roughly 20-40x faster than the CPU proof-of-concept.
  * Stratified train/validation split with per-epoch validation metrics.
  * Best-checkpoint selection — saves the epoch with the best validation macro-F1,
    not the last epoch (the v2.6 run saved the last epoch, which is what caused
    the small UCI/Mishra regressions).
  * Class-weighted loss (default) or focal loss — directly counteracts the
    phishing->spam collapse by penalising the majority-class bias.
  * Gradient clipping, warmup, and a linear LR schedule.

Outputs a checkpoint plus a sidecar JSON of the per-epoch validation curve.

Example (Apple Silicon):
  python evals/finetune_tier1.py \
    --base   src/mBERT/training/model-training/mbert_ots_model_2.5.pth \
    --synthetic src/mBERT/training/model-training/dataset/synthetic_fable5_v1.csv \
    --original  src/mBERT/training/model-training/dataset/sms_spam_phishing_dataset_v2.4_combined.csv \
    --out    src/mBERT/training/model-training/mbert_ots_model_2.7.pth \
    --epochs 4 --batch-size 32 --loss class_weighted
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

try:
    from sklearn.metrics import f1_score, recall_score
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

REPO_ROOT = Path(__file__).resolve().parent.parent
VOCAB_FILE = REPO_ROOT / "evals/assets/bert-base-multilingual-cased-vocab.txt"
LABEL2ID = {"ham": 0, "spam": 1, "phishing": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MAX_LEN = 96
csv.field_size_limit(10 * 1024 * 1024)


def detect_device():
    if torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon MPS"
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA GPU"
    return torch.device("cpu"), "CPU"


def read_csv(path, limit_per_label=None, seed=7):
    buckets = {k: [] for k in LABEL2ID}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = (row.get("text") or "").strip()
            label = row.get("label")
            if text and label in LABEL2ID:
                buckets[label].append(text)
    rng = random.Random(seed)
    out = []
    for label, texts in buckets.items():
        if limit_per_label and len(texts) > limit_per_label:
            texts = rng.sample(texts, limit_per_label)
        out.extend((t, LABEL2ID[label]) for t in texts)
    return out


def stratified_split(rows, val_frac, seed=7):
    """Split keeping class proportions in both train and val."""
    by_label = {k: [] for k in LABEL2ID.values()}
    for text, label in rows:
        by_label[label].append((text, label))
    rng = random.Random(seed)
    train, val = [], []
    for label, items in by_label.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_frac))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class SmsDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tok = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        text, label = self.rows[i]
        enc = self.tok(text, truncation=True, max_length=MAX_LEN,
                       padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "labels": torch.tensor(label)}


def focal_loss(logits, targets, weight, gamma=2.0):
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    ce = F.nll_loss(logp, targets, weight=weight, reduction="none")
    pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
    return ((1 - pt) ** gamma * ce).mean()


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    preds, golds = [], []
    for batch in loader:
        logits = model(input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device)).logits
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        golds.extend(batch["labels"].tolist())
    n = len(golds)
    acc = sum(p == g for p, g in zip(preds, golds)) / n
    # block accuracy: ham vs (spam|phishing)
    block = sum((g != 0) == (p != 0) for p, g in zip(preds, golds)) / n
    if HAVE_SKLEARN:
        macro_f1 = f1_score(golds, preds, average="macro", zero_division=0)
        rec = recall_score(golds, preds, average=None, labels=[0, 1, 2], zero_division=0)
        phish_recall = rec[2]
    else:
        macro_f1 = acc
        tp = sum(p == 2 and g == 2 for p, g in zip(preds, golds))
        gold2 = sum(g == 2 for g in golds)
        phish_recall = tp / gold2 if gold2 else 0.0
    return {"accuracy": acc, "block_accuracy": block,
            "macro_f1": macro_f1, "phishing_recall": phish_recall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--original", required=True)
    ap.add_argument("--orig-per-label", type=int, default=2500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--loss", choices=["class_weighted", "focal", "plain"],
                    default="class_weighted")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device, device_name = detect_device()
    print(f"Device: {device_name}")

    tokenizer = BertTokenizerFast(vocab_file=str(VOCAB_FILE), do_lower_case=False)

    rows = read_csv(args.synthetic, seed=args.seed)
    rows += read_csv(args.original, limit_per_label=args.orig_per_label, seed=args.seed)
    train_rows, val_rows = stratified_split(rows, args.val_frac, seed=args.seed)
    print(f"Train: {len(train_rows)} | Val: {len(val_rows)}")
    print(f"Train dist: {Counter(ID2LABEL[l] for _, l in train_rows)}")

    # Class weights from the TRAIN split (inverse frequency, normalised).
    counts = Counter(l for _, l in train_rows)
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (len(LABEL2ID) * counts.get(i, 1)) for i in range(len(LABEL2ID))],
        dtype=torch.float, device=device,
    )
    print(f"Class weights (ham/spam/phishing): {weights.tolist()}")

    model = BertForSequenceClassification(BertConfig(vocab_size=119547, num_labels=3))
    model.load_state_dict(torch.load(args.base, map_location="cpu", weights_only=True))
    model.to(device)

    train_loader = DataLoader(SmsDataset(train_rows, tokenizer),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SmsDataset(val_rows, tokenizer),
                            batch_size=args.batch_size, shuffle=False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup = int(0.1 * total_steps)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    history = []
    best_f1 = -1.0
    best_epoch = -1
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            optim.zero_grad()
            logits = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits
            targets = batch["labels"].to(device)
            if args.loss == "focal":
                loss = focal_loss(logits, targets, weights)
            elif args.loss == "class_weighted":
                loss = F.cross_entropy(logits, targets, weight=weights)
            else:
                loss = F.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            running += loss.item()
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {running/50:.4f}", flush=True)
                running = 0.0

        metrics = evaluate(model, val_loader, device)
        metrics["epoch"] = epoch
        history.append(metrics)
        print(f"[val] epoch {epoch}: acc={metrics['accuracy']:.4f} "
              f"block={metrics['block_accuracy']:.4f} macroF1={metrics['macro_f1']:.4f} "
              f"phishR={metrics['phishing_recall']:.4f}", flush=True)

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            model.eval()
            torch.save(model.state_dict(), out_path)
            print(f"  -> new best (macroF1={best_f1:.4f}), saved to {out_path}", flush=True)

    sidecar = out_path.with_suffix(".trainlog.json")
    json.dump({"device": device_name, "loss": args.loss, "epochs": args.epochs,
               "best_epoch": best_epoch, "best_macro_f1": best_f1,
               "history": history}, open(sidecar, "w"), indent=2)
    print(f"\nDone. Best epoch {best_epoch} (macroF1={best_f1:.4f}).")
    print(f"Best checkpoint: {out_path}")
    print(f"Validation curve: {sidecar}")


if __name__ == "__main__":
    main()
