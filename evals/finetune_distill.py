#!/usr/bin/env python3
"""Fine-tune mBERT on TypeSafe-labeled data with soft targets (knowledge distillation).

Input is the CSV written by `evals/distill_labels.py build`: text, label (the
teacher's choice) and p_ham/p_spam/p_phishing (the teacher's probabilities).
The loss is

    (1 - alpha) * CE(student, hard label [, class weights])
        + alpha * KL(teacher_probs || softmax(student))

so the student learns the teacher's decision and, on rows where the teacher was
unsure, its uncertainty. Texts get the production cleanup, batches are padded
dynamically (an SMS is ~25 tokens, so this is 3-4x faster than padding to 96),
validation is stratified and the best epoch by macro-F1 is saved.

    python evals/finetune_distill.py \\
        --base  src/mBERT/training/model-training/mbert_ots_model_2.5.pth \\
        --train src/mBERT/training/model-training/dataset/curated/distill_v2.9_train.csv \\
        --out   src/mBERT/training/model-training/mbert_ots_model_2.9-s7.pth \\
        --epochs 2 --lr 2e-5 --alpha 0.5 --seed 7
"""
import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loaders import production_normalize  # noqa: E402

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


def read_rows(path, limit=None, seed=7):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            text = (r.get("text") or "").strip()
            if not text or r.get("label") not in LABEL2ID:
                continue
            probs = [float(r.get(f"p_{c}", 1.0 if c == r["label"] else 0.0) or 0.0) for c in LABEL2ID]
            z = sum(probs) or 1.0
            rows.append((text, LABEL2ID[r["label"]], [p / z for p in probs], r.get("source", "")))
    if limit and len(rows) > limit:
        rows = random.Random(seed).sample(rows, limit)
    return rows


def stratified_split(rows, val_frac, seed):
    by = {}
    for r in rows:
        by.setdefault(r[1], []).append(r)
    rng = random.Random(seed)
    train, val = [], []
    for items in by.values():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_frac))
        val += items[:n_val]
        train += items[n_val:]
    rng.shuffle(train)
    return train, val


class Encoded(Dataset):
    def __init__(self, rows, tokenizer):
        enc = tokenizer([r[0] for r in rows], truncation=True, max_length=MAX_LEN)
        self.ids = enc["input_ids"]
        self.labels = [r[1] for r in rows]
        self.probs = [r[2] for r in rows]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i], self.probs[i]


def collate(batch):
    # Pad to a multiple of 16, not the exact longest row: MPS caches a compiled
    # graph per input shape, and arbitrary widths made it slow down 3x over a run.
    width = max(len(ids) for ids, _, _ in batch)
    width = min(MAX_LEN, ((width + 15) // 16) * 16)
    input_ids = torch.zeros(len(batch), width, dtype=torch.long)
    mask = torch.zeros(len(batch), width, dtype=torch.long)
    for i, (ids, _, _) in enumerate(batch):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        mask[i, :len(ids)] = 1
    return {"input_ids": input_ids, "attention_mask": mask,
            "labels": torch.tensor([l for _, l, _ in batch]),
            "probs": torch.tensor([p for _, _, p in batch], dtype=torch.float)}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, golds = [], []
    for b in loader:
        logits = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device)).logits
        preds += logits.argmax(1).tolist()
        golds += b["labels"].tolist()
    n = len(golds)
    acc = sum(p == g for p, g in zip(preds, golds)) / n
    f1s = {}
    for c in LABEL2ID.values():
        tp = sum(p == c and g == c for p, g in zip(preds, golds))
        fp = sum(p == c and g != c for p, g in zip(preds, golds))
        fn = sum(p != c and g == c for p, g in zip(preds, golds))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s[ID2LABEL[c]] = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    block = sum((p != 0) == (g != 0) for p, g in zip(preds, golds)) / n
    false_block = sum(p != 0 and g == 0 for p, g in zip(preds, golds)) / max(1, sum(g == 0 for g in golds))
    return {"accuracy": round(acc, 4), "macro_f1": round(sum(f1s.values()) / 3, 4), "f1": {k: round(v, 4) for k, v in f1s.items()},
            "block_accuracy": round(block, 4), "false_block_rate": round(false_block, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of the soft-target KL term")
    ap.add_argument("--class-weighted", action="store_true", help="inverse-frequency weights on the hard CE term")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, help="subsample rows (smoke tests)")
    ap.add_argument("--max-steps", type=int, help="stop early (smoke tests)")
    ap.add_argument("--eval-every", type=int, default=0, help="also validate every N steps and keep the best")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device, device_name = detect_device()
    print(f"Device: {device_name}", flush=True)

    rows = read_rows(args.train, limit=args.limit, seed=args.seed)
    texts = production_normalize([r[0] for r in rows])
    rows = [(t, r[1], r[2], r[3]) for t, r in zip(texts, rows)]
    train_rows, val_rows = stratified_split(rows, args.val_frac, args.seed)
    print(f"Train: {len(train_rows)} | Val: {len(val_rows)} | dist: {Counter(ID2LABEL[r[1]] for r in train_rows)} | "
          f"sources: {Counter(r[3] for r in train_rows)}", flush=True)

    tokenizer = BertTokenizerFast(vocab_file=str(VOCAB_FILE), do_lower_case=False)
    train_loader = DataLoader(Encoded(train_rows, tokenizer), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(Encoded(val_rows, tokenizer), batch_size=128, shuffle=False, collate_fn=collate)

    counts = Counter(r[1] for r in train_rows)
    weights = None
    if args.class_weighted:
        total = sum(counts.values())
        weights = torch.tensor([total / (3 * counts.get(i, 1)) for i in range(3)], dtype=torch.float, device=device)
        print(f"Class weights: {weights.tolist()}")

    model = BertForSequenceClassification(BertConfig(vocab_size=119547, num_labels=3))
    model.load_state_dict(torch.load(args.base, map_location="cpu", weights_only=True))
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = int(0.06 * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: s / max(1, warmup) if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history, best, best_at = [], -1.0, None
    step, t0 = 0, time.time()

    def validate(tag):
        nonlocal best, best_at
        m = evaluate(model, val_loader, device)
        m["at"] = tag
        history.append(m)
        print(f"[val {tag}] acc={m['accuracy']} macroF1={m['macro_f1']} f1={m['f1']} block={m['block_accuracy']} "
              f"falseBlock={m['false_block_rate']}", flush=True)
        if m["macro_f1"] > best:
            best, best_at = m["macro_f1"], tag
            torch.save(model.state_dict(), out_path)
            print(f"  -> new best, saved {out_path}", flush=True)
        model.train()

    done = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for b in train_loader:
            optim.zero_grad()
            logits = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device)).logits
            hard = F.cross_entropy(logits, b["labels"].to(device), weight=weights)
            soft = F.kl_div(F.log_softmax(logits, dim=1), b["probs"].to(device), reduction="batchmean")
            loss = (1 - args.alpha) * hard + args.alpha * soft
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            running += loss.item()
            step += 1
            if step % 100 == 0:
                el = time.time() - t0
                print(f"epoch {epoch} step {step}/{total_steps} loss {running/100:.4f} "
                      f"{step/el:.2f} it/s eta {(total_steps-step)/(step/el)/60:.0f} min", flush=True)
                running = 0.0
            if args.eval_every and step % args.eval_every == 0:
                validate(f"e{epoch}s{step}")
            if args.max_steps and step >= args.max_steps:
                done = True
                break
        validate(f"epoch{epoch}")
        if done:
            break

    sidecar = out_path.with_suffix(".trainlog.json")
    json.dump({"device": device_name, "base": args.base, "train": args.train, "epochs": args.epochs, "lr": args.lr,
               "alpha": args.alpha, "class_weighted": args.class_weighted, "seed": args.seed, "batch_size": args.batch_size,
               "train_rows": len(train_rows), "val_rows": len(val_rows), "best_at": best_at, "best_macro_f1": best,
               "minutes": round((time.time() - t0) / 60, 1), "history": history}, open(sidecar, "w"), indent=2)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Best {best_at} (macroF1={best}). Checkpoint: {out_path}")


if __name__ == "__main__":
    main()
