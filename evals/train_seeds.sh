#!/usr/bin/env bash
# Train a release candidate with several seeds and score each one.
#   evals/train_seeds.sh <version> <public-eval-dir> <seed> [<seed> ...]
# e.g. evals/train_seeds.sh 2.12 /Volumes/ExternalSSD/ots-scratch/public_eval 7 13 42
# Checkpoints go to $OTS_CKPT_DIR (default: the model-training directory).
# Starting weights are $OTS_BASE (default: the 2.5 checkpoint).
set -euo pipefail
cd "$(dirname "$0")/.."
ver=$1; pub=$2; shift 2
MT=src/mBERT/training/model-training
CKPT_DIR=${OTS_CKPT_DIR:-$MT}
BASE=${OTS_BASE:-$MT/mbert_ots_model_2.5.pth}  # starting weights
echo "=== base $BASE"
for seed in "$@"; do
  out=$CKPT_DIR/mbert_ots_model_${ver}-s${seed}.pth
  echo "=== seed $seed -> $out  ($(date))"
  ots/bin/python evals/finetune_tier1.py \
    --base "$BASE" \
    --train-csv $MT/dataset/curated/train_v${ver}.csv \
    --out "$out" --epochs 2 --lr 2e-5 --batch-size 32 --loss plain --val-frac 0.03 \
    --eval-every 1500 --seed "$seed" \
    2>&1 | grep --line-buffered -v Warning
  echo "=== scoring seed $seed  ($(date))"
  evals/score_candidate.sh "$out" "v${ver}-s${seed}" "$pub"
done
echo "=== all seeds done ($(date))"
