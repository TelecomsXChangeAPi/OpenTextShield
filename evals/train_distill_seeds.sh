#!/usr/bin/env bash
# Train the 2.9 distillation candidate with several seeds and score each one.
#   evals/train_distill_seeds.sh <version-tag> <public-eval-dir> <seed> [<seed> ...]
# e.g. evals/train_distill_seeds.sh 2.9 /tmp/ots_eval 7 13 42
set -euo pipefail
cd "$(dirname "$0")/.."
ver=$1; pub=$2; shift 2
MT=src/mBERT/training/model-training
for seed in "$@"; do
  out=$MT/mbert_ots_model_${ver}-s${seed}.pth
  echo "=== seed $seed -> $out  ($(date))"
  ots/bin/python evals/finetune_distill.py \
    --base $MT/mbert_ots_model_2.5.pth \
    --train $MT/dataset/curated/distill_v${ver}_train.csv \
    --out "$out" --epochs 2 --lr 2e-5 --alpha 0.5 --seed "$seed" --eval-every 1500 \
    2>&1 | grep --line-buffered -v Warning
  echo "=== scoring seed $seed  ($(date))"
  evals/score_candidate.sh "$out" "v${ver}-s${seed}" "$pub"
done
echo "=== all seeds done ($(date))"
