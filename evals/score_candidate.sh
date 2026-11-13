#!/usr/bin/env bash
# Score one checkpoint on every eval set used for the 2.9 distillation work.
#   evals/score_candidate.sh <checkpoint.pth> <tag> [eval_dir_with_uci.tsv_and_imc25.csv]
# Writes evals/results/summary_<tag>.json plus per-set prediction dumps.
set -euo pipefail
cd "$(dirname "$0")/.."
ckpt=$1; tag=$2; pub=${3:-/tmp/ots_eval}
args=(--model "$ckpt" --tag "$tag"
  --dataset csv:evals/datasets/bench100_v1.csv
  --dataset fable5:evals/datasets/fable5_adversarial_v1_clean.csv
  --dataset csv:evals/datasets/hard_legit_a2p_v1.csv
  --dataset mishra:evals/datasets/mishra_soni_5971.csv
  --dataset csv:src/mBERT/training/model-training/dataset/curated/distill_v2.9_heldout.csv)
[ -f "$pub/uci.tsv" ] && args+=(--dataset "uci:$pub/uci.tsv")
[ -f "$pub/imc25.csv" ] && args+=(--dataset "imc25:$pub/imc25.csv:8000")
ots/bin/python evals/run_eval.py "${args[@]}" > /dev/null
ots/bin/python - "$tag" <<'PY'
import json, sys
s = json.load(open(f"evals/results/summary_{sys.argv[1]}.json"))
print(f"{'set':28} {'n':>6} {'acc':>6} {'block':>6} {'phishR':>7} {'falseBlk':>8}")
for k, m in s.items():
    ham = m["per_class"]["ham"]; fb = 1 - ham["recall"] if ham["support"] else float("nan")
    print(f"{k:28} {m['n']:6d} {m['accuracy_3class']:6.1%} {m['accuracy_binary_block']:6.1%} "
          f"{m['per_class']['phishing']['recall']:7.1%} {fb:8.1%}")
PY
