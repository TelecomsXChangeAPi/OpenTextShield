# Tier-1 Improvement Runbook (Apple Silicon / M1)

Goal: retrain the classifier from the v2.5 base using the Tier-1 improvements
(GPU/MPS, validation split, best-checkpoint selection, class-weighted loss) and
verify the gains against the public benchmarks. On an M1 this takes minutes, not
the ~1 hour the CPU proof-of-concept took.

Everything needed is already on this branch: the v2.5 base model (Git LFS), the
synthetic dataset, the original corpus, the vendored tokenizer vocab, and the
benchmark loaders. No Hugging Face download is required.

## 0. One-time setup

```bash
# from the repo root, on the sms-classification-intelligence-eval branch
git pull origin sms-classification-intelligence-eval     # get the Tier-1 scripts

python3.12 -m venv ots && source ots/bin/activate
pip install --upgrade pip
# Broad pins so this installs against current stable wheels (don't pin to an
# unreleased torch). Newer minors are fine.
pip install "torch>=2.3,<3.0" "transformers>=4.40,<5.0" "tokenizers>=0.19,<1.0" scikit-learn

# confirm Apple Silicon GPU is visible
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

Make sure the LFS model is actually present (not a pointer):

```bash
git lfs pull
ls -la src/mBERT/training/model-training/mbert_ots_model_2.5.pth   # should be ~680 MB
```

## 1. Download the two public benchmarks (Mishra & Soni already ships in-repo)

```bash
curl -sL -o /tmp/uci.tsv   https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv
curl -sL -o /tmp/imc25.csv https://raw.githubusercontent.com/reportsmishing/Smishing-Dataset-IMC25/main/dataset/final_dataset_output.csv
```

## 2. Baseline — score v2.5 (so you have a clean before)

```bash
python evals/run_eval.py \
  --model src/mBERT/training/model-training/mbert_ots_model_2.5.pth \
  --dataset uci:/tmp/uci.tsv \
  --dataset mishra:evals/datasets/mishra_soni_5971.csv \
  --dataset imc25:/tmp/imc25.csv:8000 \
  --dataset fable5 \
  --tag v2.5
```

## 3. Tier-1 fine-tune (runs on the M1 GPU automatically)

```bash
python evals/finetune_tier1.py \
  --base   src/mBERT/training/model-training/mbert_ots_model_2.5.pth \
  --synthetic src/mBERT/training/model-training/dataset/synthetic_fable5_v1.csv \
  --original  src/mBERT/training/model-training/dataset/sms_spam_phishing_dataset_v2.4_combined.csv \
  --out    src/mBERT/training/model-training/mbert_ots_model_2.7.pth \
  --epochs 2 --batch-size 32 --loss plain --lr 1e-5
```

This is the **shipped recipe** for model 2.7: a gentle fine-tune (plain
cross-entropy, lr 1e-5, 2 epochs) continued from the v2.5 base over the synthetic
set plus a 2,500/class rehearsal sample. Earlier heavier runs (class-weighted
loss, 4 epochs, lr 2e-5) scored higher on the in-domain validation split but
*overfit* — they regressed the real-world IMC25 block rate. Backing off to this
gentler configuration recovered generalization and beat v2.5 on every benchmark.

It prints `Device: Apple Silicon MPS`, a per-epoch validation line, and saves the
**best** epoch (by validation macro-F1) — plus a `*.trainlog.json` with the curve.

> ⚠️ The ~99% macro-F1 in the trainlog is on the **in-domain validation split**,
> not the public benchmarks. It is an optimistic signal — do not interpret it as
> real-world accuracy. The numbers that matter are the public-benchmark scores
> from step 4 (and the IMC25 block rate in particular). See `evals/REPORT.md`.

Knobs worth trying (departures from the shipped recipe — validate against IMC25
before trusting any gain, since the in-domain split is an optimistic signal):
- `--loss class_weighted` / `--loss focal` — weight the rare phishing class; both
  raised validation F1 here but hurt real-world generalization.
- `--epochs 4` / `--lr 2e-5` — more/larger steps; the same overfitting caveat.
- `--orig-per-label 4000` — more rehearsal data (better protects classic spam,
  slower).

## 4. Score the new model (the after)

```bash
python evals/run_eval.py \
  --model src/mBERT/training/model-training/mbert_ots_model_2.7.pth \
  --dataset uci:/tmp/uci.tsv \
  --dataset mishra:evals/datasets/mishra_soni_5971.csv \
  --dataset imc25:/tmp/imc25.csv:8000 \
  --dataset fable5 \
  --tag v2.7

# side-by-side
python evals/compare_runs.py \
  --before evals/results/summary_v2.5.json \
  --after  evals/results/summary_v2.7.json
```

## 5. (Optional) Calibrate the phishing/spam boundary — no retraining

If phishing is still being routed into spam, find a per-class logit bias on a
held-out set and verify it on a *different* set (calibrate and verify on separate
data — calibrating and scoring on the same set is optimistic):

```bash
# calibrate on Mishra
python evals/calibrate_thresholds.py \
  --model src/mBERT/training/model-training/mbert_ots_model_2.7.pth \
  --data evals/datasets/mishra_soni_5971.csv:mishra \
  --objective macro_f1 --out evals/results/calibration.json

# verify the chosen bias on the adversarial set (held-out)
python evals/run_eval.py \
  --model src/mBERT/training/model-training/mbert_ots_model_2.7.pth \
  --dataset fable5 --logit-bias 0.0,-1.0,2.0      # use the bias it printed
```

## 6. Promote

If the after numbers hold up (UCI block ≥ 99%, IMC25 block up, no surprise
regressions), rename `mbert_ots_model_2.7.pth` to the version you ship, commit it
via Git LFS, and update `default_mbert_version` / the model path in
`src/api_interface/config/settings.py`.

The obfuscation hardening (homoglyph + zero-width normalisation in the live
batching path) is already wired in on this branch — no model change needed for it.
