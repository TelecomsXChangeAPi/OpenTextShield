# ⚠️ Synthetic data — handle with context

`synthetic_fable5_v1.csv` (and any future `synthetic_*.csv` in this directory) is
**machine-generated** by `evals/generate_synthetic.py`. It is NOT real-world
labeled traffic.

## What it is

Template-generated SMS covering specific attack archetypes the v2.5 model missed
(toll/delivery-fee scams, bank-lock, government-refund, family impersonation,
vishing callbacks, OTP theft, and obfuscation styles) plus a benign "hard
negative" twin for every attack. Brand names, URLs, phone numbers, and amounts
are randomized fakes — they look plausible by design so the model learns the
decision boundary, not surface keywords.

## Rules of use

- **Do not** feed this file into an automated training pipeline as ground-truth
  real data without a human in the loop. It is a *supplement* to real labeled
  data, deliberately skewed toward hard cases — its class balance and phrasing
  are not representative of production traffic.
- Keep it paired with a rehearsal sample of the real corpus (see
  `evals/finetune_tier1.py`) to avoid skewing the model toward synthetic phrasing.
- Regenerate (don't hand-edit) via `evals/generate_synthetic.py`; the generator
  is seeded and deterministic.

The shipped model 2.7 was trained on this synthetic set **plus** a 2,500/class
rehearsal sample of `sms_spam_phishing_dataset_v2.4_combined.csv`. See
`evals/REPORT.md` for the full recipe and benchmark results.
