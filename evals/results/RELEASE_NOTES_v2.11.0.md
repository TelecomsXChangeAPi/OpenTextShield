# OpenTextShield v2.11.0 — Obfuscation and SMPP bypass fixes (model 2.7 unchanged)

**Release candidate (`v2.11.0-rc.1`).** Platform version `2.11.0`. The shipped
classifier stays **model 2.7**. This RC is for validation on its own branch and
is **not** a production release.

## Headline

Attackers could get past OpenTextShield without beating the model at all: by
writing in fullwidth letters, by adding a header the proxy trusted, by padding a
message past the API's length limit, or by hiding the text in a second field.
This release closes those paths and stops the proxy from delivering unsure
verdicts as safe. No model change, so no regression risk on classification.

## What changed

**1. Disguised text is cleaned properly** (`enhanced_preprocessing.py`)

| Obfuscation of the same phishing messages (n=544) | Before | After |
|---|---|---|
| Fullwidth letters (`ＰａｙＰａｌ`) | 42.6% blocked | **100%** |
| Look-alike letters outside the old map (`ӏ ѕ ԁ`) | 80.9% blocked | **100%** |
| All styles combined | 92.1% | **99.3%** |

The old folding also damaged real text: it rewrote Russian and Greek messages
into mixed-script nonsense and mapped Cyrillic `у` to `u` through a duplicate
key. Folding is now context-aware, and on ~19,000 benchmark messages the new
cleanup changes **zero** predictions, where the old one flipped 18.

**2. The SMPP proxy no longer delivers unsure verdicts as safe**

Below `confidence_threshold` the proxy forwarded every spam or phishing verdict
as ham. Across all saved benchmark predictions that delivered 564 IMC25 phishing
messages and prevented only 5 false blocks. The new `below_threshold_action`
defaults to `use_label`; set it to `forward_as_ham` for the old behaviour.

**3. Classification can no longer be skipped by a sender**

- A single-shift header on text with no escape characters, and plain-ASCII
  ISO-2022-JP payloads, are now classified instead of skipped.
- Shift headers on non-GSM data coding are flagged rather than trusted.
- Messages over 512 characters are fitted to the API limit (start and end kept)
  instead of failing the call and being forwarded unclassified.
- `short_message` and `message_payload` are classified together when both are
  present, so a harmless short message cannot hide a phishing payload.
- New `unclassifiable_action` lets strict operators reject what cannot be read.
- Config is validated at startup, and malformed API replies no longer leave a
  client without a `submit_sm_resp`.

**4. Evaluation now matches production**

`run_eval.py` and `calibrate_thresholds.py` tokenised raw text while the API
normalised first, so published numbers did not describe the deployed pipeline
for non-ASCII traffic. Both now apply the same cleanup (`--raw-text` reproduces
older numbers). Training does too, via `finetune_tier1.py --train-csv`.

**5. Honest eval sets**

- `fable5_adversarial_v1_clean.csv` (71 rows): the suite without rows that share
  a template with the synthetic training set. The previous token-Jaccard check
  missed obfuscated and translated twins; two rows had digit-for-digit copies in
  training.
- `hard_legit_a2p_v1.csv` (40 rows): legitimate bank, delivery, billing and code
  messages that look like scams, to measure false blocks. Handwritten synthetic.

**6. Training data cleaned and documented** (not shipped)

`docs/LABELING_GUIDE.md` plus `evals/label_audit.py` produce a relabeled,
deduplicated training set. Retraining on it gives a real trade rather than a
free win, so **model 2.7 stays**. Full evidence, including three seeds per
configuration, is in `evals/results/DATA_CLEANUP_2.8.md`.

## Tests

| Suite | Result |
|---|---|
| API unit tests | 39/39 |
| SMPP offline (`npm test`) | 137/137 |
| SMPP integration against model 2.7 | 47/47 and 32/32, identical to the previous proxy |

## Upgrade notes

- **Behaviour change:** unsure spam and phishing verdicts are now acted on. If
  your traffic makes false blocks the bigger risk, set
  `classification.below_threshold_action` to `forward_as_ham` in `config.json`.
- **Startup is now strict:** an unknown rule action, previously a silent hang,
  stops the proxy with a clear error.
- No model file changes, so no Git LFS pull is required for this release.

## Validating this RC

```bash
git checkout local-hardening-step1
cd src/smpp_interface && npm test
cd ../.. && python -m pytest src/api_interface/tests/ -q
python evals/run_eval.py \
  --model src/mBERT/training/model-training/mbert_ots_model_2.7.pth \
  --dataset fable5:evals/datasets/fable5_adversarial_v1_clean.csv \
  --dataset csv:evals/datasets/hard_legit_a2p_v1.csv --tag v2.11.0-rc1
```
