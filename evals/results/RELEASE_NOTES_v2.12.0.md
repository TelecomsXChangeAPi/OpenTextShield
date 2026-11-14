# Model 2.12 (release v2.12.0)

13 November 2026. Model 2.12 replaces 2.9. The API is unchanged apart from its
version number; `/health` reports model `2.12`.

## What changed

- **No external labeling service.** The code paths, stored answers and
  references for the external reviewing model used for 2.9 are removed.
  Labels now come from the labeling guide's rules (`evals/label_audit.py`),
  12,672 reviewed template clusters (`label_review_2.12.csv`) and the label
  each source gave.
- **9,547 new generated training rows** in 27 languages (phishing 5,033 /
  ham 3,611 / spam 903), each graded blind by a second pass and deduplicated
  against the training and eval sets. Themes and counts:
  `src/mBERT/training/model-training/dataset/curated/SYNTHETIC_V2.12.md`.
- **Training:** 162,297 rows (phishing 69,968 / ham 66,088 / spam 26,241),
  fine-tuned from the 2.9 weights, 2 epochs, lr 2e-5, batch 32, seed 42.
  Starting from 2.9 means the weights still carry what 2.9 learned from the
  external labels; nothing external is needed to train, label or serve.

## Results against 2.9

Same checkpoints, same sets, same text normalisation. None of the three public
sets contains generated data. Precision / recall / F1 in percent.

| Set | Class (rows) | 2.9 | 2.12 |
|---|---|---|---|
| Mishra & Soni (5,971) | ham (4,844) | 97.8 / 99.4 / 98.6 | 98.9 / 98.9 / 98.9 |
| | spam (489) | 73.3 / 78.1 / 75.6 | 68.3 / 88.3 / 77.1 |
| | phishing (638) | 89.5 / 73.5 / 80.7 | 84.1 / 65.4 / **73.5** |
| UCI SMS Spam (5,574) | ham (4,827) | 98.6 / 99.3 / 99.0 | 99.4 / 98.9 / 99.2 |
| | spam (747) | 96.4 / 49.9 / 65.8 | 98.9 / 61.9 / 76.1 |
| IMC25 smishing (8,005) | spam (382) | 34.6 / 34.3 / 34.4 | 34.6 / 36.4 / 35.5 |
| | phishing (7,623) | 96.7 / 69.6 / 81.0 | 96.9 / 71.9 / 82.6 |
| IMC25, second disjoint sample (18,268) | spam (1,112) | 33.0 / 33.5 / 33.2 | 33.6 / 35.6 / 34.5 |
| | phishing (17,156) | 95.8 / 66.0 / 78.1 | 95.9 / 68.2 / 79.7 |

| Set | Accuracy 2.9 | Accuracy 2.12 | Block accuracy 2.9 | Block accuracy 2.12 |
|---|---|---|---|---|
| Mishra & Soni | 94.9 | 94.5 | 97.7 | 98.2 |
| UCI SMS Spam | 92.7 | 94.0 | 98.2 | 98.6 |
| IMC25 smishing | 68.0 | 70.2 | 73.3 | 75.6 |
| Corpus hold-out, 5% by hash (6,538) | 94.3 | 99.0 | 97.3 | 99.6 |
| Hand-written bench (100) | 98.0 | 98.0 | 99.0 | 99.0 |
| Hard legitimate notices (40) | 97.5 | 100.0 | 97.5 | 100.0 |
| Generated adversarial probe (71) | 95.8 | 95.8 | 97.2 | 98.6 |

Inference speed is unchanged (same architecture): 310 to 510 messages per
second in batches on an Apple Silicon GPU for both models.

## Known regression: Mishra phishing

2.12 is better than or equal to 2.9 on every class of every real set except
Mishra phishing, where F1 drops from 80.7 to 73.5 (recall 73.5% to 65.4%).
Mishra labels UK phone-upgrade and dating advertisements as smishing; the
labeling guide and UCI call the same messages spam, and 2.9 had learned
Mishra's reading from the external labels. Most of these messages are still
blocked, as spam: Mishra block accuracy rises from 97.7% to 98.2%. Three
training rounds (two from the 2.5 weights, one from 2.9) did not close the gap
without giving up the UCI and Mishra spam gains, and the release was approved
with it documented.

On the 71-row generated probe, phishing F1 is 98.8 to 96.4 (one phishing
message of 41 missed, one more message wrongly called phishing), with spam and ham up.

## Other candidates

Per-set summaries for every candidate are in this directory: `v2.12-r1-*`
(round-1 data, from 2.5), `v2.12-r2-*` (round-2 data, from 2.5) and
`v2.12-s*` / `v2.12-soup` (round-2 data, from 2.9). The released checkpoint is
`v2.12-s42`.

## Rollback

`OTS_MBERT_MODEL_PATH=mBERT/training/model-training/mbert_ots_model_2.9.pth`
loads 2.9 without a code change, or pin the image tag back to `2.9`.
