# Data cleanup and the 2.8 candidate

What changed, what it bought, and what it cost. Nothing here is released: model
2.7 stays in production until this is validated on real traffic.

## What was done

1. **A written labeling rule** (`docs/LABELING_GUIDE.md`), matching how the
   fable5 suite is already labeled.
2. **The training data model 2.7 learned from was relabeled against it**
   (`evals/label_audit.py`, `dataset/curated/`). 1,498 of 9,394 rows changed
   label and 263 were removed. TypeSafe (jev-1.13.0) acted only as a second
   rater: it flagged rows, and the decisions were made by a person.
3. **Real advertising spam and real legitimate notices were added** from the
   rest of the corpus, kept only where the corpus label and TypeSafe agreed.
4. **Retrained with model 2.7's exact recipe** (v2.5 base, plain loss, lr 1e-5,
   2 epochs, batch 32, seed 7), changing only the data and adding the
   production text cleanup.

## The trade-off we found

How many legitimate notices sit in the training data decides where the model
sits between blocking attacks and passing real messages. Block rate is what
production acts on, since spam and phishing are both rejected.

| Training mix | IMC25 block | Mishra block | fable5 clean block | Obfuscated block | UCI false blocks | Mishra false blocks | fable5 clean false blocks | Hard legit blocked |
|---|---|---|---|---|---|---|---|---|
| **2.7 (today)** | 72.4% | 98.8% | 100% | 99.3% | 0.5% | 0.5% | 6.2% | 13/40 |
| 2.8c: +548 notices, no lures | 69.6% | 98.5% | 100% | 97.2% | 0.5% | 0.5% | 6.2% | 9/40 |
| **2.8e: +548 notices, +329 lures** | **73.8%** | 99.0% | 100% | 98.7% | 0.5% | 0.5% | 6.2% | 16/40 |
| 2.8f: +329 notices, +329 lures | 77.3% | 99.6% | 100% | 99.6% | 0.7% | 0.8% | 12.5% | 17/40 |
| 2.8d: no notices, no lures | 80.6% | 99.6% | 100% | 99.8% | 0.7% | 0.8% | 12.5% | 18/40 |

Reading it:

- **Legitimate notices cut false blocks and cost attack blocking.** Added alone
  (2.8c) they teach the model to pass anything shaped like a bank or parcel
  message: IMC25 dropped 2.8 points, and the messages it started passing were
  real attacks (SBI PAN-card lures, Santander, BPOST, a MijnOverheid refund
  lure). Pairing them with notice-shaped attacks recovers most of that.
- **2.8e is the only candidate that beats 2.7 without raising false blocks on
  real ham.** It blocks more on IMC25 (+1.4 points) and Mishra (+0.2), reaches
  100% on the fable5 clean split, and matches 2.7's false-block rate on UCI,
  Mishra and fable5.
- **2.8d and 2.8f block far more** (IMC25 +8.2 and +4.9 points) but double the
  false blocks on fable5 legitimate messages. That is the wrong trade for an
  operator: a blocked bank alert is more visible than a delivered scam.
- **The hard legit set says 2.7 is best.** It is 40 handwritten messages by one
  author, so treat it as a smoke test, not evidence. The corpus ham sets agree
  that 2.8e ties 2.7.
- **Spam recall falls everywhere** because prize scams are now labeled phishing:
  UCI 99.7% to 81.3%, Mishra 98.0% to 93.7% for 2.8e. Those messages are still
  blocked (UCI block rate stays 99.9%), so this is a reporting change, not a
  miss. Any dashboard splitting spam from phishing needs re-baselining.
- **Phishing recall improves a lot** where the taxonomy now agrees with the
  benchmark: Mishra 6.1% to 30.9%, fable5 clean 75.6% to 78.0%.

## Recommendation

- Keep 2.7 in production.
- Treat **2.8e** (`mbert_ots_model_2.8-candidate.pth`, trained on
  `dataset/curated/train_v2.8_candidate.csv`) as the release candidate to
  validate on real traffic, watching the false-block rate on A2P alerts.
- The obvious next gain: more notice-shaped attack examples. The corpus pool ran
  dry at 329 accepted, which is why 2.8e still carries 548 notices against 329
  lures. Generating more through `evals/generate_synthetic.py` should move the
  block rate up without giving back the false-block gains.
- 102 flagged rows are still unreviewed and stay out of training until someone
  decides them (`dataset/curated/label_review.csv`).

## Reproducing

```bash
python evals/label_audit.py build          # cleaned data, no API key needed
python evals/finetune_tier1.py \
  --base      src/mBERT/training/model-training/mbert_ots_model_2.5.pth \
  --train-csv src/mBERT/training/model-training/dataset/curated/train_v2.8_candidate.csv \
  --out       src/mBERT/training/model-training/mbert_ots_model_2.8-candidate.pth \
  --epochs 2 --batch-size 32 --loss plain --lr 1e-5
```

Benchmarks were scored with the production text cleanup and each model's top
label, which is what the SMPP proxy now acts on.
