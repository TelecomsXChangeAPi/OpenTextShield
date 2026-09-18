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

**These are single runs.** The seed section below supersedes the per-candidate
claims here: the differences between neighbouring rows are inside the run-to-run
spread. The direction of the dial is real; the exact ordering is not.

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
- **2.8e looked like the only candidate beating 2.7 without raising false
  blocks.** Three seeds later that turned out to be a lucky run, not a property
  of the data.
- **2.8d and 2.8f block far more** (IMC25 +8.2 and +4.9 points) but double the
  false blocks on fable5 legitimate messages. That is the wrong trade for an
  operator: a blocked bank alert is more visible than a delivered scam.
- **The hard legit set is 40 handwritten messages by one author**, so treat it
  as a smoke test, not evidence.
- **Spam recall falls everywhere** because prize scams are now labeled phishing:
  UCI 99.7% to 81.3%, Mishra 98.0% to 93.7% for 2.8e. Those messages are still
  blocked (UCI block rate stays 99.9%), so this is a reporting change, not a
  miss. Any dashboard splitting spam from phishing needs re-baselining.
- **Phishing recall improves a lot** where the taxonomy now agrees with the
  benchmark: Mishra 6.1% to 30.9%, fable5 clean 75.6% to 78.0%.

## Run-to-run variance changes the conclusion

The table above is one training run per configuration. Repeating the same
configuration with three seeds shows the spread is large enough to swamp those
differences, so single-run comparisons were misleading.

Mean of 3 seeds (seeds 7, 13, 29), same recipe, against model 2.7:

| Metric | 2.7 | Cleaned data | Cleaned data + generated notice pairs |
|---|---|---|---|
| IMC25 block | 72.4% | 76.7% | 76.0% |
| UCI block | 99.9% | 99.6% | 99.6% |
| Mishra block | 98.8% | 99.3% | 99.1% |
| fable5 clean block | 100% | 100% | 100% |
| Obfuscated block | 99.3% | 99.1% | 97.5% |
| UCI false blocks | 0.5% | 0.6% | 0.8% |
| Mishra false blocks | 0.5% | 0.7% | 0.9% |
| fable5 clean false blocks | 6.2% | 10.4% | 2.1% |
| Hard legit blocked | 13/40 | 21/40 | 13/40 |

- **Without the generated pairs the model over-blocks real notices.** It flags
  messages 2.7 passes: "Your Amazon package was delivered", "PayPal: You sent
  $45.00", "Venmo: Alex Kim paid you $18.50", "Sberbank: transfer received".
  The legitimate notices taken from the corpus are mostly one-time codes, while
  the attack side gained delivery, payment and billing lures, so the model
  learned that transactional shapes are suspicious.
- **With the pairs, false blocks on legitimate-looking messages come back down**
  (fable5 clean 6.2% to 2.1%, hard legit back to 13/40), and IMC25 blocking
  stays 3.6 points above 2.7.
- **It is still a trade.** The paired model blocks about 290 more attacks per
  8,000 IMC25 messages, and adds roughly 35 false blocks per 10,000 legitimate
  messages on UCI and Mishra. It is also 1.8 points worse on obfuscated text.
- **Single runs are not evidence.** Any future comparison needs at least three
  seeds; the seed spread reaches 6 points on fable5 false blocks.

## Recommendation

- **Keep model 2.7 in production and ship the platform fixes without it.** The
  code fixes in this release (obfuscation cleanup, the SMPP threshold and skip
  bypasses) are wins with no model risk. The data work is committed and
  reproducible, but no trained candidate is strictly better than 2.7.
- **If more blocking is wanted**, build with `--with-notice-pairs` and validate
  on real traffic. That configuration blocks the most real-world smishing of any
  run here while keeping false blocks on legitimate-looking messages below 2.7,
  at the cost of about 35 extra false blocks per 10,000 ordinary ham messages.
- **Next experiments worth running**, each with three seeds: class-weighted loss
  to offset phishing now being the largest class (42% of rows), and more benign
  transaction and delivery notices, which is the shape the model most often gets
  wrong.
- All 102 flagged rows have been reviewed (`dataset/curated/label_review.csv`).

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
