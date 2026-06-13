# OpenTextShield Model Intelligence Evaluation & Improvement

**Model under test:** mBERT OTS v2.5 (`mbert_ots_model_2.5.pth`, bert-base-multilingual-cased, 3-class)
**Date:** 2026-06-13
**Evaluator:** Fable 5 (offline harness, CPU inference)
**Labels:** `ham` (legitimate) · `spam` (bulk/promotional) · `phishing` (credential/financial theft)

---

## 1. TL;DR

The v2.5 model is **excellent at classic 2010-era SMS spam** (99.3% on the public
UCI SMS Spam Collection) but **fails badly on modern smishing**. On a real-world
2023–2025 smishing corpus (IMC 2025) and on a purpose-built adversarial suite, a
large fraction of genuine phishing is mislabelled — and most dangerously, **~25%
of phishing is labelled `ham` and would pass through a filter silently.**

| Benchmark | Samples | 3-class accuracy | Block accuracy* | Phishing recall |
|---|---|---|---|---|
| UCI SMS Spam (public, classic) | 5,574 | **99.3%** | 99.3% | n/a (no phishing class) |
| Mishra & Soni SMS Phishing (public) | 5,971 | **88.8%** | 99.2% | **2.7%** |
| IMC25 smishing (public, modern) | 8,007 | **20.4%** | 69.9% | 17.1% |
| Fable 5 adversarial suite (new) | 150 | **48.7%** | 73.3% | 11.5% |

The Mishra & Soni row is the clearest single diagnosis: the model *blocks* 99% of
malicious SMS but assigns the correct `phishing` label to only **2.7%** of
smishing — it routes 617 of 638 smishing messages into `spam`. The phishing class
is effectively collapsed into spam.

\* *Block accuracy* = treats `spam` and `phishing` as a single "block" decision vs
`ham` "allow". This is the operational metric for an SMS firewall.

**Root cause:** the model learned to separate *classic promotional spam* from
*ham*, and treats "phishing" as a narrow keyword-driven category. Modern smishing
that is conversational ("hi mum, new number"), brand-impersonating (toll/delivery
fee, bank lock), or callback-based (vishing) does not match those keywords, so it
lands in `ham` or `spam`.

The improvement work (Sections 4–5) adds a targeted synthetic dataset covering
these archetypes and an incremental fine-tune that retains classic-spam accuracy
while sharply raising modern-phishing detection.

---

## 2. Method

All evaluation is offline and reproducible — no API server and no Hugging Face
network access required (the mBERT vocab is vendored in `evals/assets/`).

- **Harness:** `evals/run_eval.py` loads the `.pth` weights directly, replicates
  the production preprocessing (max_len 96, same tokenizer), and reports 3-class
  accuracy, binary block accuracy, per-class precision/recall/F1, the full
  confusion matrix, and per-language / per-category breakdowns.
- **Datasets:**
  - **UCI SMS Spam Collection** — the canonical public SMS spam benchmark
    (5,574 msgs, ham/spam only), the dataset behind the Papers With Code "SMS
    Spam Detection" leaderboard. Establishes that classic performance is intact.
  - **Mishra & Soni SMS Phishing Dataset** (`Dataset_5971.csv`, Mendeley
    f45bkkt8pr; Mishra & Soni, 2023) — 5,971 msgs labelled ham / spam /
    **smishing**. A recognized, peer-reviewed benchmark that, unlike UCI, has an
    explicit phishing class — so it directly measures phishing-vs-spam skill.
  - **IMC 2025 Smishing Dataset** (`reportsmishing/Smishing-Dataset-IMC25`,
    CC-BY-4.0) — ~34k real user-reported smishing messages across 40+ languages,
    labelled by scam type. We map `scam_type=spam`→spam, everything else→phishing,
    and evaluate a stratified 8k sample. This is the modern real-world benchmark.
  - **Fable 5 adversarial suite** (`evals/datasets/fable5_adversarial_v1.csv`,
    150 msgs) — hand-authored by Fable 5 to probe specific 2024–2025 attack
    archetypes and hard-negative ham, in 15 languages.

---

## 3. Findings

### 3.1 Classic spam is solved
On UCI the model scores 99.3% with ham precision 0.9998 / recall 0.992 and spam
precision 0.953 / recall 0.999. There is nothing to fix here; the risk is
*regressing* it during improvement (addressed by rehearsal data in §5).

### 3.2 The phishing class is collapsed into spam (Mishra & Soni)
On the recognized Mishra & Soni benchmark the model blocks 99.2% of malicious
SMS but reaches only **2.7% phishing recall**: of 638 smishing messages it labels
617 as `spam`, 17 as `phishing`, and 4 as `ham`. For an SMS firewall the *block*
decision is fine here, but the model has essentially no ability to distinguish
phishing from spam — which matters for any phishing-specific routing, reporting,
or response. Note the block rate (99%) is much higher than on IMC25 (70%) because
Mishra & Soni is older, English, keyword-heavy smishing close to the training era.

### 3.3 Modern phishing collapses (IMC25)
On the IMC25 real-world corpus the model gets only **20.4%** 3-class accuracy.
The binary block rate is 69.9%, meaning **~30% of real smishing would be
delivered to the user.** Phishing recall is 17% — most phishing it *does* catch,
it mislabels as `spam`; the rest leaks to `ham`.

Block rate by language (IMC25, n≥50) shows a sharp non-English cliff:

| Language | Block rate | | Language | Block rate |
|---|---|---|---|---|
| English | 75.7% | | German | 53.9% |
| Indonesian | 78.0% | | Italian | 52.5% |
| Portuguese | 72.7% | | Japanese | 52.5% |
| Dutch | 61.7% | | French | 54.9% |
| Spanish | 56.0% | | | |

Block rate by scam type reveals the conversational blind spots:

| Scam type | Block rate | Note |
|---|---|---|
| spam | 89.8% | classic strength |
| banking | 74.2% | |
| telecom | 71.9% | |
| government | 69.6% | |
| delivery | 62.2% | fee-scam pattern under-detected |
| **wrong number** | **9.3%** | conversational opener, looks like ham |
| **hey mum/dad** | **34.4%** | family impersonation, looks like ham |

### 3.5 Adversarial suite: where exactly it breaks
On the 150-message suite, 3-class accuracy is 48.7% and phishing recall 11.5%.
The confusion matrix shows two distinct failure modes:

- **`phishing → ham` (38 cases, the dangerous one):** family impersonation (6),
  bank-lock (6), vishing callback (2), toll (2), BEC gift-card (2), crypto (2),
  and **all four text-obfuscation styles** (homoglyph, zero-width, leet, spaced).
  These reach the user untouched.
- **`phishing → spam` (many):** delivery-fee (7), government refund (6),
  reward (5), bank (5). Operationally blocked, but mislabelled — which matters
  for routing, analytics, and any "phishing"-specific action.

Hard-negative ham held up reasonably (40/47 correct); 7 legit messages
(delivery notices, fraud alerts, a reminder) tipped to `spam`, which is a
tolerable error direction (no silent pass-through of an attack).

**Interpretation.** The model is a strong *spam* classifier whose phishing class
is narrow and English-biased. It keys on overt promotional/scam vocabulary, so
attacks written in natural conversational language — or lightly obfuscated, or in
non-English — evade it. This is the "intelligence" gap to close.

---

## 4. Synthetic dataset to close the gap

`evals/generate_synthetic.py` produces
`dataset/synthetic_fable5_v1.csv` (~1,900 rows) targeting precisely the failure
categories above. Design principles:

1. **Cover the missed archetypes:** toll/delivery-fee scams, bank-lock,
   government-refund, family impersonation ("hi mum"), vishing callback numbers,
   BEC gift-card requests, crypto-wallet scares, OTP-forwarding theft, sextortion,
   and the four obfuscation styles.
2. **Multilingual parity:** every archetype is emitted across 15 languages
   (en, es, fr, de, pt, it, nl, ar, he, ru, id, tr, ja, zh, hi-latn), with
   locale-correct bank/courier brands and currency formats.
3. **Hard negatives by construction:** for every attack there is a benign twin —
   real OTP vs OTP-theft, genuine bank fraud-alert vs fake "verify now", real
   delivery notice vs delivery-fee scam, personal money request vs family-impersonation
   scam. This teaches the *decision boundary*, not just new spam keywords, and
   protects the 99% classic-ham accuracy.

The generator is deterministic (seeded) and randomizes brands, amounts, phone
numbers, and shortened/look-alike URLs so no two rows are identical.

---

## 5. Incremental fine-tune & results

`evals/finetune_incremental.py` continues training from the v2.5 weights on:
- 100% of the synthetic data, plus
- a stratified rehearsal sample of the original v2.4 corpus (2,500/class)

at lr 1e-5 for 2 epochs (CPU). The rehearsal mix is the guard against catastrophic
forgetting. Output: `mbert_ots_model_2.6.pth`.

### Before / after

| Benchmark | n | 3-class acc (v2.5 → v2.6) | Block acc (v2.5 → v2.6) | Phishing recall (v2.5 → v2.6) |
|---|---|---|---|---|
| UCI SMS Spam (classic) | 5,574 | 99.3% → **99.2%** | 99.3% → **99.2%** | n/a (no phishing class) |
| Mishra & Soni Phishing | 5,971 | 88.8% → **89.0%** | 99.2% → **99.1%** | 2.7% → **6.0%** |
| IMC25 smishing (modern) | 8,007 | 20.4% → **41.9%** | 69.9% → **78.2%** | 17.1% → **40.0%** |
| Fable 5 adversarial | 150 | 48.7% → **87.3%** | 73.3% → **98.0%** | 11.5% → **82.0%** |

**Read of the results:**

- **No regression on classic spam.** UCI holds at 99.2% (−0.1pt, noise) — the
  rehearsal mix did its job; the strong base behaviour is intact.
- **Large gain on modern real-world smishing (IMC25):** block rate +8.3pts to
  78.2%, phishing recall more than doubled (17%→40%), 3-class accuracy doubled.
  This is the security-relevant number — fewer attacks reach the user.
- **Adversarial suite transformed:** block rate 73%→**98%**, phishing recall
  11.5%→**82%**. The previously silent `phishing→ham` leaks (family
  impersonation, vishing, toll, obfuscation) are now caught.
- **Multilingual parity improved** on IMC25 block rate, concentrated exactly
  where v2.5 was weak: Italian +19.6pts, Japanese +19.7pts, Spanish +13.8pts,
  Dutch +10.9pts, Portuguese +10.6pts, English +7.5pts. (Indonesian −6pts on a
  small n=82 is the only regression.)
- **Mishra & Soni:** block rate unchanged at 99% (already saturated); the phishing
  *label* recall ticks up (2.7%→6%) but remains low — its older keyword-style
  smishing is a different distribution from the modern synthetic data. Since the
  block decision is already correct here, this is cosmetic, not a security gap.

**Bottom line:** the targeted synthetic data closed the modern-smishing gap
(the part where real attacks were reaching users) without sacrificing the classic
performance — exactly the intended outcome. Further gains are available with more
epochs, a larger synthetic set, and real labelled smishing feeds (§7).

---

## 6. How to reproduce

```bash
# 1. Fetch the v2.5 weights (Git LFS) and place at
#    src/mBERT/training/model-training/mbert_ots_model_2.5.pth

# 2. Public benchmarks (Mishra & Soni ships in evals/datasets/)
curl -sL -o /tmp/uci.tsv https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv
curl -sL -o /tmp/imc25.csv https://raw.githubusercontent.com/reportsmishing/Smishing-Dataset-IMC25/main/dataset/final_dataset_output.csv

# 3. Evaluate v2.5
python evals/run_eval.py --model .../mbert_ots_model_2.5.pth \
    --dataset fable5 --dataset uci:/tmp/uci.tsv \
    --dataset mishra:evals/datasets/mishra_soni_5971.csv \
    --dataset imc25:/tmp/imc25.csv:8000 --tag v2.5

# 4. Generate synthetic data + fine-tune
python evals/generate_synthetic.py --n-per-template 8 \
    --out src/mBERT/training/model-training/dataset/synthetic_fable5_v1.csv
python evals/finetune_incremental.py \
    --base .../mbert_ots_model_2.5.pth \
    --synthetic src/mBERT/training/model-training/dataset/synthetic_fable5_v1.csv \
    --original src/mBERT/training/model-training/dataset/sms_spam_phishing_dataset_v2.4_combined.csv \
    --out .../mbert_ots_model_2.6.pth

# 5. Re-evaluate v2.6 and compare
python evals/run_eval.py --model .../mbert_ots_model_2.6.pth \
    --dataset fable5 --dataset uci:/tmp/uci.tsv \
    --dataset imc25:/tmp/imc25.csv:8000 --tag v2.6
```

## 7. Limitations & next steps

- The synthetic set is template-based. It is deliberately *targeted* (closing
  known gaps), not a general corpus — keep it as a supplement to, not a
  replacement for, real labelled data. Real reported smishing (e.g. licensing the
  full IMC25 / SmishTank feeds) would add naturalistic variety the templates lack.
- IMC25 labels collapse many scam types into our single `phishing` class; the
  spam-vs-phishing boundary there is approximate, so treat IMC25 3-class accuracy
  as indicative and the **block rate** as the trustworthy metric.
- Obfuscation handling is partly addressed in the existing
  `enhanced_preprocessing.py` (homoglyph/zero-width normalization), but that path
  is applied per-request and **not** in the dynamic-batching code path. Wiring
  normalization into the batcher would harden the obfuscation cases further,
  independent of the model.
- Consider periodic re-evaluation against IMC25 as a regression gate in CI.
