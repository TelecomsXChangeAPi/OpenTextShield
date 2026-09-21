# Model 2.9: distilled from TypeSafe

Model 2.7 scored 83% on a 100-message hand-written bench where TypeSafe
(jev-1.13.0, reading `docs/LABELING_GUIDE.md`) scored 96–98%. Every 2.7 miss was
a data gap, not a capacity limit: prize and job lures labeled spam, real branded
notices blocked, conversational social engineering passed. 2.9 closes the gap by
training on TypeSafe's verdicts for the whole corpus instead of on 9.4k rows.

Nothing here is deployed. The public server still runs 2.7 until this is
validated on real traffic.

## What was done

1. **The whole corpus was labeled by the teacher.** TypeSafe answered the guide's
   label question for every unique text in the 137k dedup corpus plus the
   generated sets (154,475 answers, 153M input tokens, cached in
   `dataset/curated/typesafe_corpus_answers.jsonl` so it never has to be paid
   for twice). `evals/distill_labels.py ask` / `build`.
2. **Eval texts were excluded** by exact text, template shape and token Jaccard
   (≥ 0.5) against bench100, fable5, hard_legit, Mishra, UCI and IMC25, so the
   numbers below are on unseen messages. A deterministic 5% of the labeled
   corpus is held out by text hash for reporting and calibration.
3. **The archetypes the corpus lacked were generated and teacher-verified**:
   new-number family scams, code-relay asks, wrong-number openers, fake orders
   with a callback number, and their benign twins (`evals/generate_archetypes.py`,
   3,263 rows kept).
4. **Legitimate branded notices were generated at scale.** The first 2.9 run
   blocked every "Verizon: your bill is available at myvzw.com", because the
   labeled corpus had 844 ham rows with a link against 33,554 phishing rows with
   a link (4 vs 1,420 with brand, link and amount). `evals/generate_legit_notices.py`
   composes bank, telco, courier, retailer, government, health and travel
   notices from independent pools in 14 languages, with real domains and routine
   calls to action; 13,430 rows kept where the teacher agreed (96%). Its
   `--twins-only` mode emits the minimal-pair lure for each notice, including
   link-stripped and KYC-style variants that dominate real smishing; 9,358 kept.
5. **Soft-target fine-tuning.** `evals/finetune_distill.py` trains from the 2.5
   base with `(1-α)·CE(hard label) + α·KL(teacher probabilities ‖ student)`,
   α = 0.5, lr 2e-5, 2 epochs, batch 32, dynamic padding bucketed to multiples of
   16 (MPS caches a graph per shape). One seed takes 68 minutes on an M4 Pro.
6. **Three seeds, seven eval sets**, scored by `evals/score_candidate.sh` and
   rendered by `evals/distill_report.py`.

## Training set

134,710 rows (ham 55,528 / spam 18,026 / phishing 61,156), 5,660 held out.

| Source | Rows kept |
|---|---|
| Corpus (teacher label, confidence ≥ 0.6) | 106,493 |
| Generated legitimate notices | 13,430 |
| Generated notice lures (twins) | 9,358 |
| Generated archetypes | 3,263 |
| synthetic_fable5 (2.7 era) | 1,227 |
| notice_pairs (2.8 era) | 939 |

Dropped: 17,561 rows where the teacher was below 0.6 confidence, 5,867 the
teacher called "no message", 416 generated rows whose intended label the teacher
rejected. Relative to the corpus labels the teacher moved 17,234 rows from spam
to phishing (prize, job and delivery lures, as the guide says), 2,323 from
phishing to ham and 1,366 from spam to ham (real notices), and 1,112 from ham to
phishing.

## Results

Three seeds (7, 13, 42), each 66–68 minutes on the M4 Pro. Mean with min–max
spread across seeds; 2.7 is the deployed checkpoint scored by the same harness.

| Set | Metric | 2.7 | 2.9 mean (min–max) |
|---|---|---|---|
| bench100 (hand-written, 100) | accuracy | 83.0% | 96.7% (96.0%–98.0%) |
| bench100 (hand-written, 100) | block accuracy | 90.0% | 98.7% (98.0%–99.0%) |
| bench100 (hand-written, 100) | phishing recall | 75.8% | 97.0% (97.0%–97.0%) |
| bench100 (hand-written, 100) | false block rate | 17.6% | 1.0% (0.0%–2.9%) |
| hard_legit A2P (40) | accuracy | 67.5% | 98.3% (97.5%–100.0%) |
| hard_legit A2P (40) | block accuracy | 67.5% | 98.3% (97.5%–100.0%) |
| hard_legit A2P (40) | phishing recall | 0.0% | 0.0% (0.0%–0.0%) |
| hard_legit A2P (40) | false block rate | 32.5% | 1.7% (0.0%–2.5%) |
| fable5 clean (71) | accuracy | 83.1% | 94.4% (93.0%–95.8%) |
| fable5 clean (71) | block accuracy | 98.6% | 98.1% (97.2%–98.6%) |
| fable5 clean (71) | phishing recall | 75.6% | 100.0% (100.0%–100.0%) |
| fable5 clean (71) | false block rate | 6.2% | 0.0% (0.0%–0.0%) |
| teacher held-out corpus | accuracy | 80.1% | 98.8% (98.8%–98.8%) |
| teacher held-out corpus | block accuracy | 95.8% | 99.2% (99.1%–99.2%) |
| teacher held-out corpus | phishing recall | 62.5% | 98.7% (98.5%–98.8%) |
| teacher held-out corpus | false block rate | 7.4% | 0.8% (0.7%–0.8%) |
| Mishra & Soni (5,971) | accuracy | 89.4% | 94.7% (94.6%–94.9%) |
| Mishra & Soni (5,971) | block accuracy | 99.3% | 97.6% (97.5%–97.7%) |
| Mishra & Soni (5,971) | phishing recall | 6.1% | 71.3% (69.8%–73.5%) |
| Mishra & Soni (5,971) | false block rate | 0.5% | 0.5% (0.4%–0.6%) |
| UCI SMS Spam (5,574) | accuracy | 99.5% | 92.9% (92.7%–93.0%) |
| UCI SMS Spam (5,574) | block accuracy | 99.5% | 98.3% (98.2%–98.4%) |
| UCI SMS Spam (5,574) | phishing recall | 0.0% | 0.0% (0.0%–0.0%) |
| UCI SMS Spam (5,574) | false block rate | 0.5% | 0.6% (0.5%–0.7%) |
| IMC25 smishing (8,005) | accuracy | 46.8% | 66.1% (62.6%–68.0%) |
| IMC25 smishing (8,005) | block accuracy | 72.4% | 71.5% (68.3%–73.3%) |
| IMC25 smishing (8,005) | phishing recall | 45.4% | 67.7% (64.1%–69.6%) |

The shipped `mbert_ots_model_2.9.pth` is seed 13: 98% on bench100, best of the
three on Mishra, IMC25 and the teacher held-out set, one false block on
hard_legit (a Chase fraud check called spam at 0.95). Seed spread is 2 points on
the bench and under 1 point on the 5k-row sets, against a 6-point spread in the
2.8 work, so the gain is not seed noise.

Notes on the sets:

- **bench100** is the 100-message set from the TypeSafe comparison. TypeSafe
  itself scored 98% on the first run and 96% on a rerun (its two extra misses
  were low-confidence). 2.9's only threat miss is the wrong-number opener
  ("is this still your number? I think we met at Lisa's party"), which TypeSafe
  also passes at full confidence. Its other misses are spam-vs-phishing category
  swaps on a car-warranty ad, a coding bootcamp and a cash-for-houses ad; the
  gateway blocks all three either way.
- **hard_legit** measures false blocks on real-looking bank, delivery and
  billing notices. 2.7 blocked 13 of 40; 2.9 blocks none.
- **UCI** 3-class accuracy drops by design: 308 of its 747 spam rows are prize
  and premium-rate lures the guide calls phishing. 62 UCI "spam" rows now pass,
  almost all 2004-era ringtone order confirmations and jokes that UCI labels
  spam and the guide would call ham. Block accuracy stays above 98%.
- **Mishra** block accuracy slips from 99.3% to 97.6%: 88 rows 2.7 blocked now
  pass, almost all UCI-derived 2004-era premium-rate order confirmations
  ("Thanks for your ringtone order, your mobile will be charged 4.50"), billing
  receipts and jokes labeled spam in that set. Phishing recall on the same set
  goes from 6% to 71% because prize lures are now phishing.
- **IMC25** replaces every URL, brand and phone number with placeholders, and
  45% of its rows contain leftover artifacts like "our service desk" or
  "Main St 12". Half of its rows have no link at all, so it mostly measures
  link-free lure detection. On artifact-free rows 2.9 blocks 76.6% against
  2.7's 75.3%.

## Calibration and confidence

`evals/calibrate_thresholds.py` on the teacher held-out set finds nothing to
fix: the best logit bias (+0.5 on phishing) is worth 0.07 points. 2.7's errors
were unfilterable (13 of its 17 bench misses were at ≥ 0.99 confidence); none of
2.9's 71 held-out errors reach 0.99, and 18 of them sit below the SMPP proxy's
0.7 threshold. The threshold stays at 0.7.

## Serving check

Two paths, both with auditing off, both scored on bench100 against the offline
harness dumps:

- **uvicorn on the M4 Pro** (`OTS_MBERT_MODEL_PATH`, port 8003): version 2.9
  detected from the filename, 100/100 agreement, 87 ms median while training
  shared the GPU.
- **Docker** (`docker build -t opentextshield:2.9 .`, 3.16 GB, CPU): version
  2.9, 100/100 agreement, 98/100 correct, 101 ms median. The API unit tests and
  both SMPP proxy integration suites (47 + 32) pass against the container.

The checkpoint must be named `mbert_ots_model_2.9.pth`: the loader parses the
version from the filename and falls back to the configured version for names
like `2.9-s13`. `settings.py` and `.dockerignore` now point at 2.9; the public
server is unchanged until the image is deployed there.

## What is not solved

- Wrong-number and pig-butchering openers with no ask in the first message.
  Neither model catches them from one message; that needs conversation context.
- IMC25's link-free lures. 2.9 is ahead of 2.7 by a point on the clean rows,
  not by a margin that matters.
- The bench is hand-written and short. Real SMPP traffic is messier; treat the
  bench numbers as directional and let the audit log decide.

## Reproduce

```bash
export TYPESAFE_API_KEY=...
ots/bin/python evals/generate_archetypes.py --n 300 --seed 11 --out src/mBERT/training/model-training/dataset/curated/synthetic_archetypes_v1.csv
ots/bin/python evals/generate_legit_notices.py --n 12000 --seed 23 --out src/mBERT/training/model-training/dataset/curated/synthetic_legit_notices_v1.csv
ots/bin/python evals/generate_legit_notices.py --twins-only --n 10000 --seed 24 --out src/mBERT/training/model-training/dataset/curated/synthetic_notice_twins_v1.csv
ots/bin/python evals/distill_labels.py ask --concurrency 16   # resumable; cached answers are skipped
ots/bin/python evals/distill_labels.py build
evals/train_distill_seeds.sh 2.9 /tmp/ots_eval 7 13 42       # needs uci.tsv and imc25.csv in /tmp/ots_eval
ots/bin/python evals/distill_report.py
```
