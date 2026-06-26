# OTS False-Positive Fix — A2P Transactional SMS

A measured false-positive problem in the OTS classifier, plus a leakage-free
training set to close it. Everything here is reproducible from the build scripts.

## The problem

On a held-out control set of **legitimate** SMS, model 2.7 classifies a large
share of real transactional A2P traffic (bank alerts, delivery notices, government
reminders, appointment reminders, OTPs) as **spam**. This is high-value traffic a
telecom firewall must pass — a bank OTP tagged spam means a user can't log in.

Conversational ham is unaffected, and it's language-independent — so this is a
*category* problem (transactional A2P), not a multilingual one.

## Root cause (in the v2.4 training data)

Transactional ham is thin and too templated to generalise:

| category | ham rows | spam rows |
|---|---|---|
| government | 174 | 2,208 |
| appointment | 11 | 16 |
| telecom | 320 | 91 |

Ham-transactional examples have a median length of ~34 chars ("Your order #X has
been shipped"); realistic branded messages are 80–140 chars, so the model overfit
the short template. Real non-English transactional ham is essentially absent.

## Baseline (model 2.7, 88-msg held-out benchmark)

| | false-positive rate |
|---|---|
| **overall** | **67%** (59/88), all → spam, none → phishing |
| government | 94% |
| delivery | 92% |
| appointment / telecom | 75% |
| bank_alert | 67% |
| otp | 50% |
| personal / personal_money | **0%** (control — conversational ham is clean) |

By language: en 68%, es 64%, it 64%, pt 73% — confirms it's category, not language.

## Files

- `build_benchmark.py` → `ham_benchmark_v1.csv` — **locked** 88-msg held-out
  benchmark. The measuring stick. *Never train on this.*
- `build_training.py` → `ham_training_v1.csv` — **486-row** training set:
  realistic branded A2P **ham** (282) plus matched **hard-negative scam twins**
  (204, labelled phishing) so the fix doesn't blind the model to real attacks.
  Gap-weighted (government/delivery heaviest), 4 languages, and
  auto-deduplicated against the benchmark (**0 overlap**).
- `summarize.py` — prints a compact FP report from a run's predictions file.

## How to measure the fix

```bash
# baseline (already captured above)
python evals/run_eval.py --model <2.7>.pth \
  --dataset csv:ham_benchmark_v1.csv --tag baseline

# fine-tune on ham_training_v1.csv (mixed with a rehearsal sample of the
# existing corpus, same recipe as the v2.6/2.7 incremental fine-tune), then:
python evals/run_eval.py --model <retrained>.pth \
  --dataset csv:ham_benchmark_v1.csv --tag after

python summarize.py baseline
python summarize.py after
```

The drop in block rate on the (untouched) benchmark is the false-positive
improvement, measured leakage-free.

## Provenance

All messages are constructed-realistic (standard A2P templates + real per-locale
institution names; fictional URLs, amounts, refs), not scraped, and not generated
by the attack-data pipeline. es/it/pt are author-verified. Scam-twin URLs are
fictional dedicated domains, never real shorteners.