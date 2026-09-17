# Eval datasets

| File | Rows | What it measures |
|---|---|---|
| `fable5_adversarial_v1.csv` | 127 | In-house adversarial suite: modern smishing, obfuscation, and legitimate look-alikes. |
| `fable5_adversarial_v1_clean.csv` | 71 | The same suite without rows that share a template with `synthetic_fable5_v1.csv`, which every model since 2.7 trains on. Use this split for claims about unseen messages. |
| `fable5_template_twins.json` | 56 | Why each row was left out of the clean split: its closest training row and the probability that both come from the same template. |
| `hard_legit_a2p_v1.csv` | 40 | Legitimate bank, delivery, billing, security and code messages that look like scams. Measures false blocks. **Handwritten synthetic examples**, not real traffic. |
| `mishra_soni_5971.csv` | 5,971 | Public Mishra & Soni SMS phishing set. Its "smishing" label includes premium-rate prize spam, which `docs/LABELING_GUIDE.md` also calls phishing. |

UCI SMS Spam and IMC25 are downloaded at eval time; see `evals/TIER1.md`.

The clean fable5 split was built with TypeSafe as a one-time reviewer (see the
JSON for the method). The token Jaccard check it replaces missed obfuscated
twins, for example `B A N K  A L E R T ... call 8 8 8 4 0 2 1 1 7 6` against a
training row with the same phone number.

```bash
python evals/run_eval.py --model <checkpoint> \
  --dataset fable5:evals/datasets/fable5_adversarial_v1_clean.csv \
  --dataset csv:evals/datasets/hard_legit_a2p_v1.csv
```
