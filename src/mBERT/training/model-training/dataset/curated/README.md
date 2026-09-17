# Curated training subset

Labels here follow `docs/LABELING_GUIDE.md`. Everything is built by
`evals/label_audit.py`; don't edit the CSVs by hand, except the `decision`
column of `label_review.csv`.

| File | What it is |
|---|---|
| `train_subset_v2.7.csv` | The exact rows model 2.7 trained on: all of `synthetic_fable5_v1.csv` plus the seed-7, 2,500-per-label rehearsal sample of `sms_spam_phishing_dataset_v2.4_combined.csv`. Original labels. |
| `typesafe_answers.jsonl` | TypeSafe's label for every unique text, keyed by a hash of the text. Saved so the build never needs TypeSafe again. |
| `label_review.csv` | Rows where TypeSafe confidently disagrees with the label. A person fills `decision` with `ham`, `spam`, `phishing` or `remove`, and optionally `decision_source` (blank means `manual`). Unreviewed rows are listed first. |
| `train_subset_v2.7_cleaned.csv` | The training file: approved rules applied, human decisions applied, unreviewed flagged rows left out. |
| `additions_candidates.csv` | Step 4 candidates: advertising spam and legitimate notices sampled from the rest of the corpus, excluding the training subset, every eval set, augmented copies and rows the approved rules would change or remove. At most 40 notices per sender family. |
| `additions_v1.csv` | Candidates kept because TypeSafe agreed with the corpus label at confidence 0.9 or higher. Labels are never changed here. |
| `train_v2.8_candidate.csv` | The cleaned subset plus the additions: the training file for the next model. |
| `summary.json` | Counts from the last build. |

## How labels are decided

1. The four approved decisions in the labeling guide are applied by pattern.
2. TypeSafe (a cloud model) only flags rows. It never changes a label on its
   own, and only public or synthetic training text is ever sent to it. The
   shipped product does not use it.
3. A person reviews flagged rows. Until then they stay out of the cleaned file.

## Rebuild

```bash
python evals/label_audit.py build            # no API key needed
python evals/label_audit.py build --flag-confidence 0.8
```

The source files are not modified. In particular `synthetic_fable5_v1.csv`
is still regenerated only by `evals/generate_synthetic.py`.
