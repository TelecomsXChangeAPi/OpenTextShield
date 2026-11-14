# Curated training data

Labels here follow `docs/LABELING_GUIDE.md`. The training file is assembled by
`evals/build_train_set.py`; don't edit the generated CSVs by hand, except the
`decision` column of `label_review_<version>.csv`.

| File | What it is |
|---|---|
| `train_subset_v2.7.csv` | The exact rows model 2.7 trained on: all of `synthetic_fable5_v1.csv` plus the seed-7, 2,500-per-label rehearsal sample of `sms_spam_phishing_dataset_v2.4_combined.csv`. Original labels. Provenance only. |
| `synthetic_notice_pairs_v1.csv` | Generated legitimate notices paired with notice-shaped lures (`evals/generate_notice_pairs.py`, model 2.8 work). |
| `synthetic_archetypes_v1.csv` | Generated conversational archetypes with benign twins (`evals/generate_archetypes.py`, model 2.9 work). |
| `synthetic_legit_notices_v1.csv`, `synthetic_notice_twins_v1.csv` | Generated branded notices on real domains in 14 languages and their lookalike lures (`evals/generate_legit_notices.py`, model 2.9 work). |
| `synthetic_v2.12_*.csv` | Free-written in-house generated rows for model 2.12, one file per theme, with `text,label,category,language`. Each row was written to an intended label and then re-graded blind against the guide; only rows where both passes agree are kept. See `SYNTHETIC_V2.12.md`. |
| `review_queue_<version>.csv` | Template clusters flagged for review: the production model confidently disagrees with the stored label. One representative text per cluster. |
| `label_review_<version>.csv` | The reviewer's decision per cluster id: `ham`, `spam`, `phishing`, `remove` or `unsure`. |
| `label_changes_<version>.csv` | Audit trail: every row whose final label differs from its source label, with the reason (`rule:<name>`, `review`, `flagged_unreviewed`). |
| `train_<version>.csv`, `heldout_<version>.csv` | The training file and the 5% corpus hold-out carved by text hash. Eval texts are excluded by exact match, template shape and token overlap. |
| `summary_<version>.json` | Counts from the last build. |

## How labels are decided

1. The four approved decisions in the labeling guide are applied by pattern
   (`evals/label_audit.py`).
2. Rows where the production checkpoint confidently disagrees with the stored
   label are grouped by template and queued for review. A reviewer decides each
   cluster against the guide; undecided clusters stay out of training.
3. Everything else keeps the label its source gave it.

Nothing here calls an external service, and no client traffic is ever used.

## Rebuild

```bash
python evals/build_train_set.py score --model src/mBERT/training/model-training/mbert_ots_model_2.9.pth --out /tmp/scored.json
python evals/build_train_set.py flag  --scored /tmp/scored.json --version 2.12   # writes review_queue_2.12.csv
python evals/build_train_set.py build --version 2.12                             # applies label_review_2.12.csv
```

The source files are not modified. In particular `synthetic_fable5_v1.csv`
is still regenerated only by `evals/generate_synthetic.py`.
