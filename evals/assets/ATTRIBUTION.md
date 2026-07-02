# Third-party assets — attribution

## bert-base-multilingual-cased-vocab.txt

- **Source:** Hugging Face model `bert-base-multilingual-cased`
  (https://huggingface.co/bert-base-multilingual-cased), originally published by
  Google Research as part of BERT (https://github.com/google-research/bert).
- **What it is:** the WordPiece vocabulary (119,547 tokens) for the multilingual
  cased BERT tokenizer. Vendored here so the offline evaluation harness
  (`evals/run_eval.py`, `evals/calibrate_thresholds.py`) can construct the
  tokenizer without any network access to the Hugging Face Hub.
- **License:** Apache License 2.0. The vocabulary file is redistributed
  unmodified under the terms of that license; see
  https://www.apache.org/licenses/LICENSE-2.0 for the full text.
- **Storage:** tracked via Git LFS (see `.gitattributes`).

## mishra_soni_5971.csv (evals/datasets/)

- **Source:** "SMS Phishing Dataset for Machine Learning and Pattern Recognition"
  by Sandhya Mishra and Devpriya Soni, Mendeley Data V1
  (https://data.mendeley.com/datasets/f45bkkt8pr/1), `Dataset_5971.csv`.
- **What it is:** 5,971 SMS messages labelled ham / spam / smishing, used by the
  offline evaluation harness as a phishing-vs-spam benchmark
  (`evals/run_eval.py --dataset mishra:...`).
- **License:** CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
  Redistributed here with attribution as required by the license. Please cite:
  Mishra, S., Soni, D. (2023), Mendeley Data, doi:10.17632/f45bkkt8pr.1.
- **Changes:** re-encoded to UTF-8 CSV with `text,label` columns; otherwise
  content is unmodified.

No other changes have been made to the vendored files. This project
(OpenTextShield) is the work of TelecomsXChange (TCXC); the attributions above
cover only the third-party artifacts named.
