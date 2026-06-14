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

No changes have been made to the vendored file. This project (OpenTextShield)
is the work of TelecomsXChange (TCXC); this attribution covers only the
third-party artifact named above.
