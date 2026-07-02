#!/usr/bin/env python3
"""
clean_ots_dataset.py - Data-quality cleaning for the OpenTextShield SMS dataset.

Removes the noise that hurts mBERT training:
  1. Rows with invalid labels (CSV-parse fragments)
  2. Machine-translation artifacts ("Sorry, I cannot provide a translation...")
  3. Junk text (empty / too short / punctuation- or symbol-only)
  4. Duplicate rows (exact + whitespace-normalized)

Usage:
    python clean_ots_dataset.py --in dataset/sms_spam_phishing_dataset_v2.4_combined.csv \
                                --out dataset/sms_spam_phishing_dataset_v2.4.1_dedup.csv
"""

import argparse
import csv
import re
import unicodedata
from collections import Counter

VALID_LABELS = {"ham", "spam", "phishing"}
MIN_TEXT_LEN = 3

TRANSLATION_MARKERS = [
    "cannot provide a translation", "i cannot provide", "unable to translate",
    "here is the translation", "translation to", "translated text", "translates to",
    "the translation of", "no es necesario realizar una traducci",
    "no necesita traducci", "i will translate", "could not be translated",
    "as it does not have any meaning", "there is no meaning in the given text",
    "ترجمة النص", "لا يمكنني تقديم", "لا حاجة لترجمة",
    "перевод текста", "перевести сообщение", "вот перевод",
    "පරිවර්තනය සඳහා", "සිංහල පරිවර්තනය", "මෙහි පරිවර්තනය",
    "மொழிபெயர்ப்பு", "மொழிபெயர்க்கவும்",
    "übersetzung:", "hier ist die übersetzung", "traduction:", "voici la traduction",
    "traduzione:", "ecco la traduzione", "```",
]

PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\t ")
    return re.sub(r"\s+", " ", text).strip()


def is_translation_artifact(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in TRANSLATION_MARKERS)


def is_junk(text: str) -> bool:
    if len(text) < MIN_TEXT_LEN:
        return True
    if PUNCT_ONLY_RE.match(text):
        return True
    if sum(unicodedata.category(c)[0] == "L" for c in text) == 0:
        return True
    return False


def clean(in_path: str, out_path: str) -> None:
    reasons = Counter()
    seen = set()
    kept_rows = []
    total = 0

    with open(in_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx_text = header.index("text") if "text" in header else 0
        idx_label = header.index("label") if "label" in header else 1
        idx_aug = header.index("augmentation_type") if "augmentation_type" in header else None

        for row in reader:
            total += 1
            if len(row) <= max(idx_text, idx_label):
                reasons["malformed_row"] += 1
                continue
            label = row[idx_label].strip().lower()
            if label not in VALID_LABELS:
                reasons["invalid_label"] += 1
                continue
            text = normalize(row[idx_text])
            if is_translation_artifact(text):
                reasons["translation_artifact"] += 1
                continue
            if is_junk(text):
                reasons["junk_text"] += 1
                continue
            key = (text.lower(), label)
            if key in seen:
                reasons["duplicate"] += 1
                continue
            seen.add(key)
            aug = row[idx_aug] if idx_aug is not None and len(row) > idx_aug else ""
            kept_rows.append((text, label, aug))

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "augmentation_type"])
        w.writerows(kept_rows)

    removed = total - len(kept_rows)
    print(f"Input rows:   {total}")
    print(f"Kept rows:    {len(kept_rows)}")
    print(f"Removed:      {removed}  ({removed / total:.1%})")
    print("Removed breakdown:")
    for reason, n in reasons.most_common():
        print(f"  {reason:22s} {n}")
    print("Clean label distribution:")
    for lbl, n in Counter(r[1] for r in kept_rows).most_common():
        print(f"  {lbl:22s} {n}")
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", dest="out", required=True)
    a = p.parse_args()
    clean(a.inp, a.out)