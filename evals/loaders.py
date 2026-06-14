#!/usr/bin/env python3
"""Shared dataset loaders for the OpenTextShield evaluation tooling.

Extracted from run_eval.py so that both run_eval.py and calibrate_thresholds.py
import the loaders normally, rather than one re-executing the other. This module
has NO argparse and NO side effects beyond raising the csv field-size limit, so
importing it is cheap and safe.

Each loader returns a list of dicts: {"text", "gold", "category", "language"}.
Labels are normalised to: ham / spam / phishing.
"""

import csv
import hashlib
import json
import random
from collections import defaultdict

LABELS = ["ham", "spam", "phishing"]

# SMS payloads with long redaction placeholders can exceed the default csv field
# size limit; raise it once at import.
csv.field_size_limit(10 * 1024 * 1024)


def load_fable5(path):
    samples = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            samples.append({
                "text": row["text"],
                "gold": row["label"],
                "category": row.get("category", ""),
                "language": row.get("language", ""),
            })
    return samples


def load_uci(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("label\t"):
                continue
            label, text = line.split("\t", 1)
            samples.append({"text": text, "gold": label, "category": "uci", "language": "en"})
    return samples


# Deterministic, neutral fill-ins for the IMC25 dataset's redaction
# placeholders so messages read like real SMS instead of containing literal
# "<URL>" tokens the model never saw in training.
_URLS = ["https://bit.ly/3xK9pQz", "http://tinyurl.com/y8m3kq", "https://t.co/8FjqLm2",
         "http://short.link/a82k1", "https://cutt.ly/w92mB"]
_PHONES = ["+31 6 84291023", "+44 7911 203948", "+1 (213) 555-0184", "06 52 48 19 27"]
_DATES = ["12/06", "May 19", "21-05", "03/11"]
_NAMES = ["DPD", "the courier", "our service desk", "the sender"]


def _fill_placeholders(text, seed):
    rng = random.Random(seed)
    for token, pool in (("<URL>", _URLS), ("<PHONE_NUMBER>", _PHONES),
                        ("<DATE_TIME>", _DATES), ("<NAMED_ENTITY>", _NAMES),
                        ("<EMAIL_ADDRESS>", ["info@service-mail.com"]),
                        ("<IBAN_CODE>", ["NL21INGB0001234567"]),
                        ("<PERSON>", ["Alex"]), ("<LOCATION>", ["Main St 12"])):
        while token in text:
            text = text.replace(token, rng.choice(pool), 1)
    return text


def load_imc25(path, sample_n=None):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            gold = "spam" if row.get("scam_type") == "spam" else "phishing"
            rows.append({
                "text": text, "gold": gold,
                "category": row.get("scam_type") or "unknown",
                "language": row.get("language") or "unknown",
            })
    if sample_n and sample_n < len(rows):
        # Stratify by language so small languages survive sampling.
        by_lang = defaultdict(list)
        for r in rows:
            by_lang[r["language"]].append(r)
        rng = random.Random(42)
        frac = sample_n / len(rows)
        sampled = []
        for lang, group in sorted(by_lang.items()):
            k = max(1, round(len(group) * frac))
            sampled.extend(rng.sample(group, min(k, len(group))))
        rows = sampled
    for r in rows:
        seed = int(hashlib.md5(r["text"].encode()).hexdigest()[:8], 16)
        r["text"] = _fill_placeholders(r["text"], seed)
    return rows


def load_mishra(path):
    """Mishra & Soni SMS Phishing Dataset (Dataset_5971.csv): LABEL,TEXT,URL,EMAIL,PHONE.
    Labels: ham / spam|Spam / Smishing|smishing -> ham / spam / phishing."""
    label_map = {"ham": "ham", "spam": "spam", "Spam": "spam",
                 "Smishing": "phishing", "smishing": "phishing"}
    samples = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            gold = label_map.get((row.get("LABEL") or "").strip())
            text = (row.get("TEXT") or "").strip()
            if gold and text:
                samples.append({"text": text, "gold": gold,
                                "category": "mishra", "language": "en"})
    return samples


def load_ots60(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [{"text": s["text"], "gold": s["ground_truth"],
             "category": s.get("category", ""), "language": s.get("language", "")}
            for s in data["samples"]]


def load_generic_csv(path):
    samples = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("text") and row.get("label") in LABELS:
                samples.append({"text": row["text"], "gold": row["label"],
                                "category": row.get("category", ""),
                                "language": row.get("language", "")})
    return samples


LOADERS = {"fable5": load_fable5, "uci": load_uci, "imc25": load_imc25,
           "mishra": load_mishra, "ots60": load_ots60, "csv": load_generic_csv}
