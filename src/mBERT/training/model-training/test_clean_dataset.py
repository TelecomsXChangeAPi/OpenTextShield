import csv
import os

DATASET = os.path.join(
    os.path.dirname(__file__),
    "dataset",
    "sms_spam_phishing_dataset_v2.4.1_dedup.csv",
)
VALID = {"ham", "spam", "phishing"}


def load():
    with open(DATASET, encoding="utf-8") as f:
        return [(r[0], r[1]) for r in list(csv.reader(f))[1:] if len(r) > 1]


def test_no_duplicates():
    rows = load()
    pairs = [(t.lower(), l) for t, l in rows]
    assert len(pairs) == len(set(pairs)), "Cleaned dataset still has duplicate rows"


def test_only_valid_labels():
    rows = load()
    assert all(l in VALID for _, l in rows), "Cleaned dataset has invalid labels"


def test_no_empty_text():
    rows = load()
    assert all(len(t.strip()) >= 3 for t, _ in rows), "Cleaned dataset has empty text"