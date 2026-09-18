#!/usr/bin/env python3
"""Build a labeled, realistic traffic corpus for soak-testing a release candidate.

Every message carries its gold label, so the soak report can measure false
blocks and misses per class over time, not just uptime. Sources are the corpora
already in the repo; nothing is fetched. This is traffic simulation, not a
benchmark: the same texts may appear in training data and that is fine here.

    python evals/soak/build_corpus.py --out ~/ots-soak/corpus.jsonl --size 60000
"""

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "evals/results"
DATA = REPO / "evals/datasets"
CURATED = REPO / "src/mBERT/training/model-training/dataset/curated"
csv.field_size_limit(10 * 1024 * 1024)

# Operator-style mix. Most real A2P/P2P traffic is legitimate.
MIX = {"ham_personal": 0.42, "ham_a2p": 0.40, "spam": 0.10, "phishing": 0.08}
OBFUSCATE_SHARE = 0.06   # of phishing: fullwidth / look-alike / zero-width / spaced
LONG_SHARE = 0.05        # of all: sent as message_payload (> 160 chars)

CYR = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "y": "у", "x": "х", "i": "і"}


def fullwidth(t):
    return "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in t)


def lookalike(t, r):
    return "".join(CYR[c.lower()] if c.lower() in CYR and r.random() < 0.5 else c for c in t)


def zero_width(t):
    return "​".join(t)


def spaced(t):
    return " ".join(t.replace(" ", ""))


def load_json(name):
    return json.load(open(RESULTS / f"predictions_{name}_v27c.json", encoding="utf-8"))


def load_csv(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def looks_redacted(t):
    return bool(re.search(r"<[A-Z_]+>|Main St 12|our service desk|the courier|the sender", t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    pools = {"ham_personal": [], "ham_a2p": [], "spam": [], "phishing": []}

    def add(pool, text, source, lang="", category=""):
        text = (text or "").strip()
        if len(text) < 4 or len(text) > 700:
            return
        pools[pool].append({"text": text, "source": source, "lang": lang or "en", "category": category})

    for r in load_json("uci"):
        add("ham_personal" if r["gold"] == "ham" else "spam", r["text"], "uci", "en", r["gold"])
    for r in load_csv(DATA / "mishra_soni_5971.csv"):
        label = r.get("label", "").lower()
        text = r.get("text") or r.get("TEXT") or ""
        if label == "ham":
            add("ham_personal", text, "mishra", "en", "ham")
        elif label == "spam":
            add("spam", text, "mishra", "en", "spam")
        elif label == "smishing":
            add("phishing", text, "mishra", "en", "smishing")
    for r in load_json("imc25"):
        if looks_redacted(r["text"]):
            continue
        add("phishing" if r["gold"] == "phishing" else "spam", r["text"], "imc25", r.get("language", ""), r.get("category", ""))
    for r in load_csv(DATA / "fable5_adversarial_v1.csv"):
        pool = {"ham": "ham_a2p" if r["category"].startswith("legit") else "ham_personal",
                "spam": "spam", "phishing": "phishing"}[r["label"]]
        add(pool, r["text"], "fable5", r.get("language", ""), r["category"])
    for r in load_csv(DATA / "hard_legit_a2p_v1.csv"):
        add("ham_a2p", r["text"], "hard_legit", r.get("language", ""), r["category"])
    for r in load_csv(CURATED / "additions_v1.csv"):
        pool = {"ads_spam": "spam", "legit_notice": "ham_a2p", "notice_phish": "phishing",
                "notice_pair_legit": "ham_a2p", "notice_pair_lure": "phishing"}[r["pool"]]
        add(pool, r["text"], "curated_additions", "", r["pool"])
    pairs = CURATED / "synthetic_notice_pairs_v1.csv"
    if pairs.exists():
        for r in load_csv(pairs):
            add("ham_a2p" if r["label"] == "ham" else "phishing", r["text"], "notice_pairs", "", r["pool"])
    for r in load_csv(CURATED / "train_subset_v2.7_cleaned.csv"):
        # synthetic fable5 rows: legit twins and multilingual lures
        if r["label"] == "ham" and re.search(r"\d{4,}|code|código|kode|paket|colis|pacco|delivered|entregado", r["text"], re.I):
            add("ham_a2p", r["text"], "synthetic", "", "legit")
        elif r["label"] == "phishing":
            add("phishing", r["text"], "synthetic", "", "lure")

    for k, v in pools.items():
        seen, uniq = set(), []
        for item in v:
            if item["text"] not in seen:
                seen.add(item["text"])
                uniq.append(item)
        pools[k] = uniq
        print(f"pool {k:13s} {len(uniq):6d} unique")

    out = []
    for i in range(args.size):
        pool = rng.choices(list(MIX), weights=list(MIX.values()))[0]
        item = dict(rng.choice(pools[pool]))
        label = "ham" if pool.startswith("ham") else pool
        obf = ""
        text = item["text"]
        if label == "phishing" and rng.random() < OBFUSCATE_SHARE:
            obf = rng.choice(["fullwidth", "lookalike", "zero_width", "spaced"])
            text = {"fullwidth": fullwidth, "lookalike": lambda t: lookalike(t, rng),
                    "zero_width": zero_width, "spaced": spaced}[obf](text)
        via = "payload" if (len(text) > 160 or rng.random() < LONG_SHARE) else "short"
        if via == "payload" and len(text) <= 160:
            # pad legitimately: a signature line, so length alone triggers the payload path
            text = text + "\n" + rng.choice(["Reply STOP to opt out.", "Msg&data rates may apply.",
                                             "Sent from my phone", "Ref: " + str(rng.randint(10**7, 10**8))])
        out.append({"id": i, "label": label, "pool": pool, "source": item["source"], "lang": item["lang"],
                    "category": item["category"], "obfuscation": obf, "via": via, "text": text})

    path = Path(args.out).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path}: {len(out)} messages", dict(Counter(r["label"] for r in out)),
          "| obfuscated:", sum(1 for r in out if r["obfuscation"]), "| via payload:", sum(1 for r in out if r["via"] == "payload"))


if __name__ == "__main__":
    main()
