#!/usr/bin/env python3
"""Curate in-house generated SMS rows before they become training data.

Generated rows carry the label their author intended. Before use they are
re-graded blind against docs/LABELING_GUIDE.md by an independent pass that sees
only the text, and a row is kept only when both passes agree. Kept rows are then
deduplicated against the corpus, the existing generated sets and every eval set.

    # 1. split the raw files into blind grading chunks (id,text) plus a private key
    python evals/curate_synthetic.py chunks --in-dir /path/gen --work-dir /path/grading --size 300
    # 2. a grader fills /path/grading/grades/chunk_NN.csv with id,label for every chunk
    # 3. merge: keep agreements, dedupe, write dataset/curated/synthetic_<version>_<theme>.csv
    python evals/curate_synthetic.py merge --work-dir /path/grading --version 2.12

Raw files are CSV with text,label,category,language; the theme is the file name
without its numeric prefix (01_legit_en.csv -> legit_en).
"""
import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_audit as la  # noqa: E402
from label_audit import CURATED_DIR, DATASET_DIR, LABELS, REPO_ROOT, EvalExclusion, text_key  # noqa: E402

EXISTING_SOURCES = [DATASET_DIR / "sms_spam_phishing_dataset_v2.4.1_dedup.csv", DATASET_DIR / "synthetic_fable5_v1.csv",
                    CURATED_DIR / "synthetic_notice_pairs_v1.csv", CURATED_DIR / "synthetic_archetypes_v1.csv",
                    CURATED_DIR / "synthetic_legit_notices_v1.csv", CURATED_DIR / "synthetic_notice_twins_v1.csv"]
LANG = re.compile(r"^[a-z]{2}$")
MIN_LEN, MAX_LEN = 12, 400
csv.field_size_limit(10 * 1024 * 1024)


def theme_of(path):
    return re.sub(r"^\d+_", "", Path(path).stem)


def read_raw(in_dir):
    rows, problems = [], Counter()
    for path in sorted(Path(in_dir).glob("*.csv")):
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for i, r in enumerate(csv.DictReader(f)):
                text = (r.get("text") or "").strip().replace("\r", " ")
                label = (r.get("label") or "").strip().lower()
                lang = (r.get("language") or "").strip().lower()
                if not text or len(text) < MIN_LEN or len(text) > MAX_LEN:
                    problems[f"{path.name}:bad_length"] += 1
                    continue
                if label not in LABELS:
                    problems[f"{path.name}:bad_label"] += 1
                    continue
                if not LANG.match(lang):
                    problems[f"{path.name}:bad_language"] += 1
                    continue
                if re.search(r"<[A-Z_]{3,}>|\[(NAME|AMOUNT|LINK|URL|DATE)\]", text):
                    problems[f"{path.name}:placeholder"] += 1
                    continue
                rows.append({"id": f"{theme_of(path)}:{i:04d}", "theme": theme_of(path), "text": text, "label": label,
                             "category": (r.get("category") or "").strip() or "unspecified", "language": lang})
    return rows, problems


def chunks(args):
    rows, problems = read_raw(args.in_dir)
    seen, unique = set(), []
    for r in rows:
        k = text_key(r["text"].lower())
        if k in seen:
            problems["duplicate_within_generated"] += 1
            continue
        seen.add(k)
        unique.append(r)
    work = Path(args.work_dir)
    (work / "chunks").mkdir(parents=True, exist_ok=True)
    (work / "grades").mkdir(parents=True, exist_ok=True)
    with open(work / "_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "theme", "text", "label", "category", "language"])
        w.writeheader()
        w.writerows(unique)
    order = unique[:]
    random.Random(args.seed).shuffle(order)
    n = 0
    for n, start in enumerate(range(0, len(order), args.size), start=1):
        with open(work / "chunks" / f"chunk_{n:02d}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "text"])
            for r in order[start:start + args.size]:
                w.writerow([r["id"], r["text"]])
    print(json.dumps({"raw_rows": len(rows), "unique_rows": len(unique), "chunks": n, "problems": dict(problems)}, indent=2))


def merge(args):
    work = Path(args.work_dir)
    with open(work / "_key.csv", newline="", encoding="utf-8") as f:
        key = {r["id"]: r for r in csv.DictReader(f)}
    grades = {}
    for path in sorted((work / "grades").glob("*.csv")):
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                g = (r.get("label") or "").strip().lower()
                if r.get("id") in key and g in LABELS + ("remove", "unsure"):
                    grades[r["id"]] = g
    # dedupe targets: everything already in the dataset, plus every eval set
    existing = set()
    for path in EXISTING_SOURCES:
        if path.exists():
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                for r in csv.DictReader(f):
                    t = (r.get("text") or "").strip()
                    if t:
                        existing.add(la._template_key(t))
    excl = EvalExclusion()

    kept, dropped, stats = defaultdict(list), [], Counter()
    for rid, r in key.items():
        g = grades.get(rid)
        if g is None:
            stats["ungraded"] += 1
            dropped.append({**r, "grade": "", "why": "ungraded"})
            continue
        if g != r["label"]:
            stats[f"disagree:{r['label']}->{g}"] += 1
            dropped.append({**r, "grade": g, "why": "disagreement"})
            continue
        if la._template_key(r["text"]) in existing:
            stats["duplicate_of_existing"] += 1
            dropped.append({**r, "grade": g, "why": "duplicate_of_existing"})
            continue
        why = excl.why_excluded(r["text"])
        if why:
            stats[f"eval_overlap_{why}"] += 1
            dropped.append({**r, "grade": g, "why": f"eval_overlap_{why}"})
            continue
        kept[r["theme"]].append(r)
        stats["kept"] += 1

    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    for old in CURATED_DIR.glob(f"synthetic_v{args.version}_*.csv"):
        old.unlink()
    for theme, rows in sorted(kept.items()):
        with open(CURATED_DIR / f"synthetic_v{args.version}_{theme}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["text", "label", "category", "language"])
            w.writeheader()
            w.writerows({k: r[k] for k in ("text", "label", "category", "language")} for r in rows)
    with open(CURATED_DIR / f"synthetic_v{args.version}_dropped.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["theme", "text", "label", "grade", "why", "language"])
        w.writeheader()
        w.writerows({k: d[k] for k in ("theme", "text", "label", "grade", "why", "language")} for d in dropped)

    all_kept = [r for rows in kept.values() for r in rows]
    summary = {"version": args.version, "generated_rows": len(key), "graded_rows": len(grades), "kept": len(all_kept),
               "dropped": len(dropped), "counts": dict(sorted(stats.items())),
               "kept_by_label": dict(Counter(r["label"] for r in all_kept)),
               "kept_by_language": dict(Counter(r["language"] for r in all_kept).most_common()),
               "kept_by_theme": {t: dict(Counter(r["label"] for r in rows)) for t, rows in sorted(kept.items())}}
    (CURATED_DIR / f"synthetic_v{args.version}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("chunks")
    p.add_argument("--in-dir", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--size", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    p = sub.add_parser("merge")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--version", default="2.12")
    args = ap.parse_args()
    {"chunks": chunks, "merge": merge}[args.cmd](args)


if __name__ == "__main__":
    main()
