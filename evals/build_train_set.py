#!/usr/bin/env python3
"""Assemble a training set from the corpus and the generated sets under docs/LABELING_GUIDE.md.

Labels are decided in-house, in this order:

  1. The guide's four approved decisions, applied by pattern (evals/label_audit.py).
  2. Reviewed decisions in dataset/curated/label_review_<version>.csv. A review
     covers a whole template cluster (same text with digits replaced), as the
     guide's template rule says. `unsure` and `remove` take the cluster out.
  3. Otherwise the row keeps the label its source gave it.

Rows flagged for review but not yet decided stay out, as the guide says for
unsure rows. Flagging is done by disagreement: the production model's confident
prediction against the rule label. Eval texts are excluded by exact text,
template shape and token overlap. A deterministic 5% of the corpus is held out
by text hash for reporting.

    # 1. score every source text with the production checkpoint (scratch, not committed)
    python evals/build_train_set.py score --model src/.../mbert_ots_model_2.9.pth --out /tmp/scored.json
    # 2. write the review queue: one representative text per flagged template cluster
    python evals/build_train_set.py flag --scored /tmp/scored.json --version 2.12
    # 3. review the queue (ids + decisions into label_review_<version>.csv), then build
    python evals/build_train_set.py build --version 2.12
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_audit as la  # noqa: E402
from label_audit import CURATED_DIR, DATASET_DIR, LABELS, REMOVE, REPO_ROOT, EvalExclusion, text_key  # noqa: E402

CORPUS_CSV = DATASET_DIR / "sms_spam_phishing_dataset_v2.4.1_dedup.csv"
PROBE_ADDITIONS = REPO_ROOT / "evals/probe/logs/training_additions.csv"
FIXED_SOURCES = [  # (path, source tag)
    (CORPUS_CSV, "corpus"),
    (DATASET_DIR / "synthetic_fable5_v1.csv", "synthetic_fable5"),
    (CURATED_DIR / "synthetic_notice_pairs_v1.csv", "notice_pairs"),
    (CURATED_DIR / "synthetic_archetypes_v1.csv", "archetypes"),
    (CURATED_DIR / "synthetic_legit_notices_v1.csv", "legit_notices"),
    (CURATED_DIR / "synthetic_notice_twins_v1.csv", "notice_twins"),
    (PROBE_ADDITIONS, "probe"),
]
HELDOUT_FRAC = 0.05
FLAG_CONFIDENCE = 0.9
DECISIONS = LABELS + (REMOVE, "unsure")


def sources_for(version):
    """Fixed sources plus every synthetic_v<version>_*.csv in the curated directory."""
    extra = sorted(p for p in CURATED_DIR.glob(f"synthetic_v{version}_*.csv") if not p.stem.endswith("_dropped"))
    return FIXED_SOURCES + [(p, p.stem.replace(f"synthetic_v{version}_", "gen_")) for p in extra]


def load_sources(version):
    rows, seen = [], set()
    for path, tag in sources_for(version):
        if not path.exists():
            print(f"  (missing, skipped) {path.relative_to(REPO_ROOT)}")
            continue
        n = 0
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                text = (r.get("text") or "").strip()
                if not text or r.get("label") not in LABELS:
                    continue
                key = text_key(text)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"key": key, "text": text, "source_label": r["label"], "source": tag,
                             "cluster": cluster_key(text)})
                n += 1
        print(f"  {n:6d} unique rows from {path.relative_to(REPO_ROOT)}")
    return rows


def cluster_key(text):
    return hashlib.sha256(la._template_key(text).encode("utf-8")).hexdigest()[:12]


def heldout(text):
    return int(hashlib.sha256(("heldout:" + text).encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF < HELDOUT_FRAC


def review_path(version):
    return CURATED_DIR / f"label_review_{version}.csv"


def queue_path(version):
    return CURATED_DIR / f"review_queue_{version}.csv"


def load_review(version):
    decisions = {}
    path = review_path(version)
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                d = (r.get("decision") or "").strip().lower()
                if d:
                    if d not in DECISIONS:
                        raise SystemExit(f"{path.name}: bad decision {d!r} for {r['id']}")
                    decisions[r["id"]] = d
    return decisions


# --------------------------------------------------------------------------- #
def score(args):
    sys.path.insert(0, str(REPO_ROOT / "evals"))
    import run_eval

    rows = load_sources(args.version)
    model, tok = run_eval.load_model(Path(args.model))
    preds, confs = run_eval.predict(model, tok, [r["text"] for r in rows], batch_size=64)
    out = [{"key": r["key"], "pred": p, "conf": round(c, 4)} for r, p, c in zip(rows, preds, confs)]
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"scored {len(out)} rows with {args.model} -> {args.out}")


def flag(args):
    scored = {r["key"]: (r["pred"], r["conf"]) for r in json.loads(Path(args.scored).read_text(encoding="utf-8"))}
    rows = load_sources(args.version)
    clusters, stats = defaultdict(list), Counter()
    for r in rows:
        rule_label, rule = la.apply_rules(r["text"], r["source_label"])
        if rule_label == REMOVE:
            stats["rule_removed"] += 1
            continue
        pred, conf = scored.get(r["key"], (None, 0.0))
        if pred is not None and pred != rule_label and conf >= args.confidence:
            clusters[r["cluster"]].append((r, rule_label, pred, conf))
            stats["flagged_rows"] += 1
    done = load_review(args.version)
    with open(queue_path(args.version), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "rows", "current_label", "model_label", "decision"])
        for cid, items in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
            r, rule_label, pred, conf = max(items, key=lambda it: it[3])
            w.writerow([cid, r["text"], len(items), rule_label, pred, done.get(cid, "")])
    stats["flagged_clusters"] = len(clusters)
    stats["already_decided"] = sum(1 for c in clusters if c in done)
    print(json.dumps(stats, indent=2))
    print(f"wrote {queue_path(args.version).relative_to(REPO_ROOT)}")


def build(args):
    rows = load_sources(args.version)
    decisions = load_review(args.version)
    flagged = set()
    if queue_path(args.version).exists():
        with open(queue_path(args.version), newline="", encoding="utf-8") as f:
            flagged = {r["id"] for r in csv.DictReader(f)}
    excl = EvalExclusion()

    train, held, changes, stats = [], [], [], Counter()
    for r in rows:
        rule_label, rule = la.apply_rules(r["text"], r["source_label"])
        if rule_label == REMOVE:
            final, why = REMOVE, f"rule:{rule}"
        elif r["cluster"] in decisions:
            final, why = decisions[r["cluster"]], "review"
            if final == "unsure":
                final = REMOVE
        elif r["cluster"] in flagged:
            final, why = REMOVE, "flagged_unreviewed"
        else:
            final, why = rule_label, f"rule:{rule}" if rule else "kept"
        stats[why.split(":")[0]] += 1
        if final != r["source_label"]:
            changes.append({"key": r["key"], "source": r["source"], "source_label": r["source_label"],
                            "final_label": final, "reason": why, "text": r["text"]})
        if final == REMOVE:
            stats["removed"] += 1
            continue
        reason = excl.why_excluded(r["text"])
        if reason:
            stats[f"excluded_{reason}"] += 1
            continue
        out = {"text": r["text"], "label": final, "source": r["source"]}
        (held if r["source"] == "corpus" and heldout(r["text"]) else train).append(out)

    train_csv = CURATED_DIR / f"train_v{args.version}.csv"
    held_csv = CURATED_DIR / f"heldout_v{args.version}.csv"
    changes_csv = CURATED_DIR / f"label_changes_v{args.version}.csv"
    for path, data in ((train_csv, train), (held_csv, held)):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["text", "label", "source"])
            w.writeheader()
            w.writerows(data)
    with open(changes_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["key", "source", "source_label", "final_label", "reason", "text"])
        w.writeheader()
        w.writerows(changes)
    summary = {"version": args.version, "source_rows": len(rows), "train_rows": len(train), "heldout_rows": len(held),
               "train_labels": dict(Counter(r["label"] for r in train)),
               "train_sources": dict(Counter(r["source"] for r in train).most_common()),
               "label_changes": len(changes),
               "change_flows": dict(Counter(f"{c['source_label']}->{c['final_label']}" for c in changes).most_common()),
               "review_decisions": len(decisions), "flagged_clusters": len(flagged),
               "counts": dict(sorted(stats.items()))}
    (CURATED_DIR / f"summary_v{args.version}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {train_csv.relative_to(REPO_ROOT)}, {held_csv.relative_to(REPO_ROOT)}, "
          f"{changes_csv.relative_to(REPO_ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("score")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--version", default="2.12")
    p = sub.add_parser("flag")
    p.add_argument("--scored", required=True)
    p.add_argument("--version", default="2.12")
    p.add_argument("--confidence", type=float, default=FLAG_CONFIDENCE)
    p = sub.add_parser("build")
    p.add_argument("--version", default="2.12")
    args = ap.parse_args()
    {"score": score, "flag": flag, "build": build}[args.cmd](args)


if __name__ == "__main__":
    main()
