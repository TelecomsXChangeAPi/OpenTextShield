#!/usr/bin/env python3
"""Label the whole training corpus with TypeSafe, then build a distillation set.

Model 2.7 learned the labeling guide from 9.4k rows. Here TypeSafe (jev) reads
docs/LABELING_GUIDE.md, as encoded in evals/label_audit.py, and answers for every
unique text in the corpus, the synthetic sets and the generated archetypes. The
answers (choice plus class probabilities) are cached in JSONL and become the
targets for evals/finetune_distill.py: hard label from the teacher's choice,
soft targets from its probabilities.

    python evals/distill_labels.py ask [--limit N] [--concurrency 12]   # needs TYPESAFE_API_KEY
    python evals/distill_labels.py build                                # offline, writes the CSVs

Eval texts (fable5, Mishra, hard_legit, the 100-message bench set, any
prediction dumps) are excluded by exact text and by template shape, so nothing
the models are scored on is ever asked about or trained on. A deterministic 5%
of the labeled corpus is held out by text hash for calibration and reporting.
"""
import argparse
import asyncio
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_audit as la  # noqa: E402
from label_audit import (QUESTIONS, QUESTIONS_VERSION, TYPESAFE_MODEL, REPO_ROOT, DATASET_DIR,  # noqa: E402
                         CURATED_DIR, text_key, _answer_record, _template_key, _eval_texts)

CORPUS_CSV = DATASET_DIR / "sms_spam_phishing_dataset_v2.4.1_dedup.csv"
SOURCES = [  # (path, source tag)
    (CORPUS_CSV, "corpus"),
    (DATASET_DIR / "synthetic_fable5_v1.csv", "synthetic_fable5"),
    (CURATED_DIR / "synthetic_notice_pairs_v1.csv", "notice_pairs"),
    (CURATED_DIR / "synthetic_archetypes_v1.csv", "archetypes"),
    (CURATED_DIR / "synthetic_legit_notices_v1.csv", "legit_notices"),
    (CURATED_DIR / "synthetic_notice_twins_v1.csv", "notice_twins"),
]
EXTRA_EVAL_FILES = [REPO_ROOT / "evals/datasets/hard_legit_a2p_v1.csv",
                    REPO_ROOT / "evals/datasets/bench100_v1.csv"]
CORPUS_ANSWERS = CURATED_DIR / "typesafe_corpus_answers.jsonl"
DISTILL_TRAIN = CURATED_DIR / "distill_v2.9_train.csv"
DISTILL_HELDOUT = CURATED_DIR / "distill_v2.9_heldout.csv"
SUMMARY = CURATED_DIR / "distill_v2.9_summary.json"
HELDOUT_FRAC = 0.05
CLASSES = ("ham", "spam", "phishing", "no_message")
MIN_CONFIDENCE = 0.6      # teacher choice below this is dropped from training
SYNTHETIC_OVERRIDE_CONFIDENCE = 0.9  # a generated row whose intended label the teacher rejects needs this to stay


def load_sources():
    rows, seen = [], set()
    for path, tag in SOURCES:
        if not path.exists():
            print(f"  (missing, skipped) {path.relative_to(REPO_ROOT)}")
            continue
        n = 0
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                text = (r.get("text") or "").strip()
                if not text or r.get("label") not in la.LABELS:
                    continue
                key = text_key(text)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"key": key, "text": text, "corpus_label": r["label"], "source": tag})
                n += 1
        print(f"  {n:6d} unique rows from {path.relative_to(REPO_ROOT)}")
    return rows


NEAR_DUP_FILES = [REPO_ROOT / "evals/datasets/bench100_v1.csv", REPO_ROOT / "evals/datasets/hard_legit_a2p_v1.csv",
                  REPO_ROOT / "evals/datasets/fable5_adversarial_v1.csv"]
NEAR_DUP_JACCARD = 0.5
_TOK = re.compile(r"[\w']+", re.U)


def _tokens(text):
    return frozenset(w for w in _TOK.findall(text.lower()) if not w.isdigit())


def eval_exclusion():
    texts = _eval_texts()
    for path in EXTRA_EVAL_FILES:
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                texts += [r.get("text") or "" for r in csv.DictReader(f)]
            print(f"  excluding eval texts from {path.relative_to(REPO_ROOT)}")
    exact = {text_key(t) for t in texts if t}
    shapes = {_template_key(t) for t in texts if t}
    near = []
    for path in NEAR_DUP_FILES:
        with open(path, newline="", encoding="utf-8") as f:
            near += [_tokens(r.get("text") or "") for r in csv.DictReader(f)]
    return exact, shapes, [t for t in near if t]


def _near_eval(text, near):
    toks = _tokens(text)
    if not toks:
        return False
    for e in near:
        inter = len(toks & e)
        if inter and inter / len(toks | e) >= NEAR_DUP_JACCARD:
            return True
    return False


def filtered_rows():
    rows = load_sources()
    exact, shapes, near = eval_exclusion()
    kept, dropped = [], Counter()
    for r in rows:
        if r["key"] in exact or _template_key(r["text"]) in shapes:
            dropped["exact_or_shape"] += 1
        elif _near_eval(r["text"], near):
            dropped["near_duplicate"] += 1
        else:
            kept.append(r)
    print(f"  dropped for overlapping an eval set: {dict(dropped)}; {len(kept)} remain")
    return kept


def load_all_answers():
    answers = la.load_answers()  # the audit's cache, same questions
    if CORPUS_ANSWERS.exists():
        for line in CORPUS_ANSWERS.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut short by a crash mid-write; it is simply asked again
            if rec.get("questions_version") == QUESTIONS_VERSION:
                answers[rec["key"]] = rec
    return answers


def ask(args):
    from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

    done = load_all_answers()
    todo = [r for r in filtered_rows() if r["key"] not in done]
    todo.sort(key=lambda r: r["source"] == "corpus")  # generated rows first: they are the gap the corpus lacks
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done)} texts already answered, asking about {len(todo)} more "
          f"(model {TYPESAFE_MODEL}, concurrency {args.concurrency})", flush=True)

    async def run():
        sem = asyncio.Semaphore(args.concurrency)
        retry = RetryPolicy(max_retries=6, backoff_initial=1.0, backoff_max=30.0, timeout=60.0)
        tokens = errors = finished = 0
        async with AsyncTypeSafeClient(retry=retry) as client:
            with open(CORPUS_ANSWERS, "a", encoding="utf-8") as out:
                async def one(r):
                    nonlocal tokens, errors, finished
                    async with sem:
                        try:
                            resp = await client.system_one({"sms": r["text"]}, QUESTIONS, model=TYPESAFE_MODEL)
                        except Exception as e:
                            errors += 1
                            if errors <= 20 or errors % 100 == 0:
                                print(f"  error on {r['key']}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                            return
                    rec = _answer_record(r["key"], resp)
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    tokens += rec["usage_in"]
                    finished += 1
                    if finished % 1000 == 0:
                        out.flush()
                        print(f"  {finished}/{len(todo)} answered, {errors} errors, {tokens:,} input tokens", flush=True)
                # bounded fan-out so 137k coroutines are not created at once
                pending = set()
                for r in todo:
                    pending.add(asyncio.create_task(one(r)))
                    if len(pending) >= args.concurrency * 4:
                        _, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                await asyncio.gather(*pending)
        print(f"done: {finished} answered, {errors} errors, {tokens:,} input tokens", flush=True)

    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(run())


def build(args):
    answers = load_all_answers()
    rows = filtered_rows()
    fields = ["text", "label", "corpus_label", "source", "confidence", "p_ham", "p_spam", "p_phishing", "p_no_message"]
    counts = Counter()
    changed = Counter()
    train, heldout = [], []
    for r in rows:
        a = answers.get(r["key"])
        if not a or "label" not in a["answers"]:
            if args.trust_unlabeled_synthetic and r["source"] != "corpus":
                # No teacher verdict yet (the TypeSafe run stopped early). Keep the
                # generator's intended label with a softened target; a later `ask`
                # + `build` replaces these with the teacher's answers.
                counts[f"generator_label_{r['source']}"] += 1
                soft = {c: (0.9 if c == r["corpus_label"] else 0.05) for c in la.LABELS}
                train.append({"text": r["text"], "label": r["corpus_label"], "corpus_label": r["corpus_label"],
                              "source": r["source"], "confidence": "", **{f"p_{c}": soft[c] for c in la.LABELS},
                              "p_no_message": 0.0})
                continue
            counts["unlabeled"] += 1
            continue
        ans = a["answers"]["label"]
        probs = {c: ans["probabilities"].get(c, 0.0) for c in CLASSES}
        if ans["choice"] == "no_message":
            counts["dropped_no_message"] += 1
            continue
        if ans["confidence"] < MIN_CONFIDENCE:
            counts["dropped_low_confidence"] += 1
            continue
        if r["source"] != "corpus" and ans["choice"] != r["corpus_label"] and ans["confidence"] < SYNTHETIC_OVERRIDE_CONFIDENCE:
            counts[f"dropped_synthetic_disagreement_{r['source']}"] += 1
            continue
        # renormalise the three real classes for the soft targets
        z = sum(probs[c] for c in la.LABELS) or 1.0
        rec = {"text": r["text"], "label": ans["choice"], "corpus_label": r["corpus_label"], "source": r["source"],
               "confidence": ans["confidence"], **{f"p_{c}": round(probs[c] / z, 4) for c in la.LABELS},
               "p_no_message": probs["no_message"]}
        if ans["choice"] != r["corpus_label"]:
            changed[f"{r['corpus_label']}->{ans['choice']}"] += 1
        # deterministic hold-out by text hash; synthetic rows all train
        if r["source"] == "corpus" and int(r["key"][:4], 16) / 65536 < HELDOUT_FRAC:
            heldout.append(rec)
        else:
            train.append(rec)
        counts[f"kept_{ans['choice']}"] += 1
    for path, data in ((DISTILL_TRAIN, train), (DISTILL_HELDOUT, heldout)):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
    summary = {"teacher_model": TYPESAFE_MODEL, "questions_version": QUESTIONS_VERSION, "min_confidence": MIN_CONFIDENCE,
               "heldout_frac": HELDOUT_FRAC, "train_rows": len(train), "heldout_rows": len(heldout),
               "train_labels": dict(Counter(r["label"] for r in train)),
               "train_sources": dict(Counter(r["source"] for r in train)),
               "counts": dict(counts), "label_changes": dict(changed.most_common())}
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ask"); p.add_argument("--limit", type=int); p.add_argument("--concurrency", type=int, default=12)
    p.set_defaults(fn=ask)
    p = sub.add_parser("build")
    p.add_argument("--trust-unlabeled-synthetic", action="store_true",
                   help="keep generated rows the teacher has not answered yet, on their intended label")
    p.set_defaults(fn=build)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
