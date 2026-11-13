#!/usr/bin/env python3
"""Clean the model 2.7 training subset against docs/LABELING_GUIDE.md.

TypeSafe is only a reviewer here: it flags rows whose label looks wrong and a
person decides. Its answers are saved in the repo, so `build` never needs
TypeSafe or an API key, and nothing from client traffic is ever sent to it.

    # 1. Freeze the exact rows model 2.7 trained on (needs torch, like finetune_tier1.py)
    python evals/label_audit.py freeze
    # 2. Ask TypeSafe about every unique text (needs typesafe-sdk and TYPESAFE_API_KEY)
    python evals/label_audit.py ask [--limit N]
    # 3. Apply the approved rules, TypeSafe flags and any human decisions
    python evals/label_audit.py build

Additions (step 4): real advertising spam and real legitimate notices from the
rest of the corpus, kept only when the corpus label and TypeSafe agree.

    python evals/label_audit.py candidates    # sample candidate rows once
    python evals/label_audit.py ask --file candidates
    python evals/label_audit.py build         # also writes additions and the combined file

Review loop: fill the `decision` column of label_review.csv with ham, spam,
phishing or remove, then run `build` again. Flagged rows without a decision
stay out of the cleaned training file, as the guide says for unsure rows.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "src/mBERT/training/model-training/dataset"
CURATED_DIR = DATASET_DIR / "curated"
SYNTHETIC_CSV = DATASET_DIR / "synthetic_fable5_v1.csv"
ORIGINAL_CSV = DATASET_DIR / "sms_spam_phishing_dataset_v2.4_combined.csv"
SUBSET_CSV = CURATED_DIR / "train_subset_v2.7.csv"
ANSWERS_JSONL = CURATED_DIR / "typesafe_answers.jsonl"
CLEANED_CSV = CURATED_DIR / "train_subset_v2.7_cleaned.csv"
REVIEW_CSV = CURATED_DIR / "label_review.csv"
SUMMARY_JSON = CURATED_DIR / "summary.json"
CANDIDATES_CSV = CURATED_DIR / "additions_candidates.csv"
# Generated notice pairs (evals/generate_notice_pairs.py), checked the same way.
# Off by default in `build`: they raise the block rate but double false blocks on
# real ham. See evals/results/DATA_CLEANUP_2.8.md. Enable with --with-notice-pairs.
NOTICE_PAIRS_CSV = CURATED_DIR / "synthetic_notice_pairs_v1.csv"
ADDITIONS_CSV = CURATED_DIR / "additions_v1.csv"
COMBINED_CSV = CURATED_DIR / "train_v2.8_candidate.csv"
# Eval texts that must never enter training. Tracked files, plus any local
# prediction dumps (they hold the UCI and IMC25 texts, which are downloaded).
EVAL_TEXT_FILES = [REPO_ROOT / "evals/datasets/fable5_adversarial_v1.csv",
                   REPO_ROOT / "evals/datasets/mishra_soni_5971.csv",
                   REPO_ROOT / "benchmark/test_dataset.json"]

LABELS = ("ham", "spam", "phishing")
REMOVE = "remove"
TYPESAFE_MODEL = "jev-1.13.0"
FLAG_CONFIDENCE = 0.7
AGREE_CONFIDENCE = 0.9
CONCURRENCY = 6

csv.field_size_limit(10 * 1024 * 1024)


# --------------------------------------------------------------------------- #
# freeze
# --------------------------------------------------------------------------- #
def freeze(args):
    """Rebuild model 2.7's training rows with finetune_tier1's own sampler."""
    sys.path.insert(0, str(REPO_ROOT / "evals"))
    from finetune_tier1 import ID2LABEL, read_csv

    rows = [("synthetic", t, l) for t, l in read_csv(SYNTHETIC_CSV, seed=args.seed)]
    rows += [("rehearsal", t, l) for t, l in read_csv(ORIGINAL_CSV, limit_per_label=args.orig_per_label, seed=args.seed)]
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUBSET_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "text", "label"])
        for i, (source, text, label_id) in enumerate(rows):
            w.writerow([f"{source[0]}{i:05d}", source, text, ID2LABEL[label_id]])
    print(f"wrote {SUBSET_CSV.relative_to(REPO_ROOT)}: {len(rows)} rows, "
          f"{dict(Counter((s, ID2LABEL[l]) for s, _, l in rows))}")


def load_subset():
    with open(SUBSET_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# candidates: rows to add in step 4
# --------------------------------------------------------------------------- #
_NOTICE = re.compile(
    r"(\b(otp|code|kode|código|codice|verification|verif|passcode|pin)\b|delivered|delivery|shipped|parcel|"
    r"package|paket|paquete|colis|pacco|\border\b|pedido|pesanan|appointment|reminder|payment|paid|debited|"
    r"credited|balance|saldo|\bbill\b|tagihan|factura|recharge|isi ulang|transaction|transaksi|account|akun|"
    r"rekening|cuenta|compte|konto|flight|booking|reservation|ticket)", re.I)
# Senders whose templates were translated into many languages; cap each so a
# few families don't dominate the additions.
_FAMILY = re.compile(r"(voicespin|shopee|netflix|gojek|telkomsel|axis|indosat|whatsapp|google|uber|amazon|"
                     r"paypal|linkaja|dana|ovo|tokopedia|grab|line)", re.I)
FAMILY_CAP = 40


def _template_key(text):
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", text.lower())).strip()


def _eval_texts():
    texts = []
    for path in EVAL_TEXT_FILES + sorted((REPO_ROOT / "evals/results").glob("predictions_*.json")):
        if not path.exists():
            continue
        if "distill_v" in path.name and "heldout" in path.name:
            continue  # the teacher-labeled hold-out is carved from the corpus by hash, not an eval set
        if path.name.endswith("_typesafe.json"):
            continue  # a TypeSafe head-to-head dump; its texts are already covered by the bench CSV
        if path.suffix == ".csv":
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                texts += [r.get("text") or r.get("message") or "" for r in csv.DictReader(f)]
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data if isinstance(data, list) else data.get("samples") or data.get("messages") or []
            texts += [r.get("text", "") for r in rows if isinstance(r, dict)]
        print(f"  excluding eval texts from {path.relative_to(REPO_ROOT)}")
    return texts


def candidates(args):
    """Sample corpus rows outside the training subset and every eval set."""
    import random

    exclude = {_template_key(r["text"]) for r in load_subset()}
    exclude |= {_template_key(t) for t in _eval_texts() if t}
    pools = {"ads_spam": [], "legit_notice": [], "notice_phish": []}
    seen, families = set(), Counter()
    with open(ORIGINAL_CSV, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    for row in rows:
        text, label = (row.get("text") or "").strip(), row.get("label")
        if label not in ("spam", "ham", "phishing") or row.get("augmentation_type") not in ("", "original"):
            continue
        key = _template_key(text)
        if not text or key in exclude or key in seen or apply_rules(text, label)[1]:
            continue
        seen.add(key)
        if label == "spam":
            pools["ads_spam"].append(text)
        elif _NOTICE.search(text):
            # Legitimate notices and notice-shaped lures are added in pairs: on
            # their own, the benign ones teach the model to pass anything shaped
            # like a bank or parcel message.
            family = _FAMILY.search(text)
            if family:
                name = f"{label}:{family.group(1).lower()}"
                if families[name] >= FAMILY_CAP:
                    continue
                families[name] += 1
            pools["legit_notice" if label == "ham" else "notice_phish"].append(text)
    sizes = {"ads_spam": args.spam, "legit_notice": args.notices, "notice_phish": args.notice_phish}
    with open(CANDIDATES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "pool", "text", "label"])
        i = 0
        for pool, texts in pools.items():
            for text in texts[:sizes[pool]]:
                label = {"ads_spam": "spam", "legit_notice": "ham", "notice_phish": "phishing"}[pool]
                w.writerow([f"a{i:05d}", pool, text, label])
                i += 1
    print(f"wrote {CANDIDATES_CSV.relative_to(REPO_ROOT)}: "
          f"{ {p: min(len(t), sizes[p]) for p, t in pools.items()} } from pools of "
          f"{ {p: len(t) for p, t in pools.items()} }")


def load_candidates(with_notice_pairs=True):
    """Rows offered as additions: sampled corpus rows, and generated pairs if asked."""
    rows = []
    paths = (CANDIDATES_CSV, NOTICE_PAIRS_CSV) if with_notice_pairs else (CANDIDATES_CSV,)
    for path in paths:
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                rows += list(csv.DictReader(f))
    return rows


# --------------------------------------------------------------------------- #
# rules: the four approved decisions, applied by pattern
# --------------------------------------------------------------------------- #
_URL = r"(?:https?://|www\.)?[\w-]+(?:\.[\w-]+)+(?:/\S*)?"
_SEND_MONEY = (r"(send money|enviar dinero|envoyer de l.argent|geld\b.{0,15}\bsenden|inviare denaro|"
               r"mengirim uang|отправить деньги)")
RULES = [
    # (name, pattern on the matching form of the text, labels it applies to, new label)
    ("decision1_won_or_owed", re.compile(
        r"^(urgent: your prize is waiting|congrats! you won .* gift card|your entry won! claim|"
        r"win \S+ now! click \S+ to claim)"), {"spam"}, "phishing"),
    ("guide_impersonation_money_request", re.compile(
        rf"{_SEND_MONEY}.{{0,30}}\birs\b|\birs\b.{{0,30}}{_SEND_MONEY}"), {"spam"}, "phishing"),
    ("decision2_generic_job_ad", re.compile(r"^job offer: earn \S+/month\. apply at "), {"phishing"}, "spam"),
    ("decision3_code_delivery", re.compile(r"^whatsapp: new security code: \S+\.? don.t share it\.?$"),
     set(LABELS), "ham"),
    ("decision4_bare_link", re.compile(rf"^{_URL}$"), set(LABELS), REMOVE),
    # "(aytd2773838383 https://goo.gl/...". Prefix only: leet folding rewrites the digits.
    ("decision4_junk_code_link", re.compile(r"^\(?aytd"), set(LABELS), REMOVE),
    # One Latin-script word ("Alert!", "Verify"). Not "no whitespace": Chinese and
    # Japanese write whole sentences without spaces.
    ("decision4_single_word", re.compile(r"^[a-z][a-z'-]{0,19}[.!?]*$"), set(LABELS), REMOVE),
    ("decision4_translator_note", re.compile(
        r"(i'?m sorry, but|could you please provide more context|doesn'?t (seem to )?(be|have) (in|a|any)|"
        r"does not have (a )?meaning|n'a pas de sens|remain(s)? unchanged|translation: sin cambios|"
        r"no tiene sentido|keine bedeutung|ليس بواجد في اللغة|tidak bisa mengunjungi tautan|"
        # assistant replies left in the corpus by the translation step
        r"cannot provide a translation|can'?t open shortened urls|do not have the capability to see|"
        r"tidak (dapat|bisa) (membuka|mengakses|memfasilitasi)|"
        r"dapatkah anda memberikan informasi lebih lanjut tentang apa yang ingin anda sampaikan|"
        r"no tiene un significado|no hay significado en esta frase|no se puede traducir|"
        r"non ha un significato|non ci sono informazioni comprensibili|"
        r"ne semble pas avoir de signification|ich bin ein ai-modell)"),
     set(LABELS), REMOVE),
]

# The rehearsal corpus holds augmented copies of each template: a label-revealing
# prefix ("WIN: ", "Important: "), emoji and symbols sprinkled between words,
# look-alike letters and leetspeak. Rules match a folded form so an approved
# decision covers every copy. The stored text is never changed.
_AUGMENT_PREFIX = re.compile(
    r"^((win|free|promo|offer|deal|important|alert|security|update|notice|fyi|reminder):\s+|"
    r"(hey|hi),\s+|just wanted to say\s+)+", re.I)
_SYMBOLS = re.compile(r"[^\w\s:.,!?'/$£€-]+")
_LEET_WORD = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b",
                            "@": "a", "$": "s", "!": "i", "|": "l"})
_PREPROCESSOR = None


def matching_form(text):
    global _PREPROCESSOR
    if _PREPROCESSOR is None:
        sys.path.insert(0, str(REPO_ROOT))
        from src.api_interface.services.enhanced_preprocessing import EnhancedPreprocessor
        _PREPROCESSOR = EnhancedPreprocessor()
    text = _PREPROCESSOR.normalize_unicode(text).lower()
    text = _SYMBOLS.sub(" ", text)
    text = re.sub(r"\s+([:.,!?])", r"\1", re.sub(r"\s+", " ", text)).strip()
    text = _AUGMENT_PREFIX.sub("", text)
    # Undo leetspeak only inside words that mix letters and look-alike digits or
    # symbols ("53cur!ty"), so real numbers and amounts stay as they are.
    return re.sub(r"\S*[a-z]\S*", lambda m: m.group().translate(_LEET_WORD)
                  if re.search(r"[a-z]", m.group()) and re.search(r"[013457@$|]", m.group().rstrip(".!?,:"))
                  and not re.fullmatch(r"[$£€]?[\d.,]+[.!?,:]?", m.group()) else m.group(), text)


def apply_rules(text, label):
    form = matching_form(text)
    for name, pattern, applies_to, new_label in RULES:
        if label in applies_to and pattern.search(form):
            return new_label, name
    return label, ""


# --------------------------------------------------------------------------- #
# ask: TypeSafe questions written from docs/LABELING_GUIDE.md
# --------------------------------------------------------------------------- #
QUESTIONS = {
    "label": {
        "type": "choice",
        "instructions": {
            "question": "Which label does the SMS in `sms` get under this labeling guide?",
            "order": "Check the options in this order and pick the first that fits: no_message, phishing, spam, ham.",
            "focus": "Label the meaning, not the wording. Disguised or translated text gets the label of what it says.",
        },
        "criteria": {
            "no_message": {
                "what": "No real message: a bare link, a single word, a note from a translator or AI tool, or gibberish.",
                "examples": ["https://bit.ly/10287", "Alert!", "Sorry, this text has no meaning in English."],
            },
            "phishing": {
                "what": "A trick: it pretends to be a bank, delivery company, carrier, government office, brand, "
                        "employer or family member; or says the reader already won, is owed, or was picked for money, "
                        "a prize, a refund or a job; or invents a problem and pushes the reader to click, call, pay or "
                        "reply; or asks for a password, card number, code, ID number or money transfer.",
                "not_for": "A real notice on the real service, a scam warning, or an ad that does none of these things.",
                "examples": ["URGENT: Your prize is waiting! Call 0906 123 4567",
                             "You need to send money to the IRS NOW at tinyurl.com/9065",
                             "You've been selected for a remote position at Amazon. $450/day. Join: t.me/jobs",
                             "Hi, can you send me the code you just received?"],
            },
            "spam": {
                "what": "Advertising: a shop, casino, loan, crypto group, health product, MLM, political ad, or any "
                        "offer to buy, join, sign up or earn money. It may be pushy or shady.",
                "not_for": "Messages that pretend to be someone, say the reader already won, or ask for codes or money.",
                "examples": ["FREE iPhone 15! Enter code 1007 at bit.ly/123", "Earn $2,464 per week working from home!",
                             "WIN BIG! 200 FREE spins, no deposit! Play now: luckyspin.bet"],
            },
            "ham": {
                "what": "Normal chat, or a real notice from a real service, including login and verification codes.",
                "not_for": "Anything that asks the reader to send a code, password, card number or money.",
                "examples": ["Your Uber code is 7782. Never share this code.",
                             "Chase: Did you make a $612 purchase at BEST BUY? Reply YES or NO.",
                             "HMRC will never text you asking for bank details."],
            },
        },
    },
}
QUESTIONS_VERSION = hashlib.sha256(json.dumps(QUESTIONS, sort_keys=True).encode()).hexdigest()[:12]


def text_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_answers():
    answers = {}
    if ANSWERS_JSONL.exists():
        for line in ANSWERS_JSONL.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["questions_version"] == QUESTIONS_VERSION:
                answers[rec["key"]] = rec
    return answers


def _answer_record(key, response):
    out = {}
    for qid, a in response.answers.items():
        if getattr(a, "choice", None) is not None:
            out[qid] = {"choice": a.choice, "confidence": round(a.confidence, 4),
                        "probabilities": {k: round(v, 4) for k, v in a.probabilities.items()}}
    return {"key": key, "model": response.model, "questions_version": QUESTIONS_VERSION,
            "usage_in": response.usage.input_tokens, "answers": out}


def ask(args):
    from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

    done = load_answers()
    todo, seen = [], set()
    for row in (load_candidates() if args.file == "candidates" else load_subset()):
        key = text_key(row["text"])
        if key not in done and key not in seen:
            seen.add(key)
            todo.append((key, row["text"]))
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done)} texts already answered, asking about {len(todo)} more")

    async def run():
        sem = asyncio.Semaphore(CONCURRENCY)
        retry = RetryPolicy(max_retries=5, backoff_initial=1.0, backoff_max=20.0, timeout=60.0)
        tokens, errors, finished = 0, 0, 0
        async with AsyncTypeSafeClient(retry=retry) as client:
            with open(ANSWERS_JSONL, "a", encoding="utf-8") as out:
                async def one(key, text):
                    nonlocal tokens, errors, finished
                    async with sem:
                        try:
                            response = await client.system_one({"sms": text}, QUESTIONS, model=TYPESAFE_MODEL)
                        except Exception as e:  # keep going; rerun `ask` to retry the gaps
                            errors += 1
                            print(f"  error on {key}: {type(e).__name__}: {e}", file=sys.stderr)
                            return
                    rec = _answer_record(key, response)
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    tokens += rec["usage_in"]
                    finished += 1
                    if finished % 500 == 0:
                        out.flush()
                        print(f"  {finished}/{len(todo)} answered, {tokens:,} input tokens")

                await asyncio.gather(*(one(k, t) for k, t in todo))
        print(f"done: {finished} answered, {errors} errors, {tokens:,} input tokens")

    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(run())


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def load_decisions():
    """id -> (decision, decision_source) from the review file."""
    decisions = {}
    if REVIEW_CSV.exists():
        with open(REVIEW_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                decision = (row.get("decision") or "").strip().lower()
                if decision:
                    if decision not in LABELS + (REMOVE,):
                        raise SystemExit(f"{REVIEW_CSV.name}: bad decision {decision!r} for {row['id']}")
                    decisions[row["id"]] = (decision, (row.get("decision_source") or "").strip() or "manual")
    return decisions


def build(args):
    subset = load_subset()
    answers = load_answers()
    decisions = load_decisions()

    cleaned, review, stats = [], [], Counter()
    for row in subset:
        rule_label, rule = apply_rules(row["text"], row["label"])
        rec = answers.get(text_key(row["text"]))
        ts_label = ts_conf = None
        if rec:
            choice = rec["answers"]["label"]
            ts_label = REMOVE if choice["choice"] == "no_message" else choice["choice"]
            ts_conf = choice["confidence"]
        flagged = ts_label is not None and ts_label != rule_label and ts_conf >= args.flag_confidence

        if row["id"] in decisions:
            final, why = decisions[row["id"]][0], "human_decision"
        elif flagged:
            final, why = None, "flagged_unreviewed"
        elif rec is None:
            final, why = rule_label, "not_asked"
        else:
            final, why = rule_label, rule or "kept"
        stats[why] += 1
        if rule:
            stats[f"rule:{rule}"] += 1

        if flagged or row["id"] in decisions:
            review.append({"id": row["id"], "source": row["source"], "text": row["text"],
                           "dataset_label": row["label"], "rule_label": rule_label, "rule": rule,
                           "typesafe_label": ts_label, "typesafe_confidence": ts_conf,
                           "decision": decisions.get(row["id"], ("", ""))[0],
                           "decision_source": decisions.get(row["id"], ("", ""))[1]})
        if final and final != REMOVE:
            cleaned.append({"text": row["text"], "label": final})
            stats[f"final:{final}"] += 1
        elif final == REMOVE:
            stats["removed"] += 1

    review.sort(key=lambda r: (r["decision"] != "", -(r["typesafe_confidence"] or 0)))
    with open(CLEANED_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label"])
        w.writeheader()
        w.writerows(cleaned)
    with open(REVIEW_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(review[0].keys()) if review else ["id"])
        w.writeheader()
        w.writerows(review)
    # Additions: keep a candidate only when TypeSafe agrees with its corpus label.
    additions, add_stats = [], Counter()
    for row in load_candidates(with_notice_pairs=args.with_notice_pairs):
        rec = answers.get(text_key(row["text"]))
        add_stats[f"{row['pool']}:candidates"] += 1
        if rec is None:
            continue
        add_stats[f"{row['pool']}:answered"] += 1
        choice = rec["answers"]["label"]
        if choice["choice"] == row["label"] and choice["confidence"] >= AGREE_CONFIDENCE:
            additions.append({"text": row["text"], "label": row["label"], "pool": row["pool"]})
            add_stats[f"{row['pool']}:accepted"] += 1
    if additions:
        with open(ADDITIONS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["text", "label", "pool"])
            w.writeheader()
            w.writerows(additions)
        with open(COMBINED_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["text", "label"])
            w.writeheader()
            w.writerows(cleaned + [{"text": a["text"], "label": a["label"]} for a in additions])

    flows = Counter(f"{r['rule_label']}->{r['typesafe_label']}" for r in review if not r["decision"])
    summary = {"subset_rows": len(subset), "cleaned_rows": len(cleaned), "review_rows": len(review),
               "additions": dict(sorted(add_stats.items())), "agree_confidence": AGREE_CONFIDENCE,
               "with_notice_pairs": args.with_notice_pairs,
               "combined_rows": len(cleaned) + len(additions),
               "combined_labels": dict(Counter(r["label"] for r in cleaned + additions)),
               "flag_confidence": args.flag_confidence, "typesafe_model": TYPESAFE_MODEL,
               "questions_version": QUESTIONS_VERSION, "counts": dict(sorted(stats.items())),
               "unreviewed_flag_flows": dict(flows.most_common())}
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--orig-per-label", type=int, default=2500)
    p = sub.add_parser("candidates")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--spam", type=int, default=1600)
    p.add_argument("--notices", type=int, default=900)
    p.add_argument("--notice-phish", type=int, default=900)
    p = sub.add_parser("ask")
    p.add_argument("--file", choices=["subset", "candidates"], default="subset")
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("build")
    p.add_argument("--flag-confidence", type=float, default=FLAG_CONFIDENCE)
    p.add_argument("--with-notice-pairs", action="store_true",
                   help="also train on the generated notice pairs (more blocking, more false blocks)")
    args = ap.parse_args()
    {"freeze": freeze, "candidates": candidates, "ask": ask, "build": build}[args.cmd](args)


if __name__ == "__main__":
    main()
