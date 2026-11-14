#!/usr/bin/env python3
"""Labeling rules and eval-set exclusion shared by the training-set builder.

Everything here implements docs/LABELING_GUIDE.md in code:

  * RULES / apply_rules(): the four approved decisions, applied by pattern to a
    folded form of the text so one decision covers every augmented copy.
  * eval_exclusion() / is_eval_text(): the texts no model may train on, by exact
    match, template shape and token overlap, so benchmark numbers stay honest.
  * freeze: rebuilds the exact rows model 2.7 trained on (provenance only).

The training set itself is assembled by evals/build_train_set.py.

    python evals/label_audit.py freeze      # writes dataset/curated/train_subset_v2.7.csv
"""

import argparse
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
# Eval texts that must never enter training. Tracked files, plus any local
# prediction dumps (they hold the UCI and IMC25 texts, which are downloaded).
EVAL_TEXT_FILES = [REPO_ROOT / "evals/datasets/fable5_adversarial_v1.csv",
                   REPO_ROOT / "evals/datasets/mishra_soni_5971.csv",
                   REPO_ROOT / "evals/datasets/hard_legit_a2p_v1.csv",
                   REPO_ROOT / "evals/datasets/bench100_v1.csv",
                   REPO_ROOT / "benchmark/test_dataset.json"]
# Small hand-written sets also get a token-overlap check, since a paraphrase of a
# bench row teaches the model the bench.
NEAR_DUP_FILES = [REPO_ROOT / "evals/datasets/bench100_v1.csv", REPO_ROOT / "evals/datasets/hard_legit_a2p_v1.csv",
                  REPO_ROOT / "evals/datasets/fable5_adversarial_v1.csv"]
NEAR_DUP_JACCARD = 0.5

LABELS = ("ham", "spam", "phishing")
REMOVE = "remove"

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


# --------------------------------------------------------------------------- #
# keys and eval exclusion
# --------------------------------------------------------------------------- #
def text_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _template_key(text):
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", text.lower())).strip()


_TOK = re.compile(r"[\w']+", re.U)


def _tokens(text):
    return frozenset(w for w in _TOK.findall(text.lower()) if not w.isdigit())


def _eval_texts():
    texts = []
    for path in EVAL_TEXT_FILES + sorted((REPO_ROOT / "evals/results").glob("predictions_*.json")):
        if not path.exists():
            continue
        if "heldout" in path.name:
            continue  # the hold-out is carved from the corpus by hash, not an eval set
        if path.suffix == ".csv":
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                texts += [r.get("text") or r.get("message") or "" for r in csv.DictReader(f)]
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data if isinstance(data, list) else data.get("samples") or data.get("messages") or []
            texts += [r.get("text", "") for r in rows if isinstance(r, dict)]
        print(f"  excluding eval texts from {path.relative_to(REPO_ROOT)}")
    return texts


class EvalExclusion:
    """Answers "may this text be trained on?" for every eval set we score."""

    def __init__(self):
        texts = _eval_texts()
        self.exact = {text_key(t) for t in texts if t}
        self.shapes = {_template_key(t) for t in texts if t}
        self.near = []
        for path in NEAR_DUP_FILES:
            with open(path, newline="", encoding="utf-8") as f:
                self.near += [t for t in (_tokens(r.get("text") or "") for r in csv.DictReader(f)) if t]

    def why_excluded(self, text):
        if text_key(text) in self.exact or _template_key(text) in self.shapes:
            return "exact_or_shape"
        toks = _tokens(text)
        if toks:
            for e in self.near:
                inter = len(toks & e)
                if inter and inter / len(toks | e) >= NEAR_DUP_JACCARD:
                    return "near_duplicate"
        return ""


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--orig-per-label", type=int, default=2500)
    args = ap.parse_args()
    {"freeze": freeze}[args.cmd](args)


if __name__ == "__main__":
    main()
