#!/usr/bin/env python3
"""Daily probe: invent realistic SMS with a cheap model, classify them with OTS,
grade OTS with a stronger model, and log what the next training round should fix.

    python evals/probe/probe.py run  [--n 150] [--api https://ots.telecomsxchange.com]
    python evals/probe/probe.py report
    python evals/probe/probe.py run --dry-run      # no LLM calls: plumbing check on bench100

Each `run` writes logs/<date>.jsonl (one row per message: text, intended label,
OTS answer, grader verdict), appends to logs/improvements.log (one line per
miss, human-readable), and refreshes REPORT.md plus logs/training_additions.csv
(the graded gold labels for every miss, ready for evals/distill_labels.py).
The generator is Claude Haiku 4.5; the grader is Claude Opus 5 reading
docs/LABELING_GUIDE.md. Both need ANTHROPIC_API_KEY (or an `ant auth login`
profile). Texts are de-duplicated against every earlier run so the set keeps
growing instead of repeating.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
GUIDE = next((p for p in (Path(os.environ.get("OTS_PROBE_GUIDE", "")), REPO / "docs/LABELING_GUIDE.md", HERE / "docs/LABELING_GUIDE.md")
              if str(p) and p.is_file()), REPO / "docs/LABELING_GUIDE.md")  # repo layout, or the copy sync.sh --push installs
LOG_DIR = Path(os.environ.get("OTS_PROBE_LOG_DIR", HERE / "logs"))
REPORT = Path(os.environ.get("OTS_PROBE_REPORT", HERE / "REPORT.md"))
GENERATOR_MODEL = os.environ.get("OTS_PROBE_GENERATOR", "claude-haiku-4-5")
GRADER_MODEL = os.environ.get("OTS_PROBE_GRADER", "claude-opus-5")
DEFAULT_API = os.environ.get("OTS_PROBE_API", "https://ots.telecomsxchange.com")
LABELS = ("ham", "spam", "phishing")

# Themes rotate by day so the daily sets cover different ground. Each run mixes
# several of these into one brief for the generator.
SECTORS = ["banking and cards", "parcel delivery", "mobile carriers and utilities", "online shopping and marketplaces",
           "government, tax and benefits", "healthcare and pharmacies", "travel, rides and hotels", "streaming and subscriptions",
           "employment and recruiting", "schools, clubs and community", "crypto and investing", "dating and social",
           "charities and fundraising", "real estate and rentals", "gaming and prizes"]
LANGS = ["English", "Spanish", "French", "German", "Portuguese", "Italian", "Dutch", "Arabic", "Hindi and Hinglish", "Indonesian",
         "Russian", "Turkish", "Greek", "Polish", "Tamil", "Swahili", "Tagalog", "Vietnamese", "Thai", "Japanese"]
STYLES = ["short and casual", "formal and complete", "typos and abbreviations", "ALL CAPS urgency", "emoji-heavy",
          "leetspeak or spaced-out letters to dodge filters", "multi-part message fragments", "long with legal footer",
          "wrong-number or cold opener with no ask yet", "reply-based (reply YES, STOP, C, 1)"]


# ----------------------------------------------------------------------------- schemas
from pydantic import BaseModel, Field  # noqa: E402


class GeneratedMessage(BaseModel):
    text: str = Field(description="The SMS exactly as a phone would show it, 20-320 characters, no placeholders like [name] or <link>")
    intended_label: Literal["ham", "spam", "phishing"]
    category: str = Field(description="short slug, e.g. bank_fraud_check, delivery_fee_lure, family_new_number, gym_promo")
    language: str = Field(description="ISO 639-1 code, e.g. en, es, ar")
    why: str = Field(description="one sentence: what makes it that label under the guide")


class GeneratedBatch(BaseModel):
    messages: List[GeneratedMessage]


class Verdict(BaseModel):
    id: int
    gold: Literal["ham", "spam", "phishing", "unusable"] = Field(
        description="the correct label under the guide; 'unusable' if the message is unrealistic, ambiguous or not a real SMS")
    ots_correct: bool
    error_type: Literal["none", "false_block", "missed_threat", "category_swap", "unusable"]
    severity: int = Field(description="0 none, 1 minor category swap, 2 real spam/phishing passed or a real notice blocked at low confidence, 3 the same at high confidence")
    note: str = Field(description="one sentence on what OTS would need to learn to get this right; empty if correct")


class GradedBatch(BaseModel):
    verdicts: List[Verdict]


# ----------------------------------------------------------------------------- helpers
def norm(text):
    return re.sub(r"\W+", " ", text.lower()).strip()


def key(text):
    return hashlib.sha1(norm(text).encode()).hexdigest()[:16]


def load_seen():
    seen = set()
    for p in LOG_DIR.glob("*.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                pass
    return seen


def guide_text():
    return GUIDE.read_text(encoding="utf-8")


def client():
    import anthropic
    return anthropic.Anthropic(max_retries=4, timeout=180.0)


# ----------------------------------------------------------------------------- generate
def generate(n, seed, dry_run=False):
    if dry_run:
        rows = list(csv.DictReader(open(REPO / "evals/datasets/bench100_v1.csv", encoding="utf-8")))
        return [GeneratedMessage(text=r["text"], intended_label=r["label"], category="bench", language="xx", why="dry run")
                for r in rows[:n]]
    rng = random.Random(seed)
    c = client()
    out, calls = [], 0
    per_call = 40
    while len(out) < n and calls < (n // per_call) + 3:
        calls += 1
        sectors = rng.sample(SECTORS, 3)
        langs = ["English"] + rng.sample(LANGS[1:], 3)
        styles = rng.sample(STYLES, 4)
        brief = (
            f"Write {per_call} distinct SMS messages a real phone might receive today. Mix: about 40% ham (real notices from real "
            f"services and ordinary personal chat), 25% spam (unsolicited advertising that does not impersonate anyone), 35% phishing "
            f"(impersonation, prize or job lures, invented problems that push a click, call, payment or reply, requests for codes, "
            f"passwords or money). Sectors to draw from: {', '.join(sectors)}. Languages: {', '.join(langs)}, at least 6 messages "
            f"not in English. Styles to include: {', '.join(styles)}.\n\n"
            "Make them hard: for every attack include a benign twin that uses the same brand and topic but asks for nothing "
            "suspicious (real OTPs, real fraud checks, real delivery updates with the courier's real domain). Real notices use real "
            "domains (chase.com, dhl.com, myvzw.com); lures use look-alikes, shorteners or a call-back number. Vary amounts, names, "
            "dates and phrasing; never reuse a sentence. No placeholders. Label each one strictly by the guide below.\n\n"
            f"<labeling_guide>\n{guide_text()}\n</labeling_guide>"
        )
        resp = c.messages.parse(model=GENERATOR_MODEL, max_tokens=8000,
                                messages=[{"role": "user", "content": brief}], output_format=GeneratedBatch)
        if resp.stop_reason == "refusal":
            print("  generator refused a batch; continuing", file=sys.stderr)
            continue
        out += resp.parsed_output.messages
        print(f"  generated {len(out)} (call {calls}, {resp.usage.input_tokens}+{resp.usage.output_tokens} tokens)", flush=True)
    return out[:n]


# ----------------------------------------------------------------------------- classify
def classify(api, text):
    req = urllib.request.Request(f"{api.rstrip('/')}/predict/", data=json.dumps({"text": text, "model": "ots-mbert"}).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "ots-probe/1.0"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r:
        j = json.load(r)
    return {"ots_label": j["label"], "ots_prob": round(float(j["probability"]), 4), "latency_ms": round((time.time() - t0) * 1000),
            "model_version": j.get("model_info", {}).get("version")}


# ----------------------------------------------------------------------------- grade
def grade(rows, dry_run=False):
    if dry_run:
        out = []
        for i, r in enumerate(rows):
            ok = r["ots_label"] == r["intended_label"]
            et = "none" if ok else ("false_block" if r["intended_label"] == "ham" else "missed_threat" if r["ots_label"] == "ham" else "category_swap")
            out.append(Verdict(id=i, gold=r["intended_label"], ots_correct=ok, error_type=et, severity=0 if ok else 2, note="" if ok else "dry run"))
        return out
    c = client()
    system = [{"type": "text", "text": (
        "You grade an SMS spam and phishing classifier. For each message you get the text, the label the message writer "
        "intended, and the classifier's answer with its confidence. Decide the correct label yourself, strictly by the guide, "
        "and do not trust the writer's intended label: if it is wrong, say so through `gold`. Mark a message `unusable` when it "
        "is not something a phone would plausibly receive, has placeholders, or is genuinely ambiguous even for a careful human. "
        "Error types: false_block = a ham message the classifier called spam or phishing; missed_threat = spam or phishing "
        "called ham; category_swap = spam and phishing confused (the gateway rejects both, so this is minor). "
        "The note must say what pattern the classifier failed to recognise, in terms a data engineer can act on.\n\n"
        f"<labeling_guide>\n{guide_text()}\n</labeling_guide>"), "cache_control": {"type": "ephemeral"}}]
    verdicts = []
    for start in range(0, len(rows), 30):
        chunk = rows[start:start + 30]
        listing = "\n".join(
            f"[{start + i}] intended={r['intended_label']} ots={r['ots_label']} (p={r['ots_prob']}) lang={r['language']} :: {r['text']}"
            for i, r in enumerate(chunk))
        resp = c.messages.parse(model=GRADER_MODEL, max_tokens=12000, system=system, output_config={"effort": "medium"},
                                messages=[{"role": "user", "content": f"Grade every message, ids {start} to {start + len(chunk) - 1}:\n\n{listing}"}],
                                output_format=GradedBatch)
        if resp.stop_reason == "refusal":
            print("  grader refused a batch; marking its rows unusable", file=sys.stderr)
            verdicts += [Verdict(id=start + i, gold="unusable", ots_correct=False, error_type="unusable", severity=0, note="grader refused") for i in range(len(chunk))]
            continue
        got = {v.id: v for v in resp.parsed_output.verdicts}
        for i in range(len(chunk)):
            verdicts.append(got.get(start + i) or Verdict(id=start + i, gold="unusable", ots_correct=False, error_type="unusable", severity=0, note="no verdict returned"))
        print(f"  graded {start + len(chunk)}/{len(rows)} (cache read {resp.usage.cache_read_input_tokens} tokens)", flush=True)
    return verdicts


# ----------------------------------------------------------------------------- run
def run(args):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    day = date.today().isoformat()
    seed = int(hashlib.sha1(day.encode()).hexdigest(), 16) % (2 ** 31)
    seen = load_seen()
    print(f"[{day}] generating {args.n} messages with {GENERATOR_MODEL} ({len(seen)} texts already seen)", flush=True)
    gen = generate(args.n, seed, args.dry_run)
    rows = []
    for g in gen:
        k = key(g.text)
        if k in seen or len(g.text.strip()) < 15:
            continue
        seen.add(k)
        rows.append({"key": k, "text": g.text.strip(), "intended_label": g.intended_label, "category": g.category,
                     "language": g.language, "why": g.why})
    print(f"  {len(rows)} new messages after de-duplication; classifying via {args.api}", flush=True)
    for r in rows:
        try:
            r.update(classify(args.api, r["text"]))
        except Exception as e:  # keep going; the row records the failure
            r.update({"ots_label": "error", "ots_prob": 0.0, "latency_ms": None, "model_version": None, "error": str(e)[:200]})
    ok_rows = [r for r in rows if r["ots_label"] in LABELS]
    print(f"  {len(ok_rows)} classified; grading with {GRADER_MODEL}", flush=True)
    verdicts = grade(ok_rows, args.dry_run)
    by_id = {v.id: v for v in verdicts}
    out_path = LOG_DIR / f"{day}.jsonl"
    imp_path = LOG_DIR / "improvements.log"
    n_written = 0
    with open(out_path, "a", encoding="utf-8") as f, open(imp_path, "a", encoding="utf-8") as imp:
        for i, r in enumerate(ok_rows):
            v = by_id.get(i)
            rec = {**r, "gold": v.gold, "ots_correct": v.ots_correct, "error_type": v.error_type, "severity": v.severity,
                   "note": v.note, "graded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "generator": GENERATOR_MODEL, "grader": GRADER_MODEL}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
            if v.error_type not in ("none", "unusable"):
                imp.write(f"{day} sev{v.severity} {v.error_type:14} gold={v.gold:8} ots={r['ots_label']}@{r['ots_prob']} "
                          f"[{r['language']}/{r['category']}] {r['text'][:110]!r} -> {v.note}\n")
    usable = [by_id[i] for i in range(len(ok_rows)) if by_id[i].gold != "unusable"]
    correct = sum(v.ots_correct for v in usable)
    print(f"  wrote {n_written} rows to {out_path.name}; accuracy today {correct}/{len(usable)} on usable rows, "
          f"{sum(v.error_type == 'false_block' for v in usable)} false blocks, {sum(v.error_type == 'missed_threat' for v in usable)} missed threats")
    report(args)


# ----------------------------------------------------------------------------- report
def load_all():
    rows = []
    for p in sorted(LOG_DIR.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def pct(a, b):
    return f"{a / b:.1%}" if b else "n/a"


def report(args):
    rows = [r for r in load_all() if r.get("gold") and r["gold"] != "unusable"]
    if not rows:
        print("no graded rows yet"); return
    days = sorted({r["graded_at"][:10] for r in rows})
    lines = ["# OTS probe report", "",
             f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Messages invented daily by {GENERATOR_MODEL}, "
             f"classified by the live API, graded against `docs/LABELING_GUIDE.md` by {GRADER_MODEL}. "
             f"{len(rows)} usable messages over {len(days)} day(s); model version(s) seen: "
             f"{', '.join(sorted({str(r.get('model_version')) for r in rows}))}.", ""]

    def stats(rs):
        ham = [r for r in rs if r["gold"] == "ham"]; thr = [r for r in rs if r["gold"] != "ham"]
        return {"n": len(rs), "acc": pct(sum(r["ots_correct"] for r in rs), len(rs)),
                "block": pct(sum((r["ots_label"] != "ham") == (r["gold"] != "ham") for r in rs), len(rs)),
                "fb": pct(sum(r["ots_label"] != "ham" for r in ham), len(ham)),
                "mt": pct(sum(r["ots_label"] == "ham" for r in thr), len(thr)),
                "lat": (sorted(r["latency_ms"] for r in rs if r.get("latency_ms"))[len(rs) // 2] if rs else 0)}
    s = stats(rows)
    lines += ["## Overall", "", "| Messages | Accuracy | Block accuracy | False block rate | Missed threat rate | Median latency |", "|---|---|---|---|---|---|",
              f"| {s['n']} | {s['acc']} | {s['block']} | {s['fb']} | {s['mt']} | {s['lat']} ms |", ""]
    lines += ["## By day", "", "| Day | Messages | Accuracy | Block accuracy | False blocks | Missed threats |", "|---|---|---|---|---|---|"]
    for d in days:
        rs = [r for r in rows if r["graded_at"][:10] == d]; t = stats(rs)
        lines.append(f"| {d} | {t['n']} | {t['acc']} | {t['block']} | {t['fb']} | {t['mt']} |")
    lines.append("")
    misses = [r for r in rows if not r["ots_correct"]]
    if misses:
        lines += ["## Where it fails", ""]
        for title, keyf in (("By error type", lambda r: r["error_type"]), ("By category", lambda r: r["category"]), ("By language", lambda r: r["language"])):
            tot = Counter(keyf(r) for r in rows); bad = Counter(keyf(r) for r in misses)
            lines += [f"### {title}", "", "| Key | Misses | Of | Miss rate |", "|---|---|---|---|"]
            for k, n in sorted(bad.items(), key=lambda kv: -kv[1])[:12]:
                lines.append(f"| {k} | {n} | {tot[k]} | {pct(n, tot[k])} |")
            lines.append("")
        lines += ["## What the next training round needs", "",
                  "Grader notes, grouped by error type, most recent first. Severity 3 means high-confidence mistakes.", ""]
        for et in ("false_block", "missed_threat", "category_swap"):
            sel = sorted([r for r in misses if r["error_type"] == et], key=lambda r: (-r["severity"], r["graded_at"]), reverse=False)[:15]
            if not sel:
                continue
            lines += [f"### {et.replace('_', ' ')} ({sum(r['error_type'] == et for r in misses)})", ""]
            for r in sel:
                lines.append(f"- sev{r['severity']} gold={r['gold']} ots={r['ots_label']}@{r['ots_prob']} [{r['language']}] \"{r['text'][:120]}\" → {r['note']}")
            lines.append("")
    else:
        lines += ["No misses graded yet.", ""]
    # training additions: every graded miss with its gold label, plus a sample of correct hard rows is not needed
    add_path = LOG_DIR / "training_additions.csv"
    with open(add_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "category", "language", "ots_label", "ots_prob", "note", "day"])
        w.writeheader()
        for r in misses:
            w.writerow({"text": r["text"], "label": r["gold"], "category": r["category"], "language": r["language"],
                        "ots_label": r["ots_label"], "ots_prob": r["ots_prob"], "note": r["note"], "day": r["graded_at"][:10]})
    lines += ["## Files", "", f"- `logs/<day>.jsonl`: every message with the OTS answer and the grader verdict.",
              f"- `logs/improvements.log`: one line per miss, appended daily.",
              f"- `logs/training_additions.csv`: {len(misses)} graded misses with their gold label, ready to add as a source in `evals/distill_labels.py`.", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {REPORT} ({len(rows)} rows, accuracy {s['acc']}, false blocks {s['fb']}, missed threats {s['mt']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--n", type=int, default=150); p.add_argument("--api", default=DEFAULT_API)
    p.add_argument("--dry-run", action="store_true", help="no LLM calls; uses bench100 to check the plumbing"); p.set_defaults(fn=run)
    p = sub.add_parser("report"); p.set_defaults(fn=report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
