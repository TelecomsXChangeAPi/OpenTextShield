#!/usr/bin/env python3
"""Summarise a soak run: outcomes per class, latency over time, resource trend, incidents.

    python evals/soak/soak_report.py ~/ots-soak/run-2026-09-18
"""

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def load_jsonl(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def pct(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * len(s)))]


def hour_key(ts):
    return ts[:13]


def main():
    run = Path(sys.argv[1]).expanduser()
    results = load_jsonl(run / "results.jsonl")
    monitor = load_jsonl(run / "monitor.jsonl")
    if not results:
        print("no results yet")
        return
    first, last = results[0]["ts"], results[-1]["ts"]
    dur_h = (datetime.fromisoformat(last.replace("Z", "+00:00")) - datetime.fromisoformat(first.replace("Z", "+00:00"))).total_seconds() / 3600
    print(f"# Soak report: {run.name}\n\n{len(results)} messages over {dur_h:.1f} h ({first} to {last})\n")

    blocked = lambda r: r["outcome"] == "rejected"
    print("## Outcomes by class\n")
    print(f"| class | sent | forwarded | rejected | error | timeout | block rate | false block |")
    print(f"|---|---|---|---|---|---|---|---|")
    for label in ("ham", "spam", "phishing"):
        rs = [r for r in results if r["label"] == label]
        if not rs:
            continue
        c = Counter(r["outcome"] for r in rs)
        rate = sum(blocked(r) for r in rs) / len(rs)
        print(f"| {label} | {len(rs)} | {c['forwarded']} | {c['rejected']} | {c['error']} | {c['timeout']} | "
              f"{rate:.1%} | {rate:.2%} |" if label == "ham" else
              f"| {label} | {len(rs)} | {c['forwarded']} | {c['rejected']} | {c['error']} | {c['timeout']} | {rate:.1%} | |")

    print("\n## Ham false blocks by pool (what an operator would see)\n")
    for pool in ("ham_personal", "ham_a2p"):
        rs = [r for r in results if r["pool"] == pool]
        if rs:
            fb = [r for r in rs if blocked(r)]
            print(f"- {pool}: {len(fb)}/{len(rs)} = {len(fb)/len(rs):.2%}")
    fb_a2p = Counter(r["category"] for r in results if r["pool"] == "ham_a2p" and blocked(r))
    if fb_a2p:
        print("  blocked A2P by category:", dict(fb_a2p.most_common(8)))

    print("\n## Phishing by obfuscation\n")
    for obf in ("", "fullwidth", "lookalike", "zero_width", "spaced"):
        rs = [r for r in results if r["label"] == "phishing" and (r.get("obf") or "") == obf]
        if rs:
            print(f"- {obf or 'plain':10s} blocked {sum(blocked(r) for r in rs)}/{len(rs)} = {sum(blocked(r) for r in rs)/len(rs):.1%}")

    print("\n## Long messages via message_payload\n")
    rs = [r for r in results if r["via"] == "payload"]
    if rs:
        c = Counter(r["outcome"] for r in rs)
        bad = [r for r in rs if r["label"] != "ham"]
        print(f"- {len(rs)} sent, outcomes {dict(c)}; attacks blocked {sum(blocked(r) for r in bad)}/{len(bad)}")

    print("\n## Latency per hour (submit_sm round trip, ms)\n")
    print("| hour (UTC) | n | p50 | p95 | p99 | errors+timeouts |")
    print("|---|---|---|---|---|---|")
    by_hour = defaultdict(list)
    for r in results:
        by_hour[hour_key(r["ts"])].append(r)
    for h in sorted(by_hour):
        rs = by_hour[h]
        ms = [r["ms"] for r in rs if r["ms"] is not None]
        bad = sum(r["outcome"] in ("error", "timeout") for r in rs)
        print(f"| {h} | {len(rs)} | {pct(ms, .5):.0f} | {pct(ms, .95):.0f} | {pct(ms, .99):.0f} | {bad} |")

    errs = [r for r in results if r["outcome"] in ("error", "timeout")]
    print(f"\n## Errors and timeouts: {len(errs)}\n")
    if errs:
        print("statuses:", dict(Counter(str(r.get('status')) for r in errs)))
        for r in errs[:5]:
            print(f"  {r['ts']} {r['outcome']} status={r.get('status')} {r.get('err') or ''}")

    if monitor:
        print("\n## Resources (first hour vs last hour)\n")
        def mem_mb(s):
            m = re.match(r"([\d.]+)([KMG]i?B)", s or "")
            if not m:
                return None
            v, u = float(m.group(1)), m.group(2)[0]
            return v * {"K": 1 / 1024, "M": 1, "G": 1024}[u]
        head, tail = monitor[:60], monitor[-60:]
        def avg(xs):
            xs = [x for x in xs if x is not None]
            return sum(xs) / len(xs) if xs else float("nan")
        print(f"- container memory: {avg(mem_mb(m['container_mem']) for m in head):.0f} MB -> {avg(mem_mb(m['container_mem']) for m in tail):.0f} MB")
        print(f"- proxy RSS: {avg(m['proxy_rss_kb'] for m in head)/1024:.0f} MB -> {avg(m['proxy_rss_kb'] for m in tail)/1024:.0f} MB")
        print(f"- health check latency: p50 {pct([m['health_ms'] for m in monitor], .5):.0f} ms, max {max(m['health_ms'] for m in monitor):.0f} ms")
        unhealthy = [m for m in monitor if m["health"] != "healthy"]
        print(f"- unhealthy samples: {len(unhealthy)} of {len(monitor)}" + (f", first at {unhealthy[0]['ts']}" if unhealthy else ""))
        restarts = max((m["container_restarts"] or 0) for m in monitor)
        print(f"- container restarts: {int(restarts)}")
        print(f"- audit dir inside container: {monitor[-1].get('audit_dir_kb') or 0:.0f} KB; proxy log: {monitor[-1].get('proxy_log_kb') or 0:.0f} KB")
        qd = [m["queue_depth"] for m in monitor if m["queue_depth"] is not None]
        if qd:
            print(f"- API queue depth: max {max(qd):.0f}")

    chaos = run / "chaos.log"
    if chaos.exists():
        print("\n## Upstream restarts (chaos)\n")
        for line in chaos.read_text().splitlines()[-10:]:
            print("  " + line)

    plog = run / "ots_smpp.log"
    if plog.exists():
        text = plog.read_text(encoding="utf-8", errors="replace")
        print("\n## Proxy log counters\n")
        for name, pat in [("classification errors (fail-open)", r"Classification failed"),
                          ("API timeouts", r"Classification API timeout"),
                          ("below threshold (acted on)", r"Below threshold"),
                          ("skipped unclassifiable", r"Classification skipped"),
                          ("upstream reconnects", r"reconnect"),
                          ("no upstream available", r"No upstream")]:
            print(f"- {name}: {len(re.findall(pat, text))}")

    print("\n## Sample of blocked legitimate messages (review these)\n")
    for r in [r for r in results if r["label"] == "ham" and blocked(r)][:15]:
        print(f"  [{r['pool']:12s} {r['source']:10s}] id={r['id']}")


if __name__ == "__main__":
    main()
