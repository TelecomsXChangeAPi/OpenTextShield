#!/usr/bin/env python3
"""Render a before/after markdown table from two run_eval summary JSON files."""
import argparse
import json


def row(name, m):
    pc = m["per_class"]["phishing"]
    return (f"| {name} | {m['n']} | {m['accuracy_3class']:.1%} | "
            f"{m['accuracy_binary_block']:.1%} | {pc['recall']:.1%} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    args = ap.parse_args()
    before = json.load(open(args.before))
    after = json.load(open(args.after))

    lines = ["| Benchmark | n | 3-class acc | Block acc | Phishing recall |",
             "|---|---|---|---|---|"]
    for ds in sorted(set(before) | set(after)):
        if ds in before:
            lines.append(row(f"{ds} — v2.5", before[ds]))
        if ds in after:
            lines.append(row(f"{ds} — v2.6", after[ds]))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
