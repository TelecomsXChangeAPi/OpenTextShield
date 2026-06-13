#!/usr/bin/env python3
"""Render a before/after markdown table from two run_eval summary JSON files.

Row labels are derived from the summary filenames by default — e.g.
``summary_v2.7.json`` yields the label ``v2.7`` — so the table reflects whatever
two runs you actually compare. Override with --before-label / --after-label when
the filename isn't descriptive.
"""
import argparse
import json
import re
from pathlib import Path


def label_from_path(path):
    """Extract a run tag from a summary filename: summary_<tag>.json -> <tag>.

    Falls back to the bare stem if the conventional ``summary_`` prefix is
    absent, so an arbitrary path still produces a sensible label.
    """
    stem = Path(path).stem  # e.g. "summary_v2.7"
    m = re.match(r"summary[_-]?(.+)", stem)
    return m.group(1) if m and m.group(1) else stem


def row(name, m):
    pc = m["per_class"]["phishing"]
    return (f"| {name} | {m['n']} | {m['accuracy_3class']:.1%} | "
            f"{m['accuracy_binary_block']:.1%} | {pc['recall']:.1%} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--before-label", default=None,
                    help="row label for the --before run (default: derived from filename)")
    ap.add_argument("--after-label", default=None,
                    help="row label for the --after run (default: derived from filename)")
    args = ap.parse_args()
    before = json.load(open(args.before))
    after = json.load(open(args.after))

    before_label = args.before_label or label_from_path(args.before)
    after_label = args.after_label or label_from_path(args.after)

    lines = ["| Benchmark | n | 3-class acc | Block acc | Phishing recall |",
             "|---|---|---|---|---|"]
    for ds in sorted(set(before) | set(after)):
        if ds in before:
            lines.append(row(f"{ds} — {before_label}", before[ds]))
        if ds in after:
            lines.append(row(f"{ds} — {after_label}", after[ds]))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
