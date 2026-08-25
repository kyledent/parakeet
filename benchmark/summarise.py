#!/usr/bin/env python3
"""
Summarise benchmark runs into a table.

Every number quoted about these benchmarks should come from here rather than
being typed into a document by hand, so that it can be re-derived when a
config, a GPU or the code changes.

Usage:
    benchmark/summarise.py <results-dir> [--markdown]

<results-dir> is the --outdir given to run_benchmark.sh; each run leaves a
result.json in a subdirectory named after its config.
"""

import argparse
import json
import pathlib
import sys


def load(root):
    runs = []
    for path in sorted(pathlib.Path(root).glob("*/result.json")):
        try:
            runs.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError) as e:
            print(f"skipping {path}: {e}", file=sys.stderr)
    return runs


STAGES = ["sample", "add_molecules", "add_fiducials", "exit_wave", "optics", "image"]


def human_bytes(n):
    if not n:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def render(runs, markdown=False):
    if not runs:
        print("no result.json found -- run benchmark/run_benchmark.sh first")
        return 1

    cols = ["config", "field", "images", "particles"] + STAGES + ["total", "image size"]
    rows = []
    for r in runs:
        s = r["specimen"]
        t = r["seconds"]
        rows.append(
            [
                r["config"].replace(".yaml", ""),
                f"{s['detector_px'][0]}x{s['detector_px'][1]}",
                str(s["num_images"]),
                f"{s['ribosomes']}+{s['fiducials']}f",
                *[f"{t[k]:.1f}" if k in t else "-" for k in STAGES],
                f"{r['seconds_total']:.1f}",
                human_bytes(r["output_bytes"].get("image")),
            ]
        )

    if markdown:
        print("| " + " | ".join(cols) + " |")
        print("|" + "|".join("---" for _ in cols) + "|")
        for row in rows:
            print("| " + " | ".join(row) + " |")
    else:
        width = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
        print("  ".join(c.ljust(w) for c, w in zip(cols, width)))
        print("  ".join("-" * w for w in width))
        for row in rows:
            print("  ".join(v.ljust(w) for v, w in zip(row, width)))

    gpus = {r.get("gpu") for r in runs if r.get("gpu")}
    if gpus:
        print("\nGPU: " + "; ".join(sorted(gpus)))
    print("seconds are wall clock for each stage, not GPU time")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("results_dir")
    ap.add_argument(
        "--markdown",
        action="store_true",
        help="emit a markdown table, for pasting into a write-up",
    )
    args = ap.parse_args()
    return render(load(args.results_dir), args.markdown)


if __name__ == "__main__":
    sys.exit(main())
