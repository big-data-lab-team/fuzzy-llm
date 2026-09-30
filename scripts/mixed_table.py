#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Rebuild tab:accuracy-matched-precision: mixed rounding against all-RN.

    uv run scripts/mixed_table.py                     # the recorded runs
    uv run scripts/mixed_table.py --in <mixed_logs>   # a rerun's logs

Reads one perplexity per run, either from the recorded outcomes
(results/mixed_followup_20260915/outcomes/<run>.json) or from the logs that
experiments/LLM/configs/gen_mixed_configs.sh names (<run>.log, last
"Result -> ... Perplexity:" line). For each MLP down-projection precision k it
reports the mixed mean and sample SD over the five SR seeds, the all-RN control,
the gap Delta = mixed - RN with its relative size and two-sided 95% Student-t
interval (df = 4), and both configurations' degradation from the t=24 reference.
"""

import argparse
import json
import re
import statistics as st
from math import sqrt
from pathlib import Path

from composition_stats import t_crit

REPO = Path(__file__).resolve().parent.parent
DEFAULT_IN = REPO / "results" / "mixed_followup_20260915" / "outcomes"
SEEDS = (101, 102, 103, 104, 105)
RESULT = re.compile(r"^Result -> .*Perplexity:\s*([\d.]+)", re.M)


def perplexity(directory, run):
    record = directory / f"{run}.json"
    if record.is_file():
        return json.loads(record.read_text())["ppl"]
    hits = RESULT.findall((directory / f"{run}.log").read_text())
    if not hits:
        raise SystemExit(f"{run}.log has no Result line")
    return float(hits[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_dir", type=Path, default=DEFAULT_IN)
    args = ap.parse_args()

    ref = perplexity(args.in_dir, "reference")
    print(f"reference (t=24, RN): {ref:.4f}\n")
    print(f"{'k':>2}  {'mixed':>13}  {'RN':>7}  {'gap (%)':>17}  {'95% CI':>16}  {'vs ref: mixed (RN)':>20}")
    for k in (8, 7, 6):
        rn = perplexity(args.in_dir, f"mlp{k}_rn")
        mixed = [perplexity(args.in_dir, f"mlp{k}_mixed_seed{s}") for s in SEEDS]
        mean, sd = st.mean(mixed), st.stdev(mixed)
        gap = mean - rn
        half = t_crit(len(mixed) - 1) * sd / sqrt(len(mixed))
        print(f"{k:>2}  {mean:6.2f}±{sd:5.2f}  {rn:7.2f}  {gap:+7.2f} ({100 * gap / rn:+6.2f}%)"
              f"  [{gap - half:+6.2f},{gap + half:+6.2f}]"
              f"  {100 * (mean / ref - 1):+6.2f}% ({100 * (rn / ref - 1):+6.2f}%)")


if __name__ == "__main__":
    main()
