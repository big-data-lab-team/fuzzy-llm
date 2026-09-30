#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Rebuild tab:prism-vector-perf from the 30 AVX-512 timing runs.

    uv run scripts/prism_bench_table.py
    uv run scripts/prism_bench_table.py --latex   # table body rows only

Reads results/prism_vs_mcaquad_512/run_<k>.log (or --in <dir>), one per
exclusive Xeon Gold 6130 node, as written by experiments/LLM/bench/prism_vs_mca.sh
(build.log sits beside them). Each run times native, PRISM SR and MCA quad RR
at full precision (t=24/53) and at reduced precision (t=8/24), in median
nanoseconds per element over seven trials.

The table uses the full-precision section. PRISM and MCA entries are the
median over runs of the per-run ratio to native time for the same operation and
type; MCA/PRISM is the median over runs of the per-run MCA to PRISM ratio.

The second block reports how much each backend's time moves between the two
precisions, as the median over runs of the per-run ratio. It is what the
sentence after the table summarizes; binary32 and binary64 are shown apart
because they do not behave alike under MCA.
"""

import argparse
import re
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_IN = REPO / "results" / "prism_vs_mcaquad_512"

OPS = ("add", "mul", "div", "fma")
TYPES = ("binary32", "binary64")
FULL, REDUCED = "24/53", "8/24"

SECTION = re.compile(r"=== run \d+ (\w+) p=(\S+) ===")
TIMING = re.compile(r"(add|mul|div|fma)\s+(binary\d+)\s+([\d.]+)")


def load(directory):
    runs = []
    for path in sorted(directory.glob("run_*.log")):
        text = path.read_text()
        if "RUN_OK" not in text:
            raise SystemExit(f"{path.name}: run did not finish (no RUN_OK)")
        times, section = {}, None
        for line in text.splitlines():
            if m := SECTION.match(line):
                section = (m[1], m[2])
            elif (m := TIMING.match(line)) and section:
                times[(*section, m[1], m[2])] = float(m[3])
        runs.append(times)
    if not runs:
        raise SystemExit(f"no run_*.log in {directory}")
    return runs


def median_ratio(runs, num, den):
    return st.median(r[num] / r[den] for r in runs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_dir", type=Path, default=DEFAULT_IN)
    ap.add_argument("--latex", action="store_true",
                    help="print only the LaTeX body rows of the table")
    args = ap.parse_args()
    runs = load(args.in_dir)

    rows = []
    for op in OPS:
        cells = []
        for ty in TYPES:
            key = lambda b: (b, FULL, op, ty)
            cells += [median_ratio(runs, key("prism"), key("native")),
                      median_ratio(runs, key("mca"), key("native")),
                      median_ratio(runs, key("mca"), key("prism"))]
        rows.append((op, cells))

    if args.latex:
        for op, cells in rows:
            print(f"{op} & " + " & ".join(f"${c:.1f}$" for c in cells) + r" \\")
        return

    print(f"{len(runs)} runs from {args.in_dir}")
    print(f"\nSlowdown vs native at t={FULL} (median over runs)")
    print(f"{'':6s}{'binary32':>27s}{'binary64':>27s}")
    print(f"{'op':6s}" + f"{'PRISM':>9s}{'MCA':>9s}{'MCA/PR':>9s}" * 2)
    for op, cells in rows:
        print(f"{op:6s}" + "".join(f"{c:9.1f}" for c in cells))

    print(f"\nTime at t={REDUCED} relative to t={FULL} (median of per-run ratios)")
    for backend in ("native", "prism", "mca"):
        for ty in TYPES:
            devs = {op: median_ratio(runs, (backend, REDUCED, op, ty),
                                     (backend, FULL, op, ty)) - 1 for op in OPS}
            worst = max(devs, key=lambda op: abs(devs[op]))
            print(f"  {backend:6s} {ty}: max |change| {100 * abs(devs[worst]):5.2f}% ({worst})  "
                  + " ".join(f"{op}={100 * d:+.2f}%" for op, d in devs.items()))


if __name__ == "__main__":
    main()
