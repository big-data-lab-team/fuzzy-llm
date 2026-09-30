"""Parse perplexity sweep logs into tidy records.

This is the paper's own copy of the parser, so that the figures can be rebuilt
from ``results/`` without a checkout of the sweep harness. It is kept
deliberately close to the upstream original,
``pablo-fuzzy-pytorch/experiments/LLM/parse_sweep_logs.py``, so that divergence
between the two is easy to see in a diff.

The sweeps write one log per configuration. Mode and seed live in the path
(``fine_sr_block_all/layer_mlp_c_proj_prec_6_seed_3.log``) while the measured
values live in the final ``Result ->`` line, so both have to be read.

A log without a ``Result ->`` line is an incomplete run and is skipped rather
than counted, which matters because a sweep still in flight leaves such files
behind: at the time of writing, the per-component level is only partly returned.
"""

import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev

RESULT = re.compile(r"^Result ->")
PRECISION = re.compile(r"Precision:\s*(\d+)")
PERPLEXITY = re.compile(r"Perplexity:\s*([\d.eE+-]+)")
TIME = re.compile(r"Time:\s*([\d.]+)s")
GROUP = re.compile(r"Group:\s*(\S+)")
LAYER = re.compile(r"Layer:\s*(\S+)")
BLOCK = re.compile(r"Block:\s*(\S+)")
SEED = re.compile(r"_seed_(\d+)\.log$")
CONFIG = re.compile(r"Configuration:\s*(\S+)")
#: ``Window: i/k`` -- which of k evenly spaced line blocks was scored. Absent
#: from every log written before the robustness study, which is why the record
#: default below is the string "0/1" rather than an integer: "0/1" (the whole
#: leading 1% slice) and "0/8" (the first of eight blocks) are different
#: samples and must never collapse to the same key, even though they happen to
#: tokenize identically.
WINDOW = re.compile(r"Window:\s*(\d+)\s*/\s*(\d+)")

#: Levels a record can be classified into, in the order the paper presents them.
LEVELS = ("global", "component", "sublayer", "blockwise", "cumulative", "mixed")

#: Protocol families. The 2026-08-17 reruns repeat several sweeps under two
#: protocols, and the logs are told apart only by their top-level directory:
#: ``legacy_*`` reproduces the original run-wide-mode configuration so the
#: cumulative t=5 arm can join the existing table without a protocol
#: discontinuity, while ``scoped_*`` is the corrected mode-scoped,
#: RN-24-background configuration. Their per-configuration file names are
#: identical, so without this field ``summarize`` pools two different
#: experiments into one mean. ``original`` is the pre-rerun checked-in data.
#:
#: ``corrected`` is narrower: it exists only for the global level, and marks
#: the ``global_fixed_*`` rerun that computes the metric outside the
#: instrumented scope. The pre-fix ``global_sr``/``global_rn`` runs it replaces
#: are no longer ingested; see :data:`SUPERSEDED`.
#: ``window`` is the multi-window robustness study: the same sites as the main
#: sweep, scored on eight disjoint slices of the test split instead of one. It
#: is a separate arm rather than a separate level because it measures the same
#: configurations; without the split every robustness row would be averaged
#: into the main-sweep mean it exists to be compared against.
ARMS = ("original", "legacy", "scoped", "corrected", "window")

#: Robustness log directories, mapped to the level each one re-measures. The
#: mapping is explicit because the fallbacks in :func:`_level` would classify
#: these by coincidence -- ``robustness_global_sr`` lands on "global" only
#: because it contains the substring ``global_`` -- and would silently move a
#: whole sweep into another level the first time a directory is renamed.
ROBUSTNESS_LEVELS = {
    "robustness_anchor": "global",
    "robustness_global": "global",
    "robustness_mlp": "component",
    "robustness_lm_head": "component",
    "robustness_cumulative": "cumulative",
}

#: Log directories that are not measurements and must never reach the records.
#:
#: ``global_sr``/``global_rn`` set precision once for the whole process, which
#: put the cross-entropy and the exp() inside the instrumented arithmetic (B1):
#: the reported perplexity was itself rounded to the swept precision. The
#: values are not merely bad but impossible -- two SR seeds at t=2 report
#: exactly 0.00, below the lower bound of 1 that perplexity has by
#: construction, RN at t=2 reports 48.00 against a full-precision reference of
#: 62.51, and every value at t<=8 lands exactly on the t-bit significand grid
#: (8192.0, 592.0, 87.5). ``global_fixed_*`` is the rerun that hooks
#: ``model.transformer`` and ``model.lm_head`` so t returns to 24 before the
#: metric runs, and is the only global-level measurement.
#:
#: These were kept in the records for a while, distinguished from the rerun by
#: the ``arm`` column alone. That is too quiet a signal: a reader who filters
#: on level rather than on arm gets a mixture of a measurement and an artifact,
#: and has been misled by it. Dropping them at ingest makes the record file
#: self-describing. The logs themselves stay under ``results/`` for anyone who
#: wants to inspect the failure.
SUPERSEDED = ("global_sr", "global_rn")



def _arm(path: Path) -> str:
    """Which protocol family a log belongs to, from its top-level directory.

    Everything that is not explicitly ``legacy_*`` in the rerun tree is part of
    the corrected protocol, including the sweeps that have no legacy
    counterpart at all (``global_*``, ``mixed_scoped``, ``native_reference``).
    ``global_fixed_*`` is the metric-external global rerun and must not be
    pooled with the still-broken ``global_*`` rows just because both would
    otherwise read as "scoped".
    """
    head = path.parts[0] if path.parts else ""
    if head.startswith("robustness_"):
        return "window"
    if head.startswith("legacy_"):
        return "legacy"
    if head.startswith("global_fixed_"):
        return "corrected"
    return "scoped"


def _mode_from_path(path: Path) -> str:
    """SR and RN runs are separated by directory, not by anything in the line.

    The IEEE reference is checked first: it is the one arm run against
    ``libinterflop_ieee.so`` rather than PRISM, so it is neither of PRISM's two
    rules and must not fall through to the ``_rn`` test.
    """
    parts = "/".join(path.parts)
    if "ieee" in parts or "native_reference" in parts:
        return "ieee"
    if "_sr" in parts or "global_sr" in parts:
        return "sr"
    if "_rn" in parts or "global_rn" in parts:
        return "rn"
    return "unknown"


def _level(result: str, block: str | None, relpath: str) -> str:
    """Classify by directory first.

    The block string alone cannot separate the blockwise from the cumulative
    sweep: the cumulative sweep's first point is a single block ("0"), spelled
    identically to a blockwise point. Only the directory distinguishes them.
    """
    if CONFIG.search(result):
        return "mixed"
    head = relpath.split("/")[0]
    for prefix, level in ROBUSTNESS_LEVELS.items():
        if head.startswith(prefix):
            return level
    if "Global" in result or "global_" in relpath:
        return "global"
    if GROUP.search(result):
        return "component"
    if "cumulative" in relpath:
        return "cumulative"
    if "blockwise" in relpath:
        return "blockwise"
    if block == "all":
        return "sublayer"
    return "blockwise"


def load(root) -> list[dict]:
    """Return one record per completed configuration under ``root``."""
    root = Path(root)
    records = []
    for path in sorted(root.rglob("*.log")):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in SUPERSEDED:
            continue  # not a measurement; see SUPERSEDED

        text = path.read_text(errors="replace")
        line = next((l for l in text.splitlines() if RESULT.match(l)), None)
        if line is None:
            continue  # incomplete run, or a control log in another format

        relpath = "/".join(relative.parts)
        config_m = CONFIG.search(line)
        precision, perplexity = PRECISION.search(line), PERPLEXITY.search(line)

        # The mixed-recipe runs report a whole-network recipe rather than one
        # swept precision, so they carry no "Precision:" field. Requiring one
        # silently discarded every such record.
        if not perplexity or not (precision or config_m):
            continue  # a Result line we do not know how to read

        block_m = BLOCK.search(line)
        window_m = WINDOW.search(line)
        block = block_m.group(1) if block_m else None
        target_m = GROUP.search(line) or LAYER.search(line)
        seed_m = SEED.search(path.name)
        time_m = TIME.search(line)

        # A recipe applies several rules at once, so no single sr/rn label is
        # honest for it; the recipe name in ``target`` carries the detail.
        mode = "mixed" if config_m else _mode_from_path(relative)
        if config_m:
            target = config_m.group(1)
        else:
            target = target_m.group(1) if target_m else "global"

        records.append(
            dict(
                level=_level(line, block, relpath),
                arm=_arm(relative),
                mode=mode,
                target=target,
                block=block or "all",
                window=f"{window_m.group(1)}/{window_m.group(2)}" if window_m else "0/1",
                precision=int(precision.group(1)) if precision else None,
                perplexity=float(perplexity.group(1)),
                seconds=float(time_m.group(1)) if time_m else None,
                seed=int(seed_m.group(1)) if seed_m else 1,
                path=str(path),
            )
        )
    return records


def summarize(records, level, keys=("target", "precision"), arm=None):
    """Collapse seeds into mean plus one sample SD, keyed by ``keys`` and mode.

    Returns ``{(key_tuple, mode): (mean, mean-sd, mean+sd, n)}``. SR arms have
    several seeds; RN arms have one because PRISM's RN is bit-reproducible, so
    its SD is reported as zero rather than inferred from nonexistent repeats.

    ``arm`` selects one protocol family. It is not optional in practice for the
    blockwise and cumulative levels, where the legacy and corrected protocols
    measure the same configurations and would otherwise be averaged together.

    Passing no ``arm`` means "the main sweep", and therefore excludes the
    ``window`` arm. Those runs re-measure configurations this function is
    otherwise summarizing, on a different slice of the test split, so pooling
    them silently mixes two datasets into one mean. Relying on callers to pass
    an arm is not enough: two of the calls in make_figures.py deliberately pass
    none, and today they are safe only because no robustness cell happens to
    land on the sublayer level. Ask for these rows explicitly with
    ``arm="window"``.
    """
    buckets = defaultdict(list)
    for r in records:
        if r["level"] != level:
            continue
        if arm is None and r.get("arm", "original") == "window":
            continue
        if arm is not None and r.get("arm", "original") != arm:
            continue
        buckets[(tuple(r[k] for k in keys), r["mode"])].append(r["perplexity"])
    out = {}
    for key, values in buckets.items():
        center = fmean(values)
        spread = stdev(values) if len(values) > 1 else 0.0
        out[key] = (center, center - spread, center + spread, len(values))
    return out


#: Columns of the distilled record file, in order.
FIELDS = ("level", "arm", "mode", "target", "block", "window", "precision",
          "perplexity", "seconds", "seed")


def write_csv(records, path) -> None:
    """Write the parsed records to a small CSV.

    Each log is ten kilobytes of which one line matters, and most of the rest is
    the container retrying a network fetch it cannot make. Distilling the sweep
    to one row per configuration is what makes the inputs to the figures small
    enough to keep beside the paper.
    """
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(records, key=lambda r: (r["level"], r.get("arm", ""),
                                                r["target"], r["block"],
                                                r.get("window", "0/1"),
                                                r["precision"] is None,
                                                r["precision"] or 0,
                                                r["mode"], r["seed"])):
            writer.writerow(r)


def read_csv(path) -> list[dict]:
    """Read records written by :func:`write_csv`."""
    import csv

    with Path(path).open(newline="") as fh:
        return [
            dict(r,
                 # A CSV written before the rerun has no arm column; those rows
                 # are the pre-rerun data by definition.
                 arm=r.get("arm") or "original",
                 # No window column means the run predates the robustness study,
                 # so it scored the whole leading slice: window 0 of 1.
                 window=r.get("window") or "0/1",
                 precision=int(r["precision"]) if r["precision"] else None,
                 perplexity=float(r["perplexity"]),
                 seconds=float(r["seconds"]) if r["seconds"] else None,
                 seed=int(r["seed"]))
            for r in csv.DictReader(fh)
        ]


if __name__ == "__main__":
    import sys

    recs = load(sys.argv[1] if len(sys.argv) > 1 else "perplexity_logs")
    by_level = defaultdict(int)
    for r in recs:
        by_level[(r["level"], r["arm"], r["mode"])] += 1
    print(f"{len(recs)} completed configurations")
    for k in sorted(by_level):
        print(f"  {k[0]:<11} {k[1]:<9} {k[2]:<8} {by_level[k]:>4}")
