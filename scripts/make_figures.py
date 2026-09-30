#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8,<4"]
# ///
"""Rebuild the paper's sweep figures from the logs in ``results/``.

    uv run scripts/make_figures.py                 # rebuild what the logs support
    uv run scripts/make_figures.py --summary       # also print the parsed numbers
    uv run scripts/make_figures.py --strict        # fail if any figure is missing data

Only the levels whose logs are present are redrawn. A level with no logs is
reported and its existing PDF is left alone, because an empty axes written over
a figure the paper cites is worse than a stale one. At the time of writing the
global and blockwise logs live on the cluster rather than in this repository, so
those two figures are skipped by default.

SR and RN are distinguished by colour *and* by linestyle and marker, so the
figures survive greyscale printing and colour-vision deficiency without relying
on hue. The two hues are the first two categorical slots of a palette validated
for CVD separation (worst adjacent pair dE 24.7 protan, 33.6 at normal vision,
against a light surface); do not substitute arbitrary colours.

Perplexity spans nine orders of magnitude across the sweeps, so every axis is
logarithmic. That is a real property of the data -- reducing the language-model
head to four bits is catastrophic -- and not a presentational choice.

Derived from ``pablo-fuzzy-pytorch/experiments/LLM/plot_paper_figures.py``.
"""

import argparse
import sys
from math import log, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parse_sweep_logs

REPO = Path(__file__).resolve().parent.parent
DEFAULT_LOGS = REPO / "results" / "perplexity_logs_epoch" / "256"
DEFAULT_RECORDS = REPO / "results" / "sweep_records.csv"
DEFAULT_OUT = REPO / "figures"

# Categorical slots 1 and 2 of the validated palette.
SR_COLOR = "#2a78d6"
RN_COLOR = "#eb6834"

# Secondary encoding, so identity is never carried by hue alone.
STYLE = {
    "sr": dict(color=SR_COLOR, marker="o", linestyle="-", label="SR"),
    "rn": dict(color=RN_COLOR, marker="s", linestyle="--", label="RN"),
}

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"

SUBLAYERS = ["attn_c_attn", "attn_c_proj", "mlp_c_fc", "mlp_c_proj"]
COMPONENTS = ["attention", "mlp", "lm_head", "attn_qkav"]
COMPONENT_TITLE = {
    "attention": "Full attention",
    "mlp": "Feedforward network (MLP)",
    "lm_head": "Language-model head",
    "attn_qkav": r"Attention products only ($QK^\top$, $AV$)",
}

#: Student-t, four degrees of freedom, two-sided 95%. The same exploratory
#: interval the tables use; see the limitations section on what it does not do.
T95_4 = 2.776
REDUCTION_LEN = {
    "attn_c_attn": 768,
    "attn_c_proj": 768,
    "mlp_c_fc": 768,
    "mlp_c_proj": 3072,
}


def style_axes(ax):
    """Recessive grid and axes; text in ink tokens, never a series colour."""
    ax.grid(True, which="major", color=GRID, linewidth=0.5, alpha=0.9)
    ax.grid(True, which="minor", color=GRID, linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=7, width=0.6)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
    })


def series(summary, key_fn, xs, mode):
    """Extract (x, mean, mean-SD, mean+SD), skipping missing points."""
    out = []
    for x in xs:
        cell = summary.get((key_fn(x), mode))
        if cell:
            out.append((x, cell[0], cell[1], cell[2]))
    return out


def draw(ax, pts, mode, band=True):
    if not pts:
        return
    x = [p[0] for p in pts]
    ax.plot(x, [p[1] for p in pts], **STYLE[mode],
            markerfacecolor="white", markeredgewidth=1.2, zorder=3)
    # One sample SD across seeds. RN is bit-reproducible so its band has zero width;
    # drawing it anyway would imply a measurement that was not made.
    if band and any(p[3] > p[2] for p in pts):
        ax.fill_between(x, [p[2] for p in pts], [p[3] for p in pts],
                        color=STYLE[mode]["color"], alpha=0.18, linewidth=0,
                        zorder=2)



def carry_forward(records, records_path):
    """Keep prior rows that the new sweep does not cover, drop the rest.

    Distilling straight from the rerun tree would silently discard anything the
    rerun did not repeat. The sublayer sweep is the live case: the reruns cover
    t=4..14 for all four targets but no t=24 at all, and the t=24 reference the
    sublayer figure draws against lived only in the earlier records.

    Superseded rows -- those measuring a configuration the rerun measured again
    -- are dropped, since the corrected arm is authoritative for them. Rows the
    rerun never touched are carried forward and marked as the earlier arm, so
    the distinction stays visible in the file rather than being implied.
    """
    if not records_path.is_file():
        return records
    cell = lambda r: (r["level"], r["target"], r["block"], r.get("window", "0/1"),
                      r["precision"], r["mode"])
    covered = {cell(r) for r in records}
    kept = [r for r in parse_sweep_logs.read_csv(records_path)
            if cell(r) not in covered]
    if kept:
        where = sorted({(r["level"], r["target"], r["precision"]) for r in kept})
        print(f"  carried forward {len(kept)} record(s) the rerun does not cover:")
        for level, target, precision in where:
            print(f"    {level} {target} t={precision}")
    return records + kept


def drop_degenerate(records, level):
    """Remove (precision, arm) cells whose reported perplexity cannot be a
    measurement.

    Perplexity is exp of a mean negative log-likelihood, so it is at least 1.
    A run that reports less exits cleanly and is recorded as an ordinary
    result, so nothing downstream would otherwise notice. The case that
    motivated this guard was the pre-B1-fix global sweep, two of whose five SR
    seeds at t=2 reported exactly 0.00; those logs are no longer ingested
    (``parse_sweep_logs.SUPERSEDED``), and the check stays as a guard against
    the next one.

    The test is a property no genuine measurement can have, not a threshold on
    how bad a result is allowed to look. Catastrophic perplexities at t=4 and
    t=6 are real and are kept. An earlier version of this function also
    discarded any precision whose deterministic RN control fell below the
    full-precision reference; that removed t=10 and t=12, where RN sits 0.2%
    under it for ordinary reasons, so the whole informative middle of the
    sweep vanished. Grouping by arm as well as precision matters once a level
    carries more than one, so that a contaminated arm cannot take a clean
    arm's measurement at the same precision down with it. A cell is dropped
    only when some run in it reports a value outside the range perplexity can
    take, and then both rounding modes in that arm go with it, so the
    remaining points stay comparable.
    """
    at = {}
    for r in records:
        if r["level"] == level:
            at.setdefault((r["precision"], r.get("arm", "original")), []).append(r)
    bad = {key for key, rs in at.items()
           if any(r["perplexity"] <= 1.0 for r in rs)}
    for precision, arm in sorted(bad):
        print(f"note: {level} t={precision} arm={arm} discarded; reported "
              f"perplexity is not physically attainable", file=sys.stderr)
    return [r for r in records
            if r["level"] != level
            or (r["precision"], r.get("arm", "original")) not in bad]


def pick_arm(records, level):
    """Choose one protocol family for a level, preferring the corrected runs.

    The reruns repeat the blockwise and cumulative sweeps under both the legacy
    and the corrected protocol, so drawing without choosing would average two
    different experiments. Hard-coding the name has the opposite failure: a
    figure rebuilt from an older record file would come back empty and look
    like missing data. Resolving against what the records actually contain
    avoids both, and says so when it had to choose.
    """
    arms = {r.get("arm", "original") for r in records if r["level"] == level}
    # "window" trails deliberately: the robustness runs re-measure configurations
    # the main sweep already covers, so they must be selectable but never
    # preferred over the arm a figure is meant to draw.
    for candidate in ("corrected", "scoped", "legacy", "original", "window"):
        if candidate in arms:
            if len(arms) > 1:
                print(f"note: {level} carries arms {sorted(arms)}; "
                      f"drawing {candidate}", file=sys.stderr)
            return candidate
    return None


# ---------------------------------------------------------------- figure 1
def fig_global(records, out):
    records = drop_degenerate(records, "global")
    s = parse_sweep_logs.summarize(records, "global", keys=("precision",),
                               arm=pick_arm(records, "global"))
    if not s:
        return None
    xs = sorted({k[0][0] for k in s if 4 <= k[0][0] <= 14})
    fig, ax = plt.subplots(figsize=(2.8, 1.85))
    for mode in ("sr", "rn"):
        draw(ax, series(s, lambda t: (t,), xs, mode), mode)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xlabel("Virtual precision $t$ (significand bits)")
    ax.set_ylabel("Perplexity")
    ax.set_title("Uniform precision everywhere", color=INK)
    ax.legend(frameon=False, loc="upper right")
    style_axes(ax)
    fig.savefig(out / "global_sweep.pdf")
    plt.close(fig)
    return "global_sweep.pdf"


# ---------------------------------------------------------------- figure 2
def fig_sublayer(records, out):
    s = parse_sweep_logs.summarize(records, "sublayer",
                               arm=pick_arm(records, "sublayer"))
    if not s:
        return None
    # The t=24 run is the in-level reference, not a swept point, and it is
    # looked up across every arm on purpose. The reruns stop at t=14, so the
    # anchor survives only in the earlier records; filtering it by the drawn
    # arm silently removes the reference line rather than reporting anything.
    anchors = parse_sweep_logs.summarize(records, "sublayer")
    ref = next((v[0] for k, v in anchors.items() if k[0][1] == 24), None)
    xs = sorted({k[0][1] for k in s if k[0][1] != 24})

    fig, axes = plt.subplots(2, 2, figsize=(5.8, 3.4), sharex=True, sharey=True)
    for ax, layer in zip(axes.flat, SUBLAYERS):
        if ref:
            ax.axhline(ref, color=MUTED, linewidth=0.7, linestyle=":", zorder=1)
        for mode in ("sr", "rn"):
            draw(ax, series(s, lambda t, L=layer: (L, t), xs, mode), mode)
        ax.set_title(f"{layer}  ($n={REDUCTION_LEN[layer]}$)", color=INK)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        style_axes(ax)

    for ax in axes[1]:
        ax.set_xlabel("Virtual precision $t$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Perplexity")
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.savefig(out / "sublayer_sweep.pdf")
    plt.close(fig)
    return "sublayer_sweep.pdf"



def verdict(summary, target, precision):
    """Which rule the exploratory interval separates at one cell, if either.

    Returns "sr", "rn" or None. This is the same test the tables apply: RN is a
    constant -- now a measured one, five bit-identical runs -- so the interval
    is one-sample around the SR mean.
    """
    sr = summary.get(((target, precision), "sr"))
    rn = summary.get(((target, precision), "rn"))
    if not sr or not rn:
        return None
    mean, lo, hi, n = sr
    half = T95_4 * ((hi - lo) / 2) / sqrt(n) if n > 1 else 0.0
    delta = mean - rn[0]
    if abs(delta) <= half:
        return None
    return "sr" if delta < 0 else "rn"


def log_records(records):
    r"""Records with ``perplexity`` replaced by its natural log.

    Feeding this to ``parse_sweep_logs.summarize`` builds the mean and sample
    SD of log perplexity instead of raw perplexity. This matters at low
    precision, where a couple of catastrophic SR seeds dominate the raw-scale
    sample SD (global $t=3$: SR perplexities range from $6.8\times10^{16}$ to
    $1.05\times10^{18}$ over five seeds, so the raw interval straddles zero and
    reports "unresolved" even though RN wins on every one of the five seeds by
    twelve orders of magnitude) and where excess loss, not raw perplexity, is
    the paper's own natural scale for this comparison
    (\cref{eq:empirical-excess-loss}).
    """
    return [dict(r, perplexity=log(r["perplexity"]))
            for r in records if r.get("perplexity") and r["perplexity"] > 0]


def verdict_1key(summary, precision, mode_key=None):
    r"""Like ``verdict``, but for a summary keyed by a single field (e.g. just
    precision, as the global level uses) rather than by ``(target, precision)``.

    Same interval test as ``verdict``: RN's single deterministic value against
    the two-sided $95\%$ Student-$t$ interval around the SR mean. Only the key
    shape differs, because ``summarize(..., keys=("precision",))`` builds
    ``((precision,), mode)`` keys instead of ``((target, precision), mode)``.
    """
    key = (mode_key, precision) if mode_key is not None else (precision,)
    sr = summary.get((key, "sr"))
    rn = summary.get((key, "rn"))
    if not sr or not rn:
        return None
    mean, lo, hi, n = sr
    half = T95_4 * ((hi - lo) / 2) / sqrt(n) if n > 1 else 0.0
    delta = mean - rn[0]
    if abs(delta) <= half:
        return None
    return "sr" if delta < 0 else "rn"


def report_global_log_resolution(records):
    """Which global-sweep precisions are resolved on log perplexity, per rule.

    Prints the raw-scale and log-scale verdict side by side at every swept
    precision so the two can be compared directly; see ``log_records`` for why
    they differ at low $t$. This is the check behind the abstract's and
    \\S5's "$t=4$ to $t=11$" resolved-advantage range: rerun this function
    rather than hand-checking the CSV if that range is ever revisited.
    """
    arm = pick_arm(records, "global")
    raw = parse_sweep_logs.summarize(records, "global", keys=("precision",), arm=arm)
    logged = parse_sweep_logs.summarize(log_records(records), "global",
                                        keys=("precision",), arm=arm)
    if not raw:
        return
    print("\n=== global sweep: raw-scale vs log-scale resolution")
    print(f"  {'t':>4}  {'raw':>5}  {'log':>5}")
    for t in sorted({k[0][0] for k in raw if 4 <= k[0][0] <= 14}):
        v_raw = verdict_1key(raw, t)
        v_log = verdict_1key(logged, t)
        print(f"  {t:>4}  {v_raw or '-':>5}  {v_log or '-':>5}")


# ---------------------------------------------------------------- figure 5
def fig_component(records, out):
    """Per-component sweep across the four main functional groups."""
    arm = pick_arm(records, "component")
    s = parse_sweep_logs.summarize(records, "component", arm=arm)
    if not s:
        return None
    anchors = parse_sweep_logs.summarize(records, "sublayer")
    ref = next((v[0] for k, v in anchors.items() if k[0][1] == 24), None)
    xs = sorted({k[0][1] for k in s})

    fig, axes = plt.subplots(2, 2, figsize=(5.8, 3.4), sharex=True)
    # Leave room for the right panels' scientific-notation y-axis labels.
    fig.subplots_adjust(wspace=0.45)
    for ax, comp in zip(axes.flat, COMPONENTS):
        if ref:
            ax.axhline(ref, color=MUTED, linewidth=0.7, linestyle=":", zorder=1)
        for mode in ("sr", "rn"):
            draw(ax, series(s, lambda t, C=comp: (C, t), xs, mode), mode)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_title(COMPONENT_TITLE.get(comp, comp), color=INK)
        style_axes(ax)

    for ax in axes[1]:
        ax.set_xlabel("Virtual precision $t$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Perplexity")
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.savefig(out / "component_sweep.pdf")
    plt.close(fig)
    return "component_sweep.pdf"


# ------------------------------------------------------------ figures 3 & 4
def fig_positional(records, out, level, filename, xlabel, title, order_key):
    s = parse_sweep_logs.summarize(records, level, keys=("block", "precision"),
                               arm=pick_arm(records, level))
    if not s:
        return None
    blocks = sorted({k[0][0] for k in s}, key=order_key)
    precisions = sorted({k[0][1] for k in s})

    # Each precision gets its own y-scale. At t=4 the perplexities run into the
    # thousands while at t=6 they sit between 62 and 80; a shared axis flattens
    # the t=6 panel and hides the gap opening with depth, which is the finding
    # that panel exists to show. These are small multiples, not one chart with
    # two scales.
    fig, axes = plt.subplots(1, len(precisions),
                            figsize=(5.8, 1.95), sharey=False)
    axes = [axes] if len(precisions) == 1 else list(axes)
    idx = {b: i for i, b in enumerate(blocks)}

    for ax, prec in zip(axes, precisions):
        for mode in ("sr", "rn"):
            pts = [(idx[b], *s[((b, prec), mode)])
                   for b in blocks if ((b, prec), mode) in s]
            # Bands show one sample SD over the SR seeds.
            draw(ax, [(p[0], p[1], p[2], p[3]) for p in pts], mode)
        ax.set_xticks(range(len(blocks)))
        ax.set_xticklabels(blocks, fontsize=7)
        # Log only where the range earns it; a narrow range on a log axis reads
        # as a straight line and hides the shape.
        lo, hi = ax.get_ylim()
        ax.set_yscale("log" if hi / max(lo, 1e-9) > 5 else "linear")
        ax.set_xlabel(xlabel)
        ax.set_title(f"$t={prec}$", color=INK)
        # The panels use independent vertical scales but share the same
        # measured quantity. A single leftmost label avoids collisions at the
        # narrow boundaries between small multiples.
        ax.set_ylabel("Perplexity" if ax is axes[0] else "")
        style_axes(ax)

    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(title, color=INK, fontsize=8, y=1.04)
    fig.savefig(out / filename)
    plt.close(fig)
    return filename


def _sort_key(entry):
    """Order summary rows so precisions sort numerically, not as strings."""
    key, mode = entry
    return tuple((str(p), 0) if isinstance(p, str) else ("", p) for p in key), mode


def print_summary(records):
    """Print the means and sample SDs the rerun tables in the paper quote."""
    for level in parse_sweep_logs.LEVELS:
        keys = ("precision",) if level == "global" else (
            ("block", "precision") if level in ("blockwise", "cumulative")
            else ("target", "precision"))
        s = parse_sweep_logs.summarize(records, level, keys=keys,
                                       arm=pick_arm(records, level))
        if not s:
            continue
        print(f"\n=== {level} (key: {', '.join(keys)})")
        for key, mode in sorted(s, key=_sort_key):
            center, lo, hi, n = s[(key, mode)]
            label = " ".join(str(p) for p in key)
            spread = f"SD={(hi - lo) / 2:.2f}" if hi > lo else ""
            print(f"  {label:<26s} {mode:<3s} n={n}  {center:10.2f}  {spread}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logs", type=Path, default=DEFAULT_LOGS,
                    help=f"sweep log root (default: {DEFAULT_LOGS})")
    ap.add_argument("--records", type=Path, default=DEFAULT_RECORDS,
                    help="distilled record file, used when the logs are absent "
                         f"(default: {DEFAULT_RECORDS})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"output directory (default: {DEFAULT_OUT})")
    ap.add_argument("--distill", action="store_true",
                    help="rewrite the record file from the logs")
    ap.add_argument("--summary", action="store_true",
                    help="print the parsed means and sample SDs")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any figure lacks logs")
    args = ap.parse_args()

    # Prefer the raw logs; fall back to the distilled records so that a clone
    # without the (large, gitignored) log tree can still rebuild the figures.
    if args.logs.is_dir():
        records = parse_sweep_logs.load(args.logs)
        source = args.logs
    elif args.records.is_file():
        records = parse_sweep_logs.read_csv(args.records)
        source = args.records
    else:
        sys.exit(f"no logs at {args.logs} and no records at {args.records}")

    print(f"parsed {len(records)} completed configurations from {source}")
    if not records:
        sys.exit("no completed configurations found; nothing to draw")

    if args.distill:
        if source != args.logs:
            sys.exit("--distill needs the raw logs, not the record file")
        parse_sweep_logs.write_csv(carry_forward(records, args.records),
                                   args.records)
        print(f"  wrote   {args.records}")

    args.out.mkdir(parents=True, exist_ok=True)
    setup()

    written, skipped = [], []
    for name, build in (
        ("global_sweep.pdf", lambda: fig_global(records, args.out)),
        ("sublayer_sweep.pdf", lambda: fig_sublayer(records, args.out)),
        ("component_sweep.pdf", lambda: fig_component(records, args.out)),
        ("blockwise_sensitivity.pdf", lambda: fig_positional(
            records, args.out, "blockwise", "blockwise_sensitivity.pdf",
            "Reduced block", "mlp_c_proj reduced in one block at a time",
            order_key=lambda b: int(b))),
        ("cumulative_propagation.pdf", lambda: fig_positional(
            records, args.out, "cumulative", "cumulative_propagation.pdf",
            "Blocks reduced", "mlp_c_proj reduced cumulatively in blocks 0..k",
            order_key=lambda b: (len(b), b))),
    ):
        (written if build() else skipped).append(name)

    for name in written:
        print(f"  wrote   {args.out / name}")
    for name in skipped:
        print(f"  SKIPPED {name}: {source} has no records for this level")

    if args.summary:
        print_summary(records)
        report_global_log_resolution(records)
    if skipped and args.strict:
        sys.exit(f"{len(skipped)} figure(s) could not be rebuilt")


if __name__ == "__main__":
    main()
