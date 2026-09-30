#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8,<4", "numpy>=1.26"]
# ///
"""Expected per-logit drift by reference probability, in absolute value.

    uv run scripts/plot_logit_drift_bins.py --summary

The probability-binned panels of `plot_logit_drift_map.py`, on their own and on
one shared logarithmic axis of |mean drift| rather than two symmetric-log signed
ones. The magnitudes span five decades between the two sites and the three
precisions, which the signed scale compresses and separate scales hide; the sign
is carried by the marker fill instead, and the one standard error over seeds by
a shaded envelope around each SR curve. RN is a single deterministic run and has
no envelope.

A point whose envelope reaches the bottom of the panel is not resolved from
zero. Points below the window are drawn as carets on the axis, as in the drift
boxplot, rather than silently clipped.

Curves and envelope boundaries are drawn as monotone cubic interpolants
(Fritsch-Carlson) of the bin values in the logarithm, evaluated on a dense
grid. The interpolation is shape preserving: it introduces no extremum between
two bins and cannot overshoot their values. Markers sit on the measured bins;
everything between them is drawn, not measured.

Reads the same `results/logit_drift/logit_drift_<site>_t<t>.npz` bundles.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from plot_drift_fluctuation import setup
from plot_logit_drift_map import (BITS, BIT_LINE, BIT_MARK, DEFAULT_IN, INK,
                                  MUTED, RN_COLOR, SITE_TITLE, SITES, SR_COLOR,
                                  load, pbin_centers, style_axes)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "figures"

# Fraction of the smallest standard error the panel's floor is set to: low
# enough that an envelope reaching zero is visibly open-ended, high enough that
# the resolved values keep most of the axis.
FLOOR_FRAC = 0.25


def pchip_slopes(x, y):
    """Fritsch-Carlson tangents: monotone on every interval, no overshoot."""
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros_like(y)
    same = d[:-1] * d[1:] > 0
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        harmonic = (w1 + w2) / (w1 / d[:-1] + w2 / d[1:])
    m[1:-1] = np.where(same, harmonic, 0.0)
    for i, (dd, hh) in ((0, (d[0], h[0])), (-1, (d[-1], h[-1]))):
        j = 1 if i == 0 else -2
        other_d, other_h = (d[1], h[1]) if i == 0 else (d[-2], h[-2])
        end = ((2.0 * hh + other_h) * dd - hh * other_d) / (hh + other_h)
        if np.sign(end) != np.sign(dd):
            end = 0.0
        elif np.sign(dd) != np.sign(other_d) and abs(end) > 3.0 * abs(dd):
            end = 3.0 * dd
        m[i] = end
    return m


def smooth(x, y, n=240):
    """Shape-preserving curve through (x, y), sampled on a dense grid."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3:
        xq = np.linspace(x[0], x[-1], n)
        return xq, np.interp(xq, x, y)
    m = pchip_slopes(x, y)
    xq = np.linspace(x[0], x[-1], n)
    i = np.clip(np.searchsorted(x, xq) - 1, 0, len(x) - 2)
    h = x[i + 1] - x[i]
    s_ = (xq - x[i]) / h
    s2, s3 = s_ * s_, s_ * s_ * s_
    yq = ((2 * s3 - 3 * s2 + 1) * y[i]
          + (s3 - 2 * s2 + s_) * h * m[i]
          + (-2 * s3 + 3 * s2) * y[i + 1]
          + (s3 - s2) * h * m[i + 1])
    return xq, yq


def smooth_log(x, y, n=240):
    """The same curve, interpolated in the logarithm the axis is drawn in."""
    xq, lq = smooth(x, np.log10(y), n)
    return xq, 10.0 ** lq


def series(cell, mode):
    """Mean drift per probability bin, its standard error, and the bin mask."""
    keep = cell["sr_pbin_count"] > 0
    cnt = np.maximum(cell["sr_pbin_count"], 1.0)
    m = cell[f"{mode}_pbin_sum"] / cnt
    if mode == "sr" and "sr_pbin_mean_se" in cell:
        se = cell["sr_pbin_mean_se"]
    else:
        se = np.zeros_like(m)
    return m, se, keep


def panel_ceiling(cells, sites):
    """Top of the shared axis: above every point and envelope drawn on it."""
    hi = 0.0
    for site in sites:
        for t in BITS:
            cell = cells.get((site, t))
            if cell is None:
                continue
            for mode in ("sr", "rn"):
                if f"{mode}_pbin_sum" not in cell:
                    continue
                m, se, keep = series(cell, mode)
                hi = max(hi, float(np.max(np.abs(m[keep]) + se[keep])))
    return 1.7 * hi


def panel_floor(cells, sites):
    """Bottom of the shared axis: below every standard error drawn on it."""
    ses = []
    for site in sites:
        for t in BITS:
            cell = cells.get((site, t))
            if cell is None:
                continue
            _, se, keep = series(cell, "sr")
            ses.extend(se[keep][se[keep] > 0])
    return FLOOR_FRAC * min(ses) if ses else 1e-6


def fig_logit_drift_bins(cells, out, summary=False):
    # Sized near the text width so that \linewidth scales it by ~0.95 rather
# than 0.73: the panels occupy the same space either way, the labels do not.
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.7), sharey=True)
    # One axis for both sites: the down-projection's drift is two orders above
    # the head's, and a shared scale is what shows that.
    floor = panel_floor(cells, SITES)
    ceiling = panel_ceiling(cells, SITES)
    for ax, site in zip(axes, SITES):
        style_axes(ax)
        edges = None
        for t in BITS:
            cell = cells.get((site, t))
            if cell is None:
                continue
            edges = cell["pbin_edges"]
            for mode, color in (("rn", RN_COLOR), ("sr", SR_COLOR)):
                if f"{mode}_pbin_sum" not in cell:
                    continue
                m, se, keep = series(cell, mode)
                x = np.arange(len(m))[keep]
                a = np.abs(m[keep])
                if mode == "sr":
                    lo = np.maximum(a - se[keep], floor)
                    hi = a + se[keep]
                    xq, loq = smooth_log(x, lo)
                    _, hiq = smooth_log(x, hi)
                    ax.fill_between(xq, np.maximum(loq, floor), hiq, color=color,
                                    alpha=0.18, linewidth=0, zorder=2)
                xq, aq = smooth_log(x, np.maximum(a, floor))
                ax.plot(xq, np.maximum(aq, floor), color=color,
                        linestyle=BIT_LINE[t], linewidth=1.1, zorder=3)
                # marker fill carries the sign of the mean
                pos = m[keep] >= 0
                for sel, face in ((pos, color), (~pos, "white")):
                    if sel.any():
                        ax.plot(x[sel], np.maximum(a[sel], floor),
                                linestyle="none", marker=BIT_MARK[t],
                                markersize=3.8, markerfacecolor=face,
                                markeredgecolor=color, markeredgewidth=0.9,
                                zorder=4)
                below = a < floor
                if below.any():
                    ax.plot(x[below], np.full(below.sum(), floor), linestyle="none",
                            marker="v", markersize=3.2, color=color, zorder=5)
                if summary:
                    print(f"  {site} t={t} {mode}: "
                          + " ".join(f"{v:+.2e}" for v in m[keep]))
        ax.set_yscale("log")
        ax.set_ylim(floor, ceiling)
        if edges is not None:
            labels = pbin_centers(edges)
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, fontsize=7.5, rotation=40, ha="right")
        ax.set_xlabel(r"reference probability $p_{r,j}$", fontsize=9)
        ax.set_title(SITE_TITLE[site], color=INK, fontsize=9.5)
    axes[0].set_ylabel("$|$mean drift in bin$|$\n(logit units, log scale)",
                       fontsize=9)
    axes[1].tick_params(labelleft=False)

    handles = [
        Line2D([], [], color=SR_COLOR, marker="o", markersize=4, linewidth=1.1,
               label="SR, $S=32$"),
        Line2D([], [], color=RN_COLOR, marker="o", markersize=4, linewidth=1.1,
               label="RN"),
        Line2D([], [], color=MUTED, marker="o", markersize=4, linestyle="none",
               label="positive"),
        Line2D([], [], color=MUTED, marker="o", markersize=4, linestyle="none",
               markerfacecolor="white", label="negative"),
    ] + [Line2D([], [], color=INK, marker=BIT_MARK[t], linestyle=BIT_LINE[t],
                markersize=4, linewidth=1.0, label=f"$t={t}$") for t in BITS]
    fig.legend(handles=handles, frameon=False, fontsize=7.5, ncol=7,
               loc="lower center", bbox_to_anchor=(0.5, 0.005),
               handletextpad=0.35, columnspacing=0.7)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    path = out / "logit_drift_bins.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    if summary:
        print(f"  wrote {path.name} and .png")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="in_dir", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    cells = {}
    for site in SITES:
        for t in BITS:
            c = load(args.in_dir, site, t)
            if c is not None:
                cells[(site, t)] = c
    if not cells:
        raise SystemExit(f"no bundles under {args.in_dir}")
    setup()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_logit_drift_bins(cells, args.out, args.summary)


if __name__ == "__main__":
    main()
