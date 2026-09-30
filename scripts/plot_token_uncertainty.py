#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8,<4"]
# ///
"""Per-token fluctuation penalty against predictive uncertainty, t=6,7,8, S=32.

    uv run scripts/plot_token_uncertainty.py --summary

tr(H_r) = 1 - sum_j p_r,j^2 (the collision-probability complement of the
model's own predictive distribution at token r) is the same quantity at both
sites and every precision, since it depends only on the shared reference
logits, not on which site is perturbed or how coarsely it is rounded. What
varies is how strongly a token's measured fluctuation penalty V_r tracks it,
and the direction of the raw sample seed cloud is added by the caller's own
correlation numbers (sigma vs tr(H)), so this script draws only V_r.

Two panels, one per site, sharing the logarithmic y-axis. Points show all
1020 scored tokens at each precision. Color and marker shape identify t;
semitransparent lines show decile medians. RN has no series because its
within-token fluctuation penalty is zero.

"""

import argparse
import csv
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_drift_fluctuation import setup

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "figures"

BIT_COLOR = {6: "#2a78d6", 7: "#bb6528", 8: "#36835b"}
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"

SITES = ("lm_head", "mlp_c_proj")
SITE_TITLE = {"lm_head": "Language-model head", "mlp_c_proj": "MLP down-projection"}
BITS = (6, 7, 8)
BIT_MARK = {6: dict(marker="o", linestyle="-"),
            7: dict(marker="^", linestyle="--"),
            8: dict(marker="s", linestyle=":")}
N_BINS = 10


def load(site, t):
    path = REPO / "results" / f"token_fisher_{site}_t{t}.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return ([float(r["trH"]) for r in rows], [float(r["V"]) for r in rows])


def style_axes(ax):
    ax.grid(True, which="major", color=GRID, linewidth=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK, labelsize=9, width=0.6)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)


def decile_medians(xs, ys, n_bins=N_BINS):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs_s = [xs[i] for i in order]
    ys_s = [ys[i] for i in order]
    n = len(xs_s)
    bx, by = [], []
    for b in range(n_bins):
        lo = n * b // n_bins
        hi = n * (b + 1) // n_bins
        if hi <= lo:
            continue
        bx.append(st.median(xs_s[lo:hi]))
        by.append(st.median(ys_s[lo:hi]))
    return bx, by


def fig_token_uncertainty(out, summary=False):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.1), sharey=True)
    for ax, site in zip(axes, SITES):
        style_axes(ax)
        for t in BITS:
            trH, V = load(site, t)
            color = BIT_COLOR[t]
            ax.scatter(trH, V, s=5, color=color, alpha=0.28,
                       marker=BIT_MARK[t]["marker"], edgecolors="none", zorder=2)
            bx, by = decile_medians(trH, V)
            ax.plot(bx, by, color=color, linewidth=1.2, alpha=0.65, zorder=3,
                    markersize=4, markerfacecolor=color, markeredgecolor=color,
                    label=f"$t={t}$", **BIT_MARK[t])
            if summary:
                print(f"  {site} t={t}: n={len(trH)}, "
                      f"decile medians x={['%.2f' % v for v in bx]}, "
                      f"y={['%.3g' % v for v in by]}")
        ax.set_yscale("log")
        ax.set_xlabel("Predictive uncertainty\n" + r"$\operatorname{tr}(H_r)=1-\|p_r\|_2^2$")
        ax.set_title(SITE_TITLE[site], color=INK, fontsize=9.5)
    axes[0].set_ylabel(r"per-token fluctuation $V_r$ (nats, log scale)")
    axes[1].legend(frameon=False, loc="lower right", fontsize=8,
                    handletextpad=0.5, borderaxespad=0.3)
    fig.tight_layout()
    path = out / "token_uncertainty.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    if summary:
        print(f"  wrote {path.name} and .png")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    setup()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_token_uncertainty(args.out, args.summary)


if __name__ == "__main__":
    main()
