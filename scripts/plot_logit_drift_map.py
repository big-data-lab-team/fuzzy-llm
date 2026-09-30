#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8,<4", "numpy>=1.26"]
# ///
"""Where the expected per-logit perturbation lives, and where it cancels.

    uv run scripts/plot_logit_drift_map.py --summary

The manuscript reports a pooled sample drift at the head whose mean is not
resolved from zero. A pooled mean over N*V = 51,262,140 token-coordinate pairs
can hide structure, so this figure opens the pool up.

Left column: the RMS of the estimated drift at four levels of pooling --
single logits b_hat[r,j], the mean over coordinates within a token, the mean
over tokens at a fixed coordinate, and the pooled mean -- against the RMS of
the corresponding finite-seed standard error, all in logit units. Seed-level
standard errors are used throughout: seeds are the independent replicates.
An estimate that lies on its own noise floor carries no resolved drift at
that level; coherent drift instead survives pooling, because averaging
reduces noise by 1/sqrt(n) and leaves structure untouched.

Right column: the mean drift of coordinates grouped by their reference
probability p_r,j. The loss sees only p-weighted contrasts between
coordinates, so a drift that varies with p is loss-relevant while a uniform
one is not.

Reads `results/logit_drift/logit_drift_<site>_t<t>.npz`, produced on the
cluster by `experiments/LLM/logit_drift_map.py` over the raw delta.npy
captures.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from plot_drift_fluctuation import setup

REPO = Path(__file__).resolve().parent.parent
DEFAULT_IN = REPO / "results" / "logit_drift"
DEFAULT_OUT = REPO / "figures"

# Same categorical slots as the other drift figures, so they read as a family.
SR_COLOR = "#2a78d6"
RN_COLOR = "#eb6834"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"
FLOOR = "#9a9a9a"

SITES = ("lm_head", "mlp_c_proj")
SITE_TITLE = {"lm_head": "Language-model head",
              "mlp_c_proj": "MLP down-projection"}
BITS = (6, 7, 8)
BIT_MARK = {6: "o", 7: "^", 8: "s"}
BIT_LINE = {6: "-", 7: "--", 8: ":"}

LEVELS = ("logit", "token", "coord", "pooled")
LEVEL_LABEL = {
    "logit": "single\nlogit",
    "token": "mean over\ncoordinates",
    "coord": "mean over\ntokens",
    "pooled": "pooled\nmean",
}


def style_axes(ax):
    ax.grid(True, which="major", color=GRID, linewidth=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK, labelsize=8.5, width=0.6)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, dtype=float) ** 2)))


def ladder(cell, mode):
    """RMS of the drift estimate at each pooling level, in logit units."""
    g = lambda k: cell[f"{mode}_{k}"]
    return {
        "logit": float(g("pooled_rms")),
        "token": rms(g("token_mean")),
        "coord": rms(g("coord_mean")),
        "pooled": abs(float(g("pooled_mean"))),
    }


def noise_ladder(cell):
    """RMS of the seed-level standard error of each SR estimate."""
    return {
        "logit": float(cell["sr_se_rms"]),
        "token": rms(cell["sr_token_mean_se"]),
        "coord": rms(cell["sr_coord_mean_se"]),
        "pooled": float(cell["sr_pooled_mean_se"]),
    }


def pbin_centers(edges):
    """Label each bin by its log10 p range; the open ends get one-sided labels."""
    lo, hi = edges[:-1], edges[1:]
    out = []
    for a, b in zip(lo, hi):
        if not np.isfinite(a):
            out.append(rf"$<10^{{{b:.0f}}}$")
        elif not np.isfinite(b):
            out.append(rf"$>10^{{{a:.0f}}}$")
        else:
            out.append(rf"$10^{{{a:.0f}}}$")
    return out


def load(in_dir, site, t):
    path = in_dir / f"logit_drift_{site}_t{t}.npz"
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def fig_logit_drift(cells, out, summary=False):
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.4))
    x = np.arange(len(LEVELS))
    lines = []
    for row, site in enumerate(SITES):
        axl, axr = axes[row]
        style_axes(axl)
        style_axes(axr)
        for t in BITS:
            cell = cells.get((site, t))
            if cell is None:
                continue
            nf = noise_ladder(cell)
            axl.plot(x, [nf[k] for k in LEVELS], color=FLOOR, linewidth=2.6,
                     alpha=0.45, solid_capstyle="round", zorder=1)
            for mode, color in (("rn", RN_COLOR), ("sr", SR_COLOR)):
                if f"{mode}_pooled_rms" not in cell:
                    continue
                lad = ladder(cell, mode)
                axl.plot(x, [lad[k] for k in LEVELS], color=color,
                         marker=BIT_MARK[t], linestyle=BIT_LINE[t],
                         markersize=4, linewidth=1.1, zorder=3,
                         markerfacecolor="white" if mode == "rn" else color,
                         markeredgecolor=color)
            if summary:
                print(f"  {site} t={t}: "
                      f"SR/noise = "
                      f"{[round(ladder(cell,'sr')[k]/nf[k], 3) for k in LEVELS]}, "
                      f"RN/noise = "
                      f"{['%.3g' % (ladder(cell,'rn')[k]/nf[k]) for k in LEVELS]}")
        axl.set_yscale("log")
        axl.set_xticks(x)
        axl.set_xticklabels([LEVEL_LABEL[k] for k in LEVELS], fontsize=7.5)
        axl.set_ylabel("RMS of estimated drift\n(logit units)", fontsize=9)
        axl.set_title(SITE_TITLE[site], color=INK, fontsize=9.5)

        # right panel: drift against reference probability
        edges = None
        for t in BITS:
            cell = cells.get((site, t))
            if cell is None:
                continue
            edges = cell["pbin_edges"]
            cnt = np.maximum(cell["sr_pbin_count"], 1.0)
            keep = cell["sr_pbin_count"] > 0
            xb = np.arange(len(cnt))
            for mode, color in (("rn", RN_COLOR), ("sr", SR_COLOR)):
                if f"{mode}_pbin_sum" not in cell:
                    continue
                m = cell[f"{mode}_pbin_sum"] / cnt
                if mode == "sr" and "sr_pbin_mean_se" in cell:
                    axr.errorbar(xb[keep], m[keep], yerr=cell["sr_pbin_mean_se"][keep],
                                 color=color, marker=BIT_MARK[t],
                                 linestyle=BIT_LINE[t], markersize=3.5,
                                 linewidth=1.1, elinewidth=0.8, capsize=1.5,
                                 zorder=3)
                else:
                    axr.plot(xb[keep], m[keep], color=color, marker=BIT_MARK[t],
                             linestyle=BIT_LINE[t], markersize=3.5,
                             linewidth=1.1, zorder=3,
                             markerfacecolor="white", markeredgecolor=color)
        axr.axhline(0.0, color=MUTED, linewidth=0.7, zorder=2)
        if edges is not None:
            labels = pbin_centers(edges)
            axr.set_xticks(np.arange(len(labels)))
            axr.set_xticklabels(labels, fontsize=7.5, rotation=40, ha="right")
        axr.set_yscale("symlog", linthresh=1e-3)
        axr.set_ylabel("mean drift in bin\n(logit units)", fontsize=9)
        axr.set_title(SITE_TITLE[site], color=INK, fontsize=9.5)
    axes[1][0].set_xlabel("pooling level", fontsize=9)
    axes[1][1].set_xlabel(r"reference probability $p_{r,j}$", fontsize=9)

    handles = [
        Line2D([], [], color=SR_COLOR, marker="o", markersize=4, linewidth=1.1,
               label="SR, $S=32$"),
        Line2D([], [], color=RN_COLOR, marker="o", markersize=4, linewidth=1.1,
               markerfacecolor="white", label="RN"),
        Line2D([], [], color=FLOOR, linewidth=2.6, alpha=0.45,
               label="SR seed-noise floor"),
    ] + [Line2D([], [], color=INK, marker=BIT_MARK[t], linestyle=BIT_LINE[t],
                markersize=4, linewidth=1.0, label=f"$t={t}$") for t in BITS]
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=6,
               loc="lower center", bbox_to_anchor=(0.5, -0.005),
               handletextpad=0.5, columnspacing=1.1)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    path = out / "logit_drift_map.pdf"
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
    fig_logit_drift(cells, args.out, args.summary)


if __name__ == "__main__":
    main()
