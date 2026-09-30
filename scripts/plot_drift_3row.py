#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8,<4", "numpy>=1.26"]
# ///
"""Figure: 3-row × 2-column drift--fluctuation decomposition.

Row 1 – Operational perturbation RMS per coordinate: sqrt(E[Δ_j²])
         for SR and |Δ_j| for deterministic RN. The finite-seed estimate
         uses sqrt(mean_s Δ_j²), with no seed averaging of the perturbation
         before taking its magnitude.

Row 2 – Centered drift: per-token vocabulary mean removed, showing the
         loss-relevant non-uniform component (maps to Q_m via Fisher weighting).

Row 3 – Effective SR fluctuation: per-coordinate std of the p-centered
         fluctuation std_s(Δ_j - E_p[Δ]) across the 32 SR runs, showing
         the loss-relevant noise the softmax charges through V_SR.
         RN panels show ξ_RN = 0.

Data source: results/logit_drift/logit_drift_{site}_t{t}.npz (rows i-ii)
and results/logit_fluct/logit_fluct_{site}_t{t}.npz (row iii). By default,
boxes use exact summaries computed from all N*V pairs by
full_grid_drift_boxes.py. Pass --sampled for the earlier 200,000-pair plot.

Run: uv run scripts/plot_drift_3row.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parent.parent
DEFAULT_IN = REPO / "results" / "logit_drift"
DEFAULT_FLUCT_IN = REPO / "results" / "logit_fluct"
DEFAULT_FULL_IN = REPO / "results" / "logit_full_boxes"
DEFAULT_OUT = REPO / "figures"

SR_COLOR = "#2a78d6"
SR_LIGHT = "#7db3f0"     # lighter SR for the total perturbation
RN_COLOR = "#eb6834"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"

SITES = ("lm_head", "mlp_c_proj")
SITE_TITLE = {"lm_head": "Head",
              "mlp_c_proj": "MLP"}
BITS = (6, 7, 8)
MODES = ("rn", "sr")
COLOR = {"sr": SR_COLOR, "rn": RN_COLOR}

WHISKER_ROOM = 2.2


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.5,
        "figure.dpi": 300,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
    })


def style_axes(ax):
    ax.grid(True, which="major", color=GRID, linewidth=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK, labelsize=7.5, width=0.6)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)


def load_captures(indir):
    """Load per-site, per-precision logit drift captures."""
    captures = {}
    for site in SITES:
        for t in BITS:
            p = Path(indir) / f"logit_drift_{site}_t{t}.npz"
            if p.exists():
                captures[(site, t)] = dict(np.load(p))
    return captures


def load_fluct_captures(indir):
    """Load per-site, per-precision centered-fluctuation captures."""
    captures = {}
    for site in SITES:
        for t in BITS:
            p = Path(indir) / f"logit_fluct_{site}_t{t}.npz"
            if p.exists():
                captures[(site, t)] = dict(np.load(p))
    return captures


def load_full_captures(indir):
    """Load six-number box summaries computed on the complete logit grid."""
    captures = {}
    for site in SITES:
        for t in BITS:
            p = Path(indir) / f"full_grid_drift_{site}_t{t}.npz"
            if not p.exists():
                raise FileNotFoundError(f"missing full-grid summary: {p}")
            captures[(site, t)] = dict(np.load(p))
    return captures


BOX_FIELDS = ("q01", "q25", "q50", "q75", "q99", "mean")
FULL_BOX_KEYS = {
    "rms_rn": "rn_rms_box",
    "rms_sr": "sr_rms_box",
    "cent_rn": "rn_cent_box",
    "cent_sr": "sr_cent_box",
    "fluc_sr": "sr_fluc_box",
}


def full_box_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(BOX_FIELDS),) or not np.isfinite(values).all():
        raise ValueError("invalid full-grid box summary")
    return dict(zip(BOX_FIELDS, values))


def compute_boxplot_stats(values, whisk=(0.01, 0.99)):
    """Compute boxplot statistics from a 1D array."""
    q01, q25, q50, q75, q99 = np.percentile(values, [
        whisk[0]*100, 25, 50, 75, whisk[1]*100])
    return {"q01": q01, "q25": q25, "q50": q50, "q75": q75, "q99": q99,
            "mean": np.mean(values)}


def auto_ylim(all_stats, pad=0.08):
    """Compute y limits that contain all boxes, extending toward whiskers."""
    q25s = [s["q25"] for s in all_stats]
    q75s = [s["q75"] for s in all_stats]
    lo_b, hi_b = min(q25s), max(q75s)
    span = hi_b - lo_b
    if span < 1e-12:
        span = 1.0
    lo_ws = [s["q01"] for s in all_stats]
    hi_ws = [s["q99"] for s in all_stats]
    lo = max(min(lo_ws), lo_b - WHISKER_ROOM * span)
    hi = min(max(hi_ws), hi_b + WHISKER_ROOM * span)
    return lo - pad * (hi - lo), hi + pad * (hi - lo)


def draw_box(ax, x, stats, color, ylim, half=0.17, alpha=0.30):
    """Draw one box with whiskers and carets for clipped whiskers."""
    q25, q50, q75 = stats["q25"], stats["q50"], stats["q75"]
    lo, hi = stats["q01"], stats["q99"]
    mean = stats["mean"]

    for end, y0, caret in ((lo, q25, "v"), (hi, q75, "^")):
        stop = min(max(end, ylim[0]), ylim[1])
        ax.plot([x, x], [y0, stop], color=color, linewidth=0.9, zorder=2)
        if ylim[0] < end < ylim[1]:
            ax.plot([x - half/2, x + half/2], [end]*2,
                    color=color, linewidth=0.9, zorder=2)
        else:
            ax.plot([x], [stop], color=color, marker=caret,
                    markersize=4, markeredgewidth=0, zorder=6)

    ax.add_patch(plt.Rectangle(
        (x - half, q25), 2*half, q75 - q25, facecolor=color,
        alpha=alpha, edgecolor=color, linewidth=0.8, zorder=3))
    ax.plot([x - half, x + half], [q50]*2, color=color,
            linewidth=1.8, solid_capstyle="butt", zorder=4)
    # mean diamond
    ax.plot([x], [mean], color=color, marker="D", markersize=3.0,
            markerfacecolor="white", markeredgecolor=color,
            markeredgewidth=0.9, linestyle="none", zorder=6)


def xpos(t, mode, off=0.20):
    return t + (off if mode == "sr" else -off)


# V_SR values from tab:decomposition (nats). Only used as a fallback
# scalar sqrt(2*V_SR) marker for a (site, t) cell missing per-coordinate
# data in fluct_captures.
V_SR = {
    ("lm_head",    6): 4.606,
    ("lm_head",    7): 1.951,
    ("lm_head",    8): 0.766,
    ("mlp_c_proj", 6): 0.0828,
    ("mlp_c_proj", 7): 0.0252,
    ("mlp_c_proj", 8): 0.0067,
}


def make_figure(captures, fluct_captures, outdir, full_captures=None):
    fig, axes = plt.subplots(3, 2, figsize=(5.8, 4.5))

    row_labels = [
        r"(i) RMS perturbation",
        r"(ii) centered drift $\tilde b_j$",
        r"(iii) centered std$(\tilde\xi_j)$",
    ]

    for col, site in enumerate(SITES):
        row_data = {t: {} for t in BITS}

        for t in BITS:
            cap = captures.get((site, t))
            if cap is None:
                continue

            n_scored = int(cap["n_scored"])
            vocab = int(cap["vocab"])
            sub_idx = cap["sub_index"]  # (200000,) flat indices into (N, V)
            token_idx = sub_idx // vocab
            S = len(cap["seeds"])
            full = None
            if full_captures is not None:
                full = full_captures[(site, t)]
                if (str(full["site"]) != site
                        or int(full["t"]) != t
                        or int(full["n_scored"]) != n_scored
                        or int(full["vocab"]) != vocab
                        or int(full["n_points"]) != n_scored * vocab
                        or not np.array_equal(full["seeds"], cap["seeds"])):
                    raise ValueError(f"full-grid summary metadata mismatch for {site} t={t}")

            # ----------------------------------------------------------
            # Row 1: Operational perturbation magnitude per run
            # ----------------------------------------------------------
            # RN: deterministic, so |Δ_j| = |b_j|
            rn_sub_b = cap["rn_sub_b"].astype(np.float64)
            row_data[t]["rms_rn"] = compute_boxplot_stats(np.abs(rn_sub_b))

            # SR: operational second moment sqrt(b_j² + σ_j²)
            sr_sub_b = cap["sr_sub_b"].astype(np.float64)  # drift at each coord
            sr_sub_se = cap["sr_sub_se"].astype(np.float64)  # SE = σ/√S
            sr_sigma = sr_sub_se * np.sqrt(S)               # operational σ_j
            sr_rms_per_run = np.sqrt(sr_sub_b**2 + (S - 1) / S * sr_sigma**2)
            row_data[t]["rms_sr"] = compute_boxplot_stats(sr_rms_per_run)

            # ----------------------------------------------------------
            # Row 2: Centered drift (per-token vocab mean removed)
            # ----------------------------------------------------------
            rn_token_mean = cap["rn_token_mean"]  # (N,)
            sr_token_mean = cap["sr_token_mean"]  # (N,)
            rn_cent = rn_sub_b - rn_token_mean[token_idx]
            sr_cent = sr_sub_b - sr_token_mean[token_idx]
            row_data[t]["cent_rn"] = compute_boxplot_stats(rn_cent)
            row_data[t]["cent_sr"] = compute_boxplot_stats(sr_cent)

            # ----------------------------------------------------------
            # Row 3: p-centered fluctuation std (per coordinate)
            # ----------------------------------------------------------
            fcap = fluct_captures.get((site, t))
            if fcap is not None:
                row_data[t]["fluc_sr"] = compute_boxplot_stats(
                    fcap["sr_fluc_sub"].astype(np.float64))
            else:
                row_data[t]["fluc_rms_sr"] = np.sqrt(
                    2.0 * V_SR.get((site, t), np.nan))

            if full is not None:
                for row_key, full_key in FULL_BOX_KEYS.items():
                    row_data[t][row_key] = full_box_stats(full[full_key])
                row_data[t].pop("fluc_rms_sr", None)

        # ==============================================================
        # Row 1: Operational perturbation magnitude per run
        # ==============================================================
        ax = axes[0, col]
        style_axes(ax)

        all_stats = []
        for t in BITS:
            for k in ("rms_rn", "rms_sr"):
                if k in row_data[t]:
                    all_stats.append(row_data[t][k])
        if all_stats:
            ylim = auto_ylim(all_stats)
            ylim = (max(ylim[0], 0.0), ylim[1])  # magnitude is non-negative
            ax.set_ylim(*ylim)

            for t in BITS:
                # RN: full perturbation (= drift)
                if "rms_rn" in row_data[t]:
                    draw_box(ax, xpos(t, "rn"), row_data[t]["rms_rn"],
                             RN_COLOR, ylim)

                # SR: operational RMS per run
                if "rms_sr" in row_data[t]:
                    draw_box(ax, xpos(t, "sr"), row_data[t]["rms_sr"],
                             SR_COLOR, ylim)

        ax.set_xlim(min(BITS) - 0.5, max(BITS) + 0.5)
        ax.set_xticks(list(BITS))
        ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel("logits")
        ax.set_title(f"{SITE_TITLE[site]} — {row_labels[0]}", color=INK, loc="left", fontsize=8.5)

        # ==============================================================
        # Row 2: Centered drift (non-uniform component)
        # ==============================================================
        ax = axes[1, col]
        style_axes(ax)

        all_stats = []
        for t in BITS:
            for k in ("cent_rn", "cent_sr"):
                if k in row_data[t]:
                    all_stats.append(row_data[t][k])
        if all_stats:
            ylim = auto_ylim(all_stats)
            if ylim[0] < 0 < ylim[1]:
                ax.axhline(0.0, color=INK, linewidth=0.6, zorder=1)
            ax.set_ylim(*ylim)
            for t in BITS:
                if "cent_rn" in row_data[t]:
                    draw_box(ax, xpos(t, "rn"), row_data[t]["cent_rn"],
                             RN_COLOR, ylim)
                if "cent_sr" in row_data[t]:
                    draw_box(ax, xpos(t, "sr"), row_data[t]["cent_sr"],
                             SR_COLOR, ylim)

        ax.set_xlim(min(BITS) - 0.5, max(BITS) + 0.5)
        ax.set_xticks(list(BITS))
        ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel("logits")
        ax.set_title(f"{SITE_TITLE[site]} — {row_labels[1]}", color=INK, loc="left", fontsize=8.5)

        # ==============================================================
        # Row 3: p-centered fluctuation std
        # ==============================================================
        ax = axes[2, col]
        style_axes(ax)

        box_ts = [t for t in BITS if "fluc_sr" in row_data[t]]
        marker_ts = [t for t in BITS
                     if "fluc_rms_sr" in row_data[t]
                     and not np.isnan(row_data[t]["fluc_rms_sr"])]

        if box_ts:
            all_stats = [row_data[t]["fluc_sr"] for t in box_ts]
            ylim = auto_ylim(all_stats)
            ylim = (max(ylim[0], 0.0), ylim[1])
            if marker_ts:
                ylim = (ylim[0],
                        max(ylim[1],
                            max(row_data[t]["fluc_rms_sr"]
                                for t in marker_ts) * 1.1))
            ax.set_ylim(*ylim)
            for t in box_ts:
                draw_box(ax, t, row_data[t]["fluc_sr"], SR_COLOR, ylim)
        else:
            ylim = (0.0, max((row_data[t]["fluc_rms_sr"]
                              for t in marker_ts), default=1.0) * 1.35)
            ax.set_ylim(*ylim)

        for t in marker_ts:
            if t in box_ts:
                continue
            rms = row_data[t]["fluc_rms_sr"]
            ax.plot([t - 0.25, t + 0.25], [rms, rms],
                    color=SR_COLOR, linewidth=1.8,
                    linestyle="--", solid_capstyle="butt", zorder=4)
            ax.plot([t], [rms], color=SR_COLOR, marker="D",
                    markersize=4.5, markerfacecolor="white",
                    markeredgecolor=SR_COLOR, markeredgewidth=1.0,
                    linestyle="none", zorder=6)

        ax.set_xlim(min(BITS) - 0.5, max(BITS) + 0.5)
        ax.set_xticks(list(BITS))
        ax.set_xlabel("virtual precision $t$ (bits)")
        if col == 0:
            ax.set_ylabel("logits")
        ax.set_title(f"{SITE_TITLE[site]} — {row_labels[2]}", color=INK, loc="left", fontsize=8.5)

        ax.annotate(
            "RN: deterministic ($\\xi_{\\mathrm{RN}} = 0$)",
            xy=(0.5, 0.92), xycoords="axes fraction",
            ha="center", va="top", fontsize=7.0,
            color=RN_COLOR, fontstyle="italic")

    # ---- Legend ----
    handles = [
        Line2D([], [], color=RN_COLOR, marker="s", linestyle="none",
               markersize=4.5, label="RN"),
        Line2D([], [], color=SR_COLOR, marker="o", linestyle="none",
               markersize=4.5, markeredgecolor=SR_COLOR,
               label="SR"),
        Line2D([], [], color=INK, marker="D", markersize=3.0,
               markerfacecolor="white", linestyle="none", label="mean"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=7.5)

    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=1.5)

    path = Path(outdir) / "drift_boxplot.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(DEFAULT_IN))
    ap.add_argument("--fluct-in-dir", default=str(DEFAULT_FLUCT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--full-grid-dir", default=str(DEFAULT_FULL_IN),
                    help="directory of exact box summaries over all token-coordinate pairs")
    ap.add_argument("--sampled", action="store_true",
                    help="use the earlier 200,000-pair approximation")
    args = ap.parse_args()

    setup()
    captures = load_captures(args.in_dir)
    if not captures:
        raise SystemExit(f"no logit_drift_*.npz files in {args.in_dir}")
    print(f"loaded {len(captures)} captures")
    fluct_captures = load_fluct_captures(args.fluct_in_dir)
    if fluct_captures:
        print(f"loaded {len(fluct_captures)} fluctuation captures")
    else:
        print(f"no logit_fluct_*.npz files in {args.fluct_in_dir}; "
              "row (iii) will fall back to sqrt(2*V_SR) markers")
    full_captures = None if args.sampled else load_full_captures(args.full_grid_dir)
    if full_captures is not None:
        print(f"loaded {len(full_captures)} complete-grid summaries")
    p = make_figure(captures, fluct_captures, args.out, full_captures)
    print(f"  wrote {p.name} and .png")


if __name__ == "__main__":
    main()
