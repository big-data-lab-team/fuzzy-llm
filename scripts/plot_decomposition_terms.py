#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.24,<3", "matplotlib>=3.8,<4"]
# ///
"""Shared loss-decomposition panels for the head and MLP down-projection.

    uv run scripts/plot_decomposition_terms.py
    uv run scripts/plot_decomposition_terms.py --summary

Writes site_decomposition_terms.pdf/.png: stacked bar decomposition of
excess loss into D, Q, V, and R for the head and MLP down-projection at t=6,7,8.
Positive terms stack upward from zero and negative terms stack downward,
with a solid horizontal mark indicating the net measured excess loss.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

REPO = Path(__file__).resolve().parent.parent
QUADRATIC_CSV = REPO / "results" / "quadratic_loss_summary.csv"
S32_CSV = REPO / "results" / "distributions_s32.csv"
DEFAULT_OUT = REPO / "figures"

SR_COLOR = "#2a78d6"
RN_COLOR = "#eb6834"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"

STYLE = {
    "sr": dict(color=SR_COLOR, marker="o", linestyle="-", label="SR"),
    "rn": dict(color=RN_COLOR, marker="s", linestyle="--", label="RN"),
}
TERMS = ("D", "Q", "V", "R")
TERM_LABEL = {"D": "(a) Signed drift $D$", "Q": "(b) Drift curvature $Q$",
              "V": "(c) Variance $V$", "R": "(d) Remainder $R$"}


def load(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(row, key):
    if row is None:
        return None
    v = row.get(key, "")
    if v in (None, ""):
        return None
    return float(v)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK, labelsize=7.5)
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def collect_site(site, bits):
    """{(mode, t): {D,Q,V,R,se_D,se_Q,se_V,se_R}} merging quadratic_loss_summary
    (S=8, all t) with distributions_s32 (S=32, t=6,7,8, overrides)."""
    cells = {}
    for r in load(QUADRATIC_CSV):
        if r["site"] != site or int(r["t"]) not in bits:
            continue
        cells[(r["mode"], int(r["t"]))] = r
    for r in load(S32_CSV):
        if r["site"] != site or int(r["t"]) not in bits:
            continue
        cells[(r["mode"], int(r["t"]))] = r
    return cells


def fig_terms(out, summary=False):
    bits = (6, 7, 8)
    sites = ("lm_head", "mlp_c_proj")
    site_label = {"lm_head": "Head", "mlp_c_proj": "MLP down-projection"}
    site_marker = {"lm_head": "o", "mlp_c_proj": "^"}
    cells = {site: collect_site(site, bits) for site in sites}
    for site in sites:
        for mode in ("rn", "sr"):
            for t in bits:
                row = cells[site].get((mode, t))
                if row is None or any(num(row, term) is None for term in TERMS):
                    raise ValueError(f"Missing decomposition: {site}, {mode}, t={t}")
                if mode == "sr" and int(row["seeds"]) != 32:
                    raise ValueError(f"Expected 32 SR seeds: {site}, t={t}")

    # Term palette: colorblind-accessible, matching the paper's aesthetic
    term_colors = {
        "D": "#2a9d8f",  # teal (drift)
        "Q": "#e76f51",  # coral (curvature)
        "V": "#2a78d6",  # blue (fluctuation)
        "R": "#8d99ae",  # slate grey (remainder)
    }

    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.15))

    bar_width = 0.36
    group_spacing = 1.0

    for ax, site in zip(axes, sites):
        style_axes(ax)
        ax.grid(True, axis="y", color=GRID, linewidth=0.7, zorder=0)
        ax.axhline(0, color=INK, linewidth=0.8, zorder=1)

        x_ticks = []
        x_ticklabels = []

        for i_t, t in enumerate(bits):
            center = i_t * group_spacing
            for i_m, mode in enumerate(("rn", "sr")):
                x = center + (i_m - 0.5) * bar_width * 1.08
                x_ticks.append(x)
                x_ticklabels.append(mode.upper())

                row = cells[site][mode, t]
                vals = {k: num(row, k) for k in TERMS}
                dl = num(row, "dL")
                se_dl = float(row.get("se_dL", 0.0)) if mode == "sr" else 0.0

                pos_bottom = 0.0
                neg_bottom = 0.0

                for k in TERMS:
                    v = vals[k]
                    if v >= 0:
                        ax.bar(x, v, bottom=pos_bottom, width=bar_width,
                               color=term_colors[k], edgecolor="white", linewidth=0.6, zorder=3)
                        pos_bottom += v
                    else:
                        ax.bar(x, v, bottom=neg_bottom, width=bar_width,
                               color=term_colors[k], edgecolor="white", linewidth=0.6, zorder=3)
                        neg_bottom += v

                # Net dL marker
                ax.plot([x - bar_width * 0.45, x + bar_width * 0.45], [dl, dl],
                        color=INK, linewidth=1.8, zorder=4)
                if mode == "sr" and se_dl > 0:
                    ax.errorbar(x, dl, yerr=se_dl, color=INK, capsize=2.2,
                                elinewidth=0.9, capthick=0.9, zorder=5)

                if summary:
                    print(f"{site} {mode} t={t}: D={vals['D']:.4f}, Q={vals['Q']:.4f}, "
                          f"V={vals['V']:.4f}, R={vals['R']:.4f}, dL={dl:.4f}")

            # Group label centered under the pair of bars
            ax.text(center, -0.13, f"$t={t}$", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7.5, color=INK)

        site_title = f"(a) {site_label[site]}" if site == "lm_head" else f"(b) {site_label[site]}"
        ax.set_title(site_title, fontsize=8.5, color=INK, loc="left")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels, fontsize=7.5, color=MUTED)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylabel("nats", fontsize=8)
        ax.tick_params(axis="x", length=2)

    # Align y=0 at 3/13 of axis height across both panels
    axes[0].set_ylim(-1.50, 5.00)
    axes[1].set_ylim(-0.12, 0.40)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=term_colors["D"], edgecolor="none", label=r"Drift $D$"),
        Patch(facecolor=term_colors["Q"], edgecolor="none", label=r"Curvature $Q$"),
        Patch(facecolor=term_colors["V"], edgecolor="none", label=r"Variance $V$"),
        Patch(facecolor=term_colors["R"], edgecolor="none", label=r"Remainder $R$"),
        Line2D([0], [0], color=INK, linewidth=1.8, label=r"Net $\delta\mathcal{L}$"),
    ]

    fig.legend(handles=legend_elements, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.00), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = out / "site_decomposition_terms.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    if summary:
        print(f"Wrote {path.name} and .png")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    setup()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_terms(args.out, args.summary)


if __name__ == "__main__":
    main()
