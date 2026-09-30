#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8,<4"]
# ///
"""Plot sample-drift marginals and SR fluctuation scaling from saved captures.

The drift panels show histogram-interpolated interquartile ranges and medians
on a common vertical scale, without duplicating the signed-loss term shown in
the decomposition figures. They describe sample means, including seed noise;
the appendix table retains RMS, extrema and out-of-range mass.

The fluctuation panels show normalized coordinate noise and its loss penalty,
with one jackknife standard error. Dashed fourfold-per-bit guides illustrate
constant-coefficient u^2 scaling; they are not fitted or theoretical bounds.

The drift panels also mark the SR sample mean over token-coordinate pairs
(a hollow diamond) with one jackknife standard error, distinct from the
signed median the boxes already show. --summary additionally prints a
two-sided Student-t test of that mean against zero (delete-one-seed
jackknife SE, S-1 degrees of freedom) -- the population-level unbiasedness
question the median cannot answer on its own.

Run: uv run scripts/plot_drift_fluctuation.py --summary
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from composition_stats import t_crit, t_sf  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO / "results" / "distributions_s32.csv"
DEFAULT_OUT = REPO / "figures"

# Categorical slots 1 and 2 of the validated palette, as in make_figures.py.
SR_COLOR = "#2a78d6"
RN_COLOR = "#eb6834"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"

STYLE = {
    "sr": dict(color=SR_COLOR, marker="o", linestyle="-", label="SR"),
    "rn": dict(color=RN_COLOR, marker="s", linestyle="--", label="RN"),
}
SITES = ("lm_head", "mlp_c_proj")
SITE_LABEL = {"lm_head": r"\code{lm\_head}", "mlp_c_proj": r"MLP down-projection"}
SITE_TITLE = {"lm_head": "Language-model head", "mlp_c_proj": "MLP down-projection"}
# Second encoding for the site, used only where both series are the same mode and
# colour therefore cannot carry the site.
SITE_MARK = {"lm_head": dict(marker="o", linestyle="-"),
             "mlp_c_proj": dict(marker="^", linestyle=":")}
BITS = (6, 7, 8)


def style_axes(ax):
    """Recessive grid and axes; text in ink tokens, never a series colour."""
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


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
    })


def load(path):
    import csv
    if not path.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        try:
            out[(r["site"], int(r["t"]), r["mode"])] = r
        except (KeyError, ValueError):
            continue
    return out


def num(row, key):
    v = row.get(key, "") if row else ""
    return None if v in ("", None) else float(v)


# --------------------------------------------------------------------------
# figure 1: the drift distribution, and the projection the loss charges
# --------------------------------------------------------------------------

def fig_drift(rows, out, summary=False):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)
    half = 0.17          # box half-width in x units
    off = 0.20           # RN/SR offset, so the boxes never overlap
    sr_seeds = next((r["seeds"] for (_, _, m), r in rows.items()
                      if m == "sr" and r.get("seeds")), "8")

    for col, site in enumerate(SITES):
        ax = axes[col]
        style_axes(ax)
        ax.axhline(0.0, color=INK, linewidth=0.8, zorder=1)
        for mode in ("rn", "sr"):
            c = STYLE[mode]["color"]
            for t in BITS:
                r = rows.get((site, t, mode))
                if not r:
                    continue
                q25, q50, q75 = (num(r, "bn_q25"), num(r, "bn_q50"),
                                 num(r, "bn_q75"))
                if None in (q25, q50, q75):
                    continue
                x = t + (off if mode == "sr" else -off)
                # 2px surface gap between adjacent fills: the boxes are offset
                # and drawn with a surface-coloured edge.
                ax.add_patch(Rectangle((x - half, q25), 2 * half, q75 - q25,
                                       facecolor=c, alpha=0.30, edgecolor=c,
                                       linewidth=0.8, zorder=3))
                ax.plot([x - half, x + half], [q50, q50], color=c,
                        linewidth=1.8, solid_capstyle="butt", zorder=4)
                ax.plot([x], [q50], color=c, zorder=5, **{
                    k: v for k, v in STYLE[mode].items()
                    if k in ("marker",)}, markersize=4.5, linestyle="none")
                # sample mean over token-coordinate pairs, one jackknife
                # standard error; undefined for RN's single deterministic run.
                b_mean, se_b_mean, u = (num(r, "b_mean"), num(r, "se_b_mean"),
                                        num(r, "u_scale"))
                if None not in (b_mean, se_b_mean, u) and u:
                    ax.errorbar([x], [b_mean / u], yerr=[se_b_mean / u],
                                color=c, elinewidth=0.9, capsize=2.5,
                                capthick=0.9, marker="D", markersize=3.4,
                                markerfacecolor="white", markeredgecolor=c,
                                markeredgewidth=1.0, linestyle="none", zorder=6)
        ax.set_title(f"({'ab'[col]}) {SITE_TITLE[site]}", color=INK, loc="left")
        if col == 0:
            ax.set_ylabel(r"sample drift $\hat b_j/u_\lambda$")
        ax.set_xlabel("virtual precision $t$ (bits)")
        ax.set_xticks(list(BITS))
        ax.set_xlim(5.5, 8.5)
        ax.set_ylim(-1.7, 5.2)

    handles = [
        Line2D([], [], color=RN_COLOR, marker="s", linestyle="--", markersize=4.5,
               label="RN (1 deterministic run)"),
        Line2D([], [], color=SR_COLOR, marker="o", linestyle="-", markersize=4.5,
               label=f"SR ($S={sr_seeds}$ seeds)"),
        Line2D([], [], color=INK, marker="D", markersize=3.6,
               markerfacecolor="white", linestyle="none",
               label=r"SR mean $\pm$ 1 SE (jackknife)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=8)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    path = out / "drift_distribution.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    if summary:
        print(f"  wrote {path.name} and .png")
    return path


# --------------------------------------------------------------------------
# figure 2: the fluctuation does not scale like the ULP
# --------------------------------------------------------------------------

def fig_fluctuation(rows, out, summary=False):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9))
    for ax in axes:
        style_axes(ax)
        ax.set_xlabel("precision $t$ (bits)")
        ax.set_xticks(list(BITS))
        ax.set_xlim(5.85, 8.15)
    for site in SITES:
        xs = list(BITS)
        data = [rows[(site, t, "sr")] for t in xs]
        ys = [num(r, "sd_rms_ulp") for r in data]
        es = [num(r, "se_sd_rms") / num(r, "u_scale") for r in data]
        axes[0].errorbar(xs, ys, yerr=es, color=SR_COLOR, capsize=3,
                        label=SITE_TITLE[site], **SITE_MARK[site])
        vs = [num(r, "V") for r in data]
        axes[1].errorbar(xs, vs, yerr=[num(r, "se_V") for r in data],
                        color=SR_COLOR, capsize=3, **SITE_MARK[site])
        guide = [vs[0] * 4.0 ** -(t - xs[0]) for t in xs]
        axes[1].plot(xs, guide, color=MUTED, linewidth=1,
                     linestyle=(0, (4, 2)), zorder=2)
    axes[0].set_ylabel(r"$\mathrm{rms}(\hat\sigma)/u_\lambda$")
    axes[0].set_title("(a) Normalized fluctuation", loc="left")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"fluctuation penalty $\hat V$ (nats)")
    axes[1].set_title("(b) Loss penalty and $u^2$ guides", loc="left")
    handles = [Line2D([], [], color=SR_COLOR, label=SITE_TITLE[s],
                      **SITE_MARK[s]) for s in SITES]
    handles.append(Line2D([], [], color=MUTED, linestyle=(0, (4, 2)),
                          linewidth=1, label=r"$4\times$ decrease per bit"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0), fontsize=8)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    path = out / "fluctuation_scaling.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    if summary:
        print(f"  wrote {path.name} and .png")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    setup()
    rows = load(args.csv)
    if not rows:
        raise SystemExit(f"no rows in {args.csv}")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.summary:
        print(f"{len(rows)} cell(s) from {args.csv.name}")
    fig_drift(rows, args.out, args.summary)
    fig_fluctuation(rows, args.out, args.summary)
    if args.summary:
        for site in SITES:
            for t in BITS:
                r = rows.get((site, t, "sr"))
                if r:
                    print(f"  {site:11s} t={t} SR  median {num(r,'bn_q50'):+.4f}  "
                          f"IQR {num(r,'bn_q75')-num(r,'bn_q25'):.3f}  "
                          f"rms {num(r,'b_rms_ulp'):.3f}  "
                          f"sd/u {num(r,'sd_rms_ulp'):.3f}  V {num(r,'V'):.5f}")
        print("\nunbiasedness (H0: mean coordinate drift = 0), "
              "two-sided Student-t, delete-one-seed jackknife SE:")
        for site in SITES:
            for t in BITS:
                r = rows.get((site, t, "sr"))
                if not r:
                    continue
                b_mean, se_b_mean = num(r, "b_mean"), num(r, "se_b_mean")
                seeds = int(num(r, "seeds") or 0)
                if None in (b_mean, se_b_mean) or seeds < 2:
                    continue
                df = seeds - 1
                stat = b_mean / se_b_mean
                p = t_sf(stat, df)
                resolved = abs(stat) > t_crit(df)
                print(f"  {site:11s} t={t} SR  t={stat:+.3f}  df={df}  "
                      f"p={p:.3g}  {'RESOLVED' if resolved else 'not resolved'}")


if __name__ == "__main__":
    main()
