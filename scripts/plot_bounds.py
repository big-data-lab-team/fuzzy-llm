#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "matplotlib>=3.8,<4", "tabulate"]
# ///
"""
Comparison of Round-to-Nearest (RN) and Stochastic Rounding (SR) Error Bounds
for floating-point accumulation / reductions (e.g. inner products).

Supports both:
1. Precision Scaling Analysis (fixed dimension n, sweeping significand precision t)
2. Dimension Scaling Analysis (fixed significand precision t, sweeping dimension n)

The main-text convention is the relative ULP scale u = 2^(1-t). Thus t=24
for binary32 gives u=2^-23, and t=53 for binary64 gives u=2^-52.

Exports high-quality vector PDF figures by default.

    uv run scripts/plot_bounds.py --mode precision   # figures/rn_vs_sr_bounds.pdf
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# SOTA Table Rendering Imports
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


class PrecisionBoundAnalysis:
    """
    Analyzes and visualizes deterministic RN error bounds vs.
    probabilistic SR concentration error bounds parameterized by
    tail failure probability alpha across significand precision values t.

    The SR bound uses the per-step bound:
    Var(delta_k) <= u^2 (Theorem 4.1 in the paper:
    (1 + u^2)^{n+1} - 1).
    """

    def __init__(
        self,
        n: int = 3072,
        lambda_param: float = 0.05,
        p_min: float = 4.0,
        p_max: float = 16.0,
        num_points: int = 500,
    ):
        self.n = n
        self.lambda_param = lambda_param
        self.sr_factor = np.sqrt(1.0 / self.lambda_param)
        self.p_min = p_min
        self.p_max = p_max
        self.num_points = num_points

        self.p_values = np.linspace(p_min, p_max, num_points)
        self.u_values = 2.0 ** (1.0 - self.p_values)

        self.p_ints = np.arange(int(np.floor(p_min)), int(np.ceil(p_max)) + 1)
        self.u_ints = 2.0 ** (1.0 - self.p_ints.astype(float))

    def rn_bound(self, u: np.ndarray | float) -> np.ndarray | float:
        """Deterministic RN error bound: (1 + u/2)^(n+1) - 1"""
        return (1.0 + u / 2.0) ** (self.n + 1) - 1.0

    def sr_bound(self, u: np.ndarray | float) -> np.ndarray | float:
        """Probabilistic SR concentration bound: sqrt(1/alpha) * sqrt((1 + u^2)^(n+1) - 1)"""
        inner = (1.0 + u**2) ** (self.n + 1) - 1.0
        return self.sr_factor * np.sqrt(np.maximum(0.0, inner))

    def lower_bound(self, u: np.ndarray | float) -> np.ndarray | float:
        """Simple Lower bound on ratio RN / SR: sqrt(alpha) * sqrt(RN(u, n))"""
        rn = self.rn_bound(u)
        return np.sqrt(self.lambda_param) * np.sqrt(np.maximum(0.0, rn))

    def upper_bound(self, u: np.ndarray | float) -> np.ndarray | float:
        """Upper bound on ratio RN / SR: sqrt(alpha) * RN(u, n) / (sqrt(n + 1) * u)"""
        rn = self.rn_bound(u)
        return (np.sqrt(self.lambda_param) * rn) / (np.sqrt(self.n + 1) * u)

    def get_asymptotes(self, u: np.ndarray | float) -> Tuple[np.ndarray | float, np.ndarray | float]:
        """First-order asymptotes for high precision (u << 1/n)"""
        rn_asymp = ((self.n + 1) / 2.0) * u
        sr_asymp = self.sr_factor * np.sqrt(self.n + 1) * u
        return rn_asymp, sr_asymp

    def get_records(self, precisions: Optional[List[int]] = None) -> List[dict]:
        if precisions is None:
            precisions = list(self.p_ints)

        records = []
        for p in precisions:
            u = 2.0 ** (1.0 - p)
            rn = float(self.rn_bound(u))
            sr = float(self.sr_bound(u))
            ratio = float(rn / sr)
            lb = float(self.lower_bound(u))
            ub = float(self.upper_bound(u))
            records.append({
                "p": p,
                "u_str": f"2^(1-{p})",
                "u_val": u,
                "rn": rn,
                "sr": sr,
                "lower_bound": lb,
                "exact_ratio": ratio,
                "upper_bound": ub,
            })
        return records

    def print_table(self, precisions: Optional[List[int]] = None, engine: str = "auto", tablefmt: str = "fancy_grid"):
        records = self.get_records(precisions)
        headers = [
            "Significand precision t", "Relative ULP scale (u)", "RN Bound",
            f"SR Bound (α={self.lambda_param})", "Lower Bound (√α·√RN)",
            "Exact Ratio (RN/SR)", "Upper Bound"
        ]
        rows = [
            [
                f"{r['p']} bits",
                r["u_str"],
                f"{r['rn']:.4e}",
                f"{r['sr']:.4e}",
                f"{r['lower_bound']:.4e}",
                f"{r['exact_ratio']:14.4e}x",
                f"{r['upper_bound']:.4e}",
            ]
            for r in records
        ]

        if engine == "rich" or (engine == "auto" and HAS_RICH and tablefmt == "rich"):
            console = Console(width=160)
            table = Table(
                title=f"\nRN vs SR Error Bounds & Ratio Analysis (Fixed Dimension n = {self.n}, α = {self.lambda_param})\n",
                box=box.ROUNDED,
                header_style="bold cyan",
                border_style="bright_blue",
            )
            for h in headers:
                table.add_column(h, justify="right")
            for row in rows:
                table.add_row(*row)
            console.print(table)
        elif (engine == "auto" or engine == "tabulate") and HAS_TABULATE:
            print(f"\nRN vs SR Error Bounds & Ratio Analysis (Fixed Dimension n = {self.n}, α = {self.lambda_param})")
            print(tabulate(rows, headers=headers, tablefmt=tablefmt, stralign="right", numalign="right"))

    def plot_bounds(
        self,
        save_path: str = "figures/rn_vs_sr_bounds.pdf",
        ymax: Optional[float] = None,
        ymin: Optional[float] = None,
        show: bool = False,
    ):
        rn_vals = self.rn_bound(self.u_values)
        sr_vals = self.sr_bound(self.u_values)
        rn_ints = self.rn_bound(self.u_ints)
        sr_ints = self.sr_bound(self.u_ints)

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=300)

        # Plot main curves with markers
        ax.plot(self.p_values, rn_vals, color="#d62728", lw=2.0, label=r"$\mathrm{RN}(u,n) = (1 + u/2)^{n+1} - 1$")
        ax.plot(self.p_ints, rn_ints, "o", color="#d62728", markersize=5.0, markeredgecolor="white", markeredgewidth=0.7)

        ax.plot(
            self.p_values,
            sr_vals,
            color="#1f77b4",
            lw=2.0,
            label=r"$\mathrm{SR}(u,n) = \sqrt{1/\alpha}\,\sqrt{(1 + u^2)^{n+1} - 1}$",
        )
        ax.plot(self.p_ints, sr_ints, "s", color="#1f77b4", markersize=5.0, markeredgecolor="white", markeredgewidth=0.7)

        ax.set_yscale("log")
        ax.set_xlabel(r"Significand precision $t$ (bits, where $u = 2^{1-t}$)", fontsize=11)
        ax.set_ylabel("Bound Value (log scale)", fontsize=11)
        ax.set_title(f"RN vs SR Error Bounds ($n = {self.n}$, $\\alpha = {self.lambda_param}$)", fontsize=11.5, fontweight="bold", pad=8)

        if ymin is None:
            ymin = 1e-6
        if ymax is None:
            ymax = 1e16
        ax.set_ylim(bottom=ymin, top=ymax)

        step = max(1, int(round((self.p_max - self.p_min) / 10)))
        ax.set_xticks(np.arange(int(self.p_min), int(self.p_max) + 1, step))
        ax.set_xlim(self.p_min, self.p_max)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(fontsize=9.5, loc="upper right", frameon=True)

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[+] Bounds figure saved to: {save_path}")

        if show:
            plt.show()
        plt.close(fig)

    def plot_ratio(
        self,
        save_path: str = "figures/rn_vs_sr_ratio.pdf",
        ymax: Optional[float] = None,
        ymin: Optional[float] = None,
        show: bool = False,
    ):
        rn_vals = self.rn_bound(self.u_values)
        sr_vals = self.sr_bound(self.u_values)
        ratios = rn_vals / sr_vals
        lb_vals = self.lower_bound(self.u_values)
        ub_vals = self.upper_bound(self.u_values)

        rn_ints = self.rn_bound(self.u_ints)
        sr_ints = self.sr_bound(self.u_ints)
        ratios_ints = rn_ints / sr_ints
        lb_ints = self.lower_bound(self.u_ints)
        ub_ints = self.upper_bound(self.u_ints)

        asymp_ratio = np.sqrt(self.n + 1) / (2.0 * self.sr_factor)

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=300)

        ax.plot(
            self.p_values,
            ub_vals,
            "--",
            color="#ff7f0e",
            lw=2.0,
            label=r"Upper Bound: $\frac{\sqrt{\alpha}\,\mathrm{RN}}{\sqrt{n+1}\,u}$",
        )
        ax.plot(self.p_ints, ub_ints, "v", color="#ff7f0e", markersize=5.5, markeredgecolor="white", markeredgewidth=0.7)

        ax.plot(
            self.p_values,
            ratios,
            color="#2ca02c",
            lw=2.2,
            label=r"Exact Ratio: $\mathrm{RN} / \mathrm{SR}$",
        )
        ax.plot(self.p_ints, ratios_ints, "o", color="#2ca02c", markersize=5.5, markeredgecolor="white", markeredgewidth=0.7)

        ax.plot(
            self.p_values,
            lb_vals,
            "--",
            color="#e377c2",
            lw=2.0,
            label=r"Lower Bound: $\sqrt{\alpha}\,\sqrt{\mathrm{RN}}$",
        )
        ax.plot(self.p_ints, lb_ints, "^", color="#e377c2", markersize=5.5, markeredgecolor="white", markeredgewidth=0.7)

        ax.axhline(y=asymp_ratio, color="black", linestyle=":", lw=1.8, label=f"Asymptotic Ratio $\\approx {asymp_ratio:.2f}\\times$")

        ax.set_yscale("log")
        ax.set_xlabel(r"Significand precision $t$ (bits, where $u = 2^{1-t}$)", fontsize=12)
        ax.set_ylabel(r"Ratio $\mathrm{RN} / \mathrm{SR}$ (log scale)", fontsize=12)
        ax.set_title(f"Advantage Ratio and Bounds ($n = {self.n}$, $\\alpha = {self.lambda_param}$)", fontsize=13, fontweight="bold", pad=10)

        if ymax is not None or ymin is not None:
            ax.set_ylim(bottom=ymin, top=ymax)

        step = max(1, int(round((self.p_max - self.p_min) / 10)))
        ax.set_xticks(np.arange(int(self.p_min), int(self.p_max) + 1, step))
        ax.set_xlim(self.p_min, self.p_max)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(fontsize=9.5, loc="upper right", frameon=True)

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[+] Ratio figure saved to: {save_path}")

        if show:
            plt.show()
        plt.close(fig)

    def plot(
        self,
        save_path_bounds: str = "figures/rn_vs_sr_bounds.pdf",
        save_path_ratio: str = "figures/rn_vs_sr_ratio.pdf",
        ymax: Optional[float] = None,
        ymin: Optional[float] = None,
        ymax_bounds: Optional[float] = None,
        ymax_ratio: Optional[float] = None,
        ymin_bounds: Optional[float] = None,
        ymin_ratio: Optional[float] = None,
        show: bool = False,
    ):
        self.plot_bounds(
            save_path=save_path_bounds,
            ymax=ymax_bounds if ymax_bounds is not None else ymax,
            ymin=ymin_bounds if ymin_bounds is not None else ymin,
            show=show,
        )
        self.plot_ratio(
            save_path=save_path_ratio,
            ymax=ymax_ratio if ymax_ratio is not None else ymax,
            ymin=ymin_ratio if ymin_ratio is not None else ymin,
            show=show,
        )


class DimensionScalingAnalysis:
    """
    Analyzes and visualizes RN vs SR error bounds and advantage ratio
    as dimension n scales (e.g., from n=10 to n=10^5) for several fixed precisions.
    """

    def __init__(
        self,
        precisions: Optional[List[int]] = None,
        lambda_param: float = 0.05,
        n_min: int = 10,
        n_max: int = 100000,
        num_points: int = 500,
    ):
        self.precisions = precisions if precisions is not None else [4, 8, 10, 16, 24]
        self.lambda_param = lambda_param
        self.sr_factor = np.sqrt(1.0 / self.lambda_param)
        self.n_min = n_min
        self.n_max = n_max
        self.num_points = num_points

        self.n_values = np.logspace(np.log10(n_min), np.log10(n_max), num_points)

    def log10_rn_bound(self, n: np.ndarray | float, u: float) -> np.ndarray | float:
        """Computes log10(RN bound) robustly across all n and u."""
        log_term = (n + 1.0) * np.log1p(u / 2.0)
        if isinstance(log_term, np.ndarray):
            result = np.empty_like(log_term)
            small = log_term < 1e-4
            mid = (log_term >= 1e-4) & (log_term <= 700.0)
            large = log_term > 700.0
            result[small] = np.log10((n[small] + 1.0) * (u / 2.0))
            result[mid] = np.log10(np.expm1(log_term[mid]))
            result[large] = log_term[large] / np.log(10.0)
            return result
        else:
            if log_term < 1e-4:
                return np.log10((n + 1.0) * (u / 2.0))
            elif log_term > 700.0:
                return log_term / np.log(10.0)
            else:
                return np.log10(np.expm1(log_term))

    def log10_sr_bound(self, n: np.ndarray | float, u: float) -> np.ndarray | float:
        """Computes log10(SR bound) robustly across all n and u."""
        log_term = (n + 1.0) * np.log1p(u**2)
        if isinstance(log_term, np.ndarray):
            result = np.empty_like(log_term)
            small = log_term < 1e-4
            mid = (log_term >= 1e-4) & (log_term <= 700.0)
            large = log_term > 700.0
            result[small] = np.log10(self.sr_factor) + 0.5 * np.log10((n[small] + 1.0) * (u**2))
            result[mid] = np.log10(self.sr_factor) + 0.5 * np.log10(np.expm1(log_term[mid]))
            result[large] = np.log10(self.sr_factor) + 0.5 * (log_term[large] / np.log(10.0))
            return result
        else:
            if log_term < 1e-4:
                return np.log10(self.sr_factor) + 0.5 * np.log10((n + 1.0) * (u**2))
            elif log_term > 700.0:
                return np.log10(self.sr_factor) + 0.5 * (log_term / np.log(10.0))
            else:
                return np.log10(self.sr_factor) + 0.5 * np.log10(np.expm1(log_term))

    def log10_ratio(self, n: np.ndarray | float, u: float) -> np.ndarray | float:
        return self.log10_rn_bound(n, u) - self.log10_sr_bound(n, u)

    def plot_bounds_vs_n(
        self,
        save_path: str = "figures/rn_vs_sr_bounds_vs_n.pdf",
        ymax: Optional[float] = None,
        ymin: Optional[float] = None,
        show: bool = False,
    ):
        """Plots RN vs SR Error Bounds as a function of dimension n for fixed precisions."""
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=300)

        palette = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#1f77b4", "#8c564b"]

        for i, p in enumerate(self.precisions):
            u = 2.0 ** (1.0 - p)
            log_rn = self.log10_rn_bound(self.n_values, u)
            log_sr = self.log10_sr_bound(self.n_values, u)
            color = palette[i % len(palette)]

            rn_plot = 10.0 ** np.clip(log_rn, -30, 20)
            sr_plot = 10.0 ** np.clip(log_sr, -30, 20)

            ax.plot(self.n_values, rn_plot, "-", color=color, lw=2.2, label=f"RN: $t={p}$ bits")
            ax.plot(self.n_values, sr_plot, "--", color=color, lw=1.8, alpha=0.85, label=f"SR: $t={p}$ bits")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Dimension $n$ (log scale)", fontsize=12)
        ax.set_ylabel("Error Bound Value (log scale)", fontsize=12)
        ax.set_title(rf"RN vs SR Error Bounds vs Dimension $n$ ($\lambda = {self.lambda_param}$)", fontsize=13, fontweight="bold", pad=10)

        if ymin is None:
            ymin = 1e-6
        if ymax is None:
            ymax = 1e12
        ax.set_ylim(bottom=ymin, top=ymax)

        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[+] Dimension bounds figure saved to: {save_path}")

        if show:
            plt.show()
        plt.close(fig)

    def plot_ratio_vs_n(
        self,
        save_path: str = "figures/rn_vs_sr_ratio_vs_n.pdf",
        ymax: Optional[float] = None,
        ymin: Optional[float] = None,
        show: bool = False,
    ):
        """Plots Advantage Ratio RN / SR as a function of dimension n for fixed precisions."""
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=300)

        palette = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#1f77b4", "#8c564b"]

        for i, p in enumerate(self.precisions):
            u = 2.0 ** (1.0 - p)
            log_ratio = self.log10_ratio(self.n_values, u)
            color = palette[i % len(palette)]

            ratio_plot = 10.0 ** np.clip(log_ratio, -10, 25)
            ax.plot(self.n_values, ratio_plot, "-", color=color, lw=2.2, label=f"$t={p}$ bits, $u=2^{{1-{p}}}$")

        asymp_ref = np.sqrt(self.lambda_param * (self.n_values + 1.0)) / 2.0
        ax.plot(self.n_values, asymp_ref, ":", color="black", lw=2.0, label=r"High-Precision: $\frac{\sqrt{\lambda(n+1)}}{2} \propto \sqrt{n}$")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Dimension $n$ (log scale)", fontsize=12)
        ax.set_ylabel(r"Advantage Ratio $\mathrm{RN} / \mathrm{SR}$ (log scale)", fontsize=12)
        ax.set_title(rf"Advantage Ratio $\mathrm{{RN}} / \mathrm{{SR}}$ vs Dimension $n$ ($\lambda = {self.lambda_param}$)", fontsize=13, fontweight="bold", pad=10)

        if ymin is None:
            ymin = 1e-1
        if ymax is None:
            ymax = 1e12
        ax.set_ylim(bottom=ymin, top=ymax)

        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[+] Dimension ratio figure saved to: {save_path}")

        if show:
            plt.show()
        plt.close(fig)

    def print_table(self, sample_dimensions: Optional[List[int]] = None, tablefmt: str = "fancy_grid"):
        """Prints formatted table showing ratio across dimensions for key precisions."""
        if sample_dimensions is None:
            sample_dimensions = [128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

        headers = ["Dimension n"] + [f"t={p} (u=2^(1-{p}))" for p in self.precisions]
        rows = []
        for n in sample_dimensions:
            row = [f"n = {n:,}"]
            for p in self.precisions:
                u = 2.0 ** (1.0 - p)
                log10_r = float(self.log10_ratio(n, u))
                if log10_r < 100:
                    val = 10.0**log10_r
                    row.append(f"{val:11.3e}x")
                else:
                    row.append(f"10^{log10_r:.1f}x")
            rows.append(row)

        if HAS_TABULATE:
            print(f"\nAdvantage Ratio RN / SR across Scaling Dimensions n (λ = {self.lambda_param})")
            print(tabulate(rows, headers=headers, tablefmt=tablefmt, stralign="right", numalign="right"))


def main():
    parser = argparse.ArgumentParser(
        description="Compare Round-to-Nearest (RN) and Stochastic Rounding (SR) error bounds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "precision", "dimension"],
        help="Analysis mode: 'precision' (sweep t), 'dimension' (sweep n), or 'all'.",
    )
    parser.add_argument(
        "--tmin", "--pmin", "-pmin",
        dest="pmin",
        type=float,
        default=4.0,
        help="Minimum significand precision t in bits for precision sweep.",
    )
    parser.add_argument(
        "--tmax", "--pmax", "-pmax",
        dest="pmax",
        type=float,
        default=16.0,
        help="Maximum significand precision t in bits for precision sweep.",
    )
    parser.add_argument(
        "--lambda", "--lambda-param", "-l",
        dest="lambda_param",
        type=float,
        default=0.05,
        help="Tail failure probability lambda in (0, 1].",
    )
    parser.add_argument(
        "-n", "--dim",
        type=int,
        default=3072,
        help="Fixed dimension n for precision sweep.",
    )
    parser.add_argument(
        "--nmin",
        type=int,
        default=10,
        help="Minimum dimension n for dimension sweep.",
    )
    parser.add_argument(
        "--nmax",
        type=int,
        default=100000,
        help="Maximum dimension n for dimension sweep.",
    )
    parser.add_argument(
        "--fixed-t", "--fixed-p",
        dest="fixed_p",
        type=int,
        nargs="+",
        default=[4, 8, 11, 16, 24],
        help="Fixed significand precision values t to plot in dimension sweep.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Global maximum y-axis limit.",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Global minimum y-axis limit.",
    )
    parser.add_argument(
        "--output-bounds", "-ob",
        type=str,
        default="figures/rn_vs_sr_bounds.pdf",
        help="Path to save precision bounds figure.",
    )
    parser.add_argument(
        "--output-ratio", "-or",
        type=str,
        default="figures/rn_vs_sr_ratio.pdf",
        help="Path to save precision ratio figure.",
    )
    parser.add_argument(
        "--output-dim-bounds",
        type=str,
        default="figures/rn_vs_sr_bounds_vs_n.pdf",
        help="Path to save dimension bounds figure.",
    )
    parser.add_argument(
        "--output-dim-ratio",
        type=str,
        default="figures/rn_vs_sr_ratio_vs_n.pdf",
        help="Path to save dimension ratio figure.",
    )
    parser.add_argument(
        "--tablefmt",
        type=str,
        default="fancy_grid",
        choices=["fancy_grid", "github", "latex", "pipe", "simple", "rich"],
        help="Format for table rendering.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip figure generation and only print table.",
    )

    args = parser.parse_args()

    # 1. Precision Sweep Mode
    if args.mode in ["all", "precision"]:
        p_analysis = PrecisionBoundAnalysis(
            n=args.dim,
            lambda_param=args.lambda_param,
            p_min=args.pmin,
            p_max=args.pmax,
        )
        p_analysis.print_table(tablefmt=args.tablefmt)
        if not args.no_plot:
            p_analysis.plot(
                save_path_bounds=args.output_bounds,
                save_path_ratio=args.output_ratio,
                ymax=args.ymax,
                ymin=args.ymin,
                show=False,
            )

    # 2. Dimension Sweep Mode
    if args.mode in ["all", "dimension"]:
        d_analysis = DimensionScalingAnalysis(
            precisions=args.fixed_p,
            lambda_param=args.lambda_param,
            n_min=args.nmin,
            n_max=args.nmax,
        )
        d_analysis.print_table(tablefmt=args.tablefmt)
        if not args.no_plot:
            d_analysis.plot_bounds_vs_n(
                save_path=args.output_dim_bounds,
                ymax=args.ymax,
                ymin=args.ymin,
                show=False,
            )
            d_analysis.plot_ratio_vs_n(
                save_path=args.output_dim_ratio,
                ymax=args.ymax,
                ymin=args.ymin,
                show=False,
            )


if __name__ == "__main__":
    main()
