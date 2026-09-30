#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Recompute the composition statistics \\cref{tab:mixed} and its prose quote.

    uv run scripts/composition_stats.py            # the numbers the section quotes
    uv run scripts/composition_stats.py --check    # also verify the t machinery

The mixed-recipe section makes two comparisons and the paper quotes a test
statistic for each, so this script exists to make those quotes reproducible from
the record file rather than recomputed by hand:

* the **block rule**, ``rn-blocks`` against ``sr-blocks``. RN is a measured
  constant at the recorded precision, so this is a one-sample interval
  around the SR mean, the same test :func:`make_figures.verdict` applies.
* the **head rule**, ``sr-blocks`` against ``sr-all``. Runs share seed identifiers,
  so this uses paired differences, as do the comparisons with ``mixed``.

Critical values are computed at the achieved degrees of freedom rather than
taken from a constant. That matters here: the recipes were first run at five
seeds and later topped up, and a test that hard-codes ``df=4`` silently reports
the wrong threshold the moment the seed count changes.

Everything is stdlib. The Student-t tail is the standard incomplete-beta
identity, and ``--check`` verifies it reproduces the df=4 value the figures use.
"""

import argparse
import csv
import statistics as st
import sys
from math import exp, inf, isfinite, lgamma, log, log1p, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECORDS = ROOT / "results" / "sweep_records.csv"

#: Table order, and the labels the paper uses for the four recipes.
RECIPES = {
    "pure_sr_rnhead": "sr-blocks",
    "pure_sr": "sr-all",
    "mixed": "mixed",
    "pure_rn": "rn-blocks",
}


# ------------------------------------------------------------ Student-t tail
def _betacf(a, b, x, itmax=200, eps=3e-16):
    """Continued fraction for the incomplete beta (Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        for num in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                    -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + num * d
            if abs(d) < 1e-300:
                d = 1e-300
            c = 1.0 + num / c
            if abs(c) < 1e-300:
                c = 1e-300
            d = 1.0 / d
            h *= d * c
        if abs(d * c - 1.0) < eps:
            break
    return h


def betai(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (lgamma(a + b) - lgamma(a) - lgamma(b)
             + a * log(x) + b * log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """Two-sided tail probability of Student-t."""
    if df <= 0 or not isfinite(t):
        return float("nan")
    return betai(df / 2.0, 0.5, df / (df + t * t))


def t_crit(df, alpha=0.05):
    """Two-sided critical value, by bisection on the tail."""
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_sf(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# --------------------------------------------------------------------- data
def load(path):
    """Return {recipe: {seed: perplexity}} for the scoped mixed level."""
    by = {}
    with Path(path).open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["level"] != "mixed" or r["arm"] != "scoped":
                continue
            values = by.setdefault(r["target"], {})
            seed = int(r["seed"])
            if seed in values:
                raise ValueError(f"duplicate seed {seed} for {r['target']}")
            values[seed] = float(r["perplexity"])
    return by


def sanity(by):
    """Reject invalid perplexities without selecting on the observed ranking.

    Perplexity can improve on the reference or exceed an RN control. Neither
    comparison is a validity bound or a reason to exclude a recorded seed.
    """
    bad = {}
    for name, values in by.items():
        flagged = [v for v in values
                   if not isfinite(v) or v < 1.0]
        if flagged:
            bad[name] = flagged
    return bad


def describe(values):
    n = len(values)
    mean = st.fmean(values)
    sd = st.stdev(values) if n > 1 else 0.0
    return mean, sd, n


def one_sample(const, values, alpha=0.05):
    """Return stochastic-arm mean minus constant, CI half-width, and resolution."""
    mean, sd, n = describe(values)
    half = t_crit(n - 1, alpha) * sd / sqrt(n) if n > 1 else 0.0
    delta = mean - const
    return delta, half, abs(delta) > half


def paired(a, b, alpha=0.05):
    """Compare seed-indexed arms, requiring complete, unambiguous pairs."""
    if set(a) != set(b) or len(a) < 2:
        raise ValueError("paired arms must have identical seed sets and n >= 2")
    differences = [a[seed] - b[seed] for seed in sorted(a)]
    mean, sd, n = describe(differences)
    se = sd / sqrt(n)
    df = n - 1
    half = t_crit(df, alpha) * se
    if se == 0.0:
        statistic = 0.0 if mean == 0.0 else (inf if mean > 0 else -inf)
        p = 1.0 if mean == 0.0 else 0.0
    else:
        statistic = mean / se
        p = t_sf(statistic, df)
    return mean, statistic, df, half, p, p < alpha


def welch(a, b, alpha=0.05):
    """Two-sample Welch comparison of two stochastic arms."""
    ma, sa, na = describe(a)
    mb, sb, nb = describe(b)
    va, vb = sa * sa / na, sb * sb / nb
    se = sqrt(va + vb)
    if se == 0.0:
        return ma - mb, inf, inf, True
    t = (ma - mb) / se
    df = (va + vb) ** 2 / (va * va / (na - 1) + vb * vb / (nb - 1))
    return ma - mb, t, df, abs(t) > t_crit(df, alpha)


def detectable(a, b, alpha=0.05, power=0.80):
    """Smallest difference this design could resolve, in perplexity.

    Reported when a comparison does not resolve, so that "not separated" can be
    read as a bound rather than as an absence of information.
    """
    _, sa, na = describe(a)
    _, sb, nb = describe(b)
    se = sqrt(sa * sa / na + sb * sb / nb)
    # z-approximation is adequate at these degrees of freedom.
    return (1.959964 + 0.841621) * se


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    ap.add_argument("--check", action="store_true",
                    help="verify the t machinery against known values")
    args = ap.parse_args()

    if args.check:
        for df, want in ((4, 2.776), (9, 2.262), (30, 2.042), (120, 1.980)):
            got = t_crit(df)
            flag = "ok" if abs(got - want) < 5e-3 else "MISMATCH"
            print(f"  t_crit(df={df:3d}) = {got:.4f}  expected {want:.3f}  {flag}")
        print()

    by_seed = load(args.records)
    if not by_seed:
        print("no mixed-level records found", file=sys.stderr)
        return 1
    by = {name: list(values.values()) for name, values in by_seed.items()}

    bad = sanity(by)
    if bad:
        print("INVALID VALUES -- analysis stopped; no seeds excluded:")
        for name, values in bad.items():
            print(f"  {RECIPES.get(name, name)}: {values}")
        return 1

    print("=== recipes ===")
    for key, label in RECIPES.items():
        if key not in by:
            continue
        mean, sd, n = describe(by[key])
        spread = f"+/-{sd:.2f}" if sd else "  (identical recorded PPL)"
        print(f"  {label:<10s} n={n:<3d} {mean:8.2f} {spread}")

    print("\n=== block rule: rn-blocks -> sr-blocks ===")
    delta, half, resolved = one_sample(st.fmean(by["pure_rn"]),
                                       by["pure_sr_rnhead"])
    print(f"  SR - RN gap {delta:.2f}   one-sample half-width {half:.2f}   "
          f"{'RESOLVED' if resolved else 'not resolved'}")

    print("\n=== head rule: sr-blocks -> sr-all ===")
    diff, t, df, half, p, resolved = paired(by_seed["pure_sr"], by_seed["pure_sr_rnhead"])
    print(f"  cost {diff:.2f}   paired t={t:.2f}   df={df}   "
          f"95% CI [{diff-half:.2f}, {diff+half:.2f}]   p={p:.6g}   "
          f"{'RESOLVED' if resolved else 'not resolved'}")

    print("\n=== mixed against the two SR recipes ===")
    for other, label in (("pure_sr_rnhead", "sr-blocks"), ("pure_sr", "sr-all")):
        diff, t, df, half, p, resolved = paired(by_seed["mixed"], by_seed[other])
        print(f"  mixed - {label:<10s} {diff:7.2f}   paired t={t:6.2f}   df={df}   "
              f"95% CI [{diff-half:.2f}, {diff+half:.2f}]   p={p:.6g}   "
              f"{'RESOLVED' if resolved else 'not resolved'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
