"""Empirical distribution of the logit perturbation and of the induced softmax
probabilities, per (site, t, mode), from the on-disk captures alone.

Section 4.2 asserts two things about the perturbation Delta that the harness has
never measured directly: that SR is operationally unbiased at the logits
(b_SR ~ 0, analysis_4_2.tex:170,211,229) and that RN's fluctuation vanishes
(eq:rn-degenerate). Both are statements about the *distribution* of Delta over
seeds, and `decompose.py` only ever forms the p-weighted Gram, which is enough
for the four aggregate terms and not enough for a bias or a variance per
coordinate. This script supplies those marginals. It runs no inference: every
number below is a reduction over `delta.npy` and `reference/logits.npy`.

Why the cross-check against decompose.py is exact, not statistical
-----------------------------------------------------------------
With H = diag(p) - p p^T,

    tr(H Sigma) = sum_j p_j Sigma_jj - p^T Sigma p,

and the second term needs no matrix: p is fixed at a token, so
p^T Sigma p = Var_s(p^T Delta), the seed variance of a scalar already stored in
scalars.npz as Ep_d. So with e_s := p^T Delta^s and sigma2_j the ddof=1
cross-seed variance,

    V_hat = 1/2 ( sum_j p_j sigma2_j - Var_s(e_s) )                        (*)

and (*) is *algebraically identical* to decompose.terms()'s
V = k(A - B)/(k - 1). Both reduce to the same expression through
mean_s x^2 - xbar^2 = ((k-1)/k) var_ddof1. Likewise

    D     = E_p[b_hat] - b_hat_y
    Q_hat = 1/2 Var_p(b_hat) - V_hat/S

exactly. The `D_decompose_abserr` / `Q_decompose_abserr` / `V_decompose_abserr`
columns are therefore floating-point equalities (~1e-12), not corroborations,
and a nonzero value there means this script is wrong. No V x V covariance is
ever formed and no Gram is ever built here.

Raw and p-centered marginals are both reported
----------------------------------------------
The loss is invariant to Delta -> Delta + c*1, so raw b and sigma2 mix a
loss-irrelevant common mode with the part that costs nats. The centered
perturbation Delta_tilde = Delta - (p^T Delta) 1 is the object the KL bound's
Gaussian model is stated on (appendix_loss.tex:78-83), and it satisfies
sum_j p_j Var_s(Delta_tilde_j) = 2 V exactly. Raw is primary for the drift and
fluctuation columns; centered is primary for the shape columns. `b_rms` beside
`bt_rms` shows how much of the drift is free.

Two traps this script is built around
-------------------------------------
1. sum_j (mean_s q^s_j - p_j) = 0 *identically*, because q^s and p are both
   normalized. A plain mean of the probability bias over classes is therefore a
   check and not a measurement -- it would read as "no bias". The reported
   aggregates are L1 and RMS, plus the target class. Moments of log q are
   p-weighted, never uniform: a uniform mean over 50257 classes is dominated by
   classes at log p ~ -30 that the loss never weights.
2. The per-coordinate standardization z^s_j = (Delta_tilde^s_j - b_tilde_j)/
   sigma_tilde_j obeys sum_s z_s^2 = S - 1 identically, hence

       |z| <= (S - 1)/sqrt(S)   =   2.47 at S = 8.

   Its histogram *cannot* show a tail past that and its kurtosis is bounded
   above whatever the truth is, so `z_exkurt ~ 0` must never be read as
   "Gaussian". The token-pooled w^s_j = (Delta_tilde^s_j - b_tilde_j)/sbar_n,
   whose denominator does not involve j, is not clipped and is the column that
   can actually falsify a Gaussian tail. `z_clip_bound` is carried in every row.

RN, and variance that is structurally absent rather than measured to be zero
---------------------------------------------------------------------------
`decompose.py` writes V = 0.0 for every RN row. That is right as theory and
misleading as data. Here an S=1 cell reports b = Delta exactly and leaves every
variance-derived column *blank*, never 0, with `variance_status` saying which
case it is. An RN cell with S > 1 is checked to be bitwise identical across
seeds and still refuses to report a fluctuation: a variance over replicas of a
deterministic map measures nothing.

Layout consumed (identical to decompose.py's)
    <root>/reference/                     logits.npy scalars.npz meta.json
    <root>/<site>/t<NN>/<mode>/seed<S>/    delta.npy  scalars.npz meta.json

Two harness contracts, both load-bearing
    * No flag here is named --out-dir. run_config.sh reacts to a task that
      produced no "Result ->" line by sed-ing --out-dir out of the argument
      string and rm-ing delta.npy/logits.npy/scalars.npz/meta.json inside it. An
      analysis task pointed at a capture cell that died would delete the
      captures it was reading. --cell-csv/--per-token-dir/--hist-dir are safe.
    * The Result line carries "Bits:" and *no* "Precision:", "Configuration:" or
      "Perplexity:" field. parse_sweep_logs.py:178 ingests only a line with
      Perplexity: and one of the other two, so withholding all three makes an
      accidental sweep_records.csv row structurally impossible rather than one
      regex away.
"""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
import warnings

import numpy as np

from decompose import (aggregate, cell_seeds, discover, load_p_chunk,
                       load_scalars, log_softmax_np, terms)

SCRIPT_VERSION = "distributions-2"

#: Per-coordinate standardized histogram. Bounded by (S-1)/sqrt(S) by
#: construction, so a wide range would be all empty bins; 0.01 is the quantile
#: resolution.
Z_EDGES = np.linspace(-4.0, 4.0, 801)
#: Token-pooled standardization is unbounded, so this one has to reach further.
W_EDGES = np.linspace(-12.0, 12.0, 1201)
QUANTILES = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)
#: Drift in units of one ULP at the representative logit magnitude. This is the
#: only shape product defined at S = 1, so it is what makes an RN row and an SR
#: row comparable; +-8 ulp covers the measured range with room for t = 4, 5.
BN_EDGES = np.linspace(-8.0, 8.0, 1601)
QNAMES = {0.001: "q001", 0.01: "q01", 0.05: "q05", 0.25: "q25", 0.5: "q50",
          0.75: "q75", 0.95: "q95", 0.99: "q99", 0.999: "q999"}


def ulp_scale(t, scales, which="rms"):
    """One ULP of the significand at the representative logit magnitude.

    ``which="rms"`` gives 2^(1-t) * rms(lambda), the primary normalisation;
    ``which="exact"`` gives 2^(1-t) * mean(2^floor(log2|lambda|)), the mean true
    ULP. They differ by a factor in [1, 2), so both go in every row and any
    normalised column can be rescaled after the fact.

    For a non-head site the instrumented reduction emits a 768-dim hidden
    vector, not the logits, so this is a nominal common scale that makes
    precisions comparable -- not that site's own ULP.
    """
    base = scales["lam_rms"] if which == "rms" else scales["lam_ulp_mean"]
    return math.ldexp(base, 1 - int(t))


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------

def complete_seeds(cell_dir, shape=None, dtype_size=4, numeric=True):
    """Seeds whose capture actually finished, in numeric order.

    Two departures from `decompose.cell_seeds`, both of which matter while an
    array is still draining:

    * It requires meta.json. `capture_logits.py` writes delta.npy, then
      scalars.npz, then meta.json last, so meta.json is the completion marker.
      A cell that is still being written has delta.npy but no meta.json, and its
      truncated array would read back as zeros -- which is precisely the failure
      run_config.sh guards against, silently biasing every fluctuation term
      upward.
    * It orders seeds numerically, not lexicographically. cell_seeds gives
      seed1, seed10, seed11, ..., seed2, so `--max-seeds 8` against a 32-seed
      pool would select seeds {1,10,11,12,13,14,15,16} rather than {1..8} -- a
      subset of a different experiment.

    When ``shape`` is given the file size is checked too, which catches a
    delta.npy that was truncated by something other than an interrupted write.
    """
    out = []
    for name in os.listdir(cell_dir):
        m = re.fullmatch(r"seed(\w+)", name)
        if not m:
            continue
        d = os.path.join(cell_dir, name)
        dp = os.path.join(d, "delta.npy")
        if not (os.path.exists(dp)
                and os.path.exists(os.path.join(d, "scalars.npz"))
                and os.path.exists(os.path.join(d, "meta.json"))):
            continue
        if shape is not None:
            want = shape[0] * shape[1] * dtype_size
            if os.path.getsize(dp) < want:
                continue
        out.append(name)
    if numeric:
        def key(n):
            tail = n[4:]
            return (0, int(tail)) if tail.isdigit() else (1, tail)
        out.sort(key=key)
    else:
        out.sort()
    return out


def chunk_budget(vocab, n_seeds, budget_bytes=1_000_000_000):
    """Tokens per chunk. Resident bytes are ~ c*V*(8S + 136): two float32
    [S,c,V] blocks plus about seventeen float64 [c,V] accumulators. Nothing here
    scales with N, so a wider window costs time and not memory."""
    # 4S for the float32 Delta block (lossless: delta.npy is float32 on disk)
    # + 8S for the float64 q-p block (computed here, so float32 would cost it
    # ~1e-7 of relative precision in Var_s(q)) + ~17 float64 [c,V] accumulators.
    per_token = vocab * (12 * n_seeds + 136)
    return int(min(128, max(4, budget_bytes // max(1, per_token))))


def sha256_of(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            if limit is not None and f.tell() > limit:
                break
    return h.hexdigest()


# --------------------------------------------------------------------------
# reference scales, for the ULP normalisation
# --------------------------------------------------------------------------

def logit_scales(ref_logits, chunk=64):
    """Magnitude scales of the reference logits, for the ULP denominators.

    ``lam_ulp_mean`` is the mean *true* ulp exponent, 2^floor(log2|lam|), got
    exactly from np.frexp rather than from a log. It matters because
    2^(1-t)*lam_rms equals ulp_t(lam_rms) only up to a factor in [1, 2): every
    row carries both so any normalized column can be rescaled afterwards.
    """
    n = ref_logits.shape[0]
    s2 = 0.0
    s_ulp = 0.0
    amax = 0.0
    count = 0
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        lam = np.asarray(ref_logits[sl], dtype=np.float64)
        s2 += float((lam * lam).sum())
        a = np.abs(lam)
        amax = max(amax, float(a.max()))
        _, e = np.frexp(a)                      # a = m * 2**e, m in [0.5, 1)
        s_ulp += float(np.ldexp(1.0, e - 1).sum())   # = 2**floor(log2 a)
        count += lam.size
    return dict(lam_rms=math.sqrt(s2 / count),
                lam_ulp_mean=s_ulp / count,
                lam_absmax=amax)


# --------------------------------------------------------------------------
# shape statistics from central power sums
# --------------------------------------------------------------------------

def central_from_shifted(P1, P2, P3, P4, k):
    """Central moments about the subset mean, from sums shifted by any origin.

    The central-moment formulas are translation equivariant, so shifting by the
    *full-sample* mean -- which makes P1 identically 0 for the full set and a
    single subtraction for a leave-one-out set -- is both legitimate and the
    numerically safest origin available. No raw power sums are ever formed, so
    there is no cancellation of large like-signed terms.
    """
    m = P1 / k
    M2 = P2 - k * m * m
    M3 = P3 - 3.0 * m * P2 + 2.0 * k * m ** 3
    M4 = P4 - 4.0 * m * P3 + 6.0 * m * m * P2 - 3.0 * k * m ** 4
    return m, M2, M3, M4


def m2_leave_one_out(E2, dev_i, km):
    """Central second sum over the seeds except ``i``, from the full sum.

    ``E2`` is sum_s (x_s - xbar)^2 over all k seeds and ``dev_i`` is
    x_i - xbar with the SAME xbar. Then sum_{s != i}(x_s - xbar)^2 = E2 - dev_i^2
    and the subset mean sits at -dev_i/km relative to xbar, so

        M2 = (E2 - dev_i^2) - km * (dev_i/km)^2.

    One pass instead of km, which is what keeps the jackknife O(S) overall.
    """
    out = (E2 - dev_i * dev_i) - (dev_i * dev_i) / km
    np.maximum(out, 0.0, out=out)
    return out


def fisher_shape(M2, M3, M4, k, mask):
    """Fisher-corrected skewness and excess kurtosis, per coordinate, averaged
    over the unmasked ones.

    Per-coordinate-then-average rather than pooling: G1 and G2 are (near)
    unbiased at each coordinate, so their coordinate mean estimates the pooled
    shape even though each individual value at S=8 is noise. The naive pooled
    ratio is not unbiased, because it standardizes with the same few draws; the
    uncorrected values are reported alongside so the size of that bias is
    visible.

    Returns (G1_sum, G2_sum, n_used) so a caller can accumulate over chunks.
    """
    if k < 4:
        return None, None, 0
    m2 = M2 / k
    m3 = M3 / k
    ok = mask & (m2 > 0)
    n_used = int(ok.sum())
    if n_used == 0:
        return None, None, 0
    m2o, m3o = m2[ok], m3[ok]
    g1 = m3o / np.power(m2o, 1.5)
    G1 = math.sqrt(k * (k - 1.0)) / (k - 2.0) * g1
    G1_sum = float(G1.sum())
    G2_sum = None
    if k >= 5:
        m4o = (M4 / k)[ok]
        g2 = m4o / (m2o * m2o) - 3.0
        G2 = (k - 1.0) / ((k - 2.0) * (k - 3.0)) * ((k + 1.0) * g2 + 6.0)
        G2_sum = float(G2.sum())
    return G1_sum, G2_sum, n_used


def hist_quantiles(counts, edges, total, quantiles=QUANTILES, under=0, over=0):
    """Quantiles by linear interpolation inside the containing bin.

    Under/overflow are tracked by the caller; a quantile that falls in an
    overflow bin is returned as None rather than clamped to an edge, so a
    truncated tail cannot masquerade as a measured one.
    """
    total += under + over
    if total <= 0:
        return {q: None for q in quantiles}
    cum = np.cumsum(counts)
    out = {}
    for q in quantiles:
        target = q * total - under
        if target <= 0 or target > counts.sum():
            out[q] = None
            continue
        k = int(np.searchsorted(cum, target, side="left"))
        if k >= len(counts):
            out[q] = None
            continue
        below = cum[k - 1] if k > 0 else 0.0
        c = counts[k]
        frac = 0.0 if c <= 0 else (target - below) / c
        out[q] = float(edges[k] + frac * (edges[k + 1] - edges[k]))
    return out


def ks_vs_normal(counts, edges, total):
    """max_bin |F_empirical - Phi| over the histogram's bin boundaries."""
    if total <= 0:
        return None
    from math import erf, sqrt
    cum = np.cumsum(counts) / total
    phi = np.array([0.5 * (1.0 + erf(e / sqrt(2.0))) for e in edges[1:]])
    return float(np.max(np.abs(cum - phi)))


# --------------------------------------------------------------------------
# per-subset accumulator
# --------------------------------------------------------------------------

class Acc:
    """Scalar and per-token accumulators for one seed subset.

    One instance holds the full-seed-set reduction; S further instances hold the
    leave-one-out reductions the jackknife needs. Every ``add`` is O(c*V) given
    quantities the caller has already formed, so the whole jackknife costs one
    extra O(S*c*V) pass rather than S full passes.
    """

    KEYS = ("b_mean", "b_rms", "b_absmax", "bt_rms", "bt_absmax",
            "sigma2_mean", "sigma2_rms", "sd_mean", "sd_rms", "sigma2t_mean",
            "sdt_rms", "frac_zero_var",
            "b_y_mean", "b_y_rms", "ep_b_mean", "sigma2_y_mean", "sd_y_rms",
            "ep_sigma2_mean", "var_ep_d_mean", "ep_sigma2t_mean",
            "D", "Q", "Q_naive", "V", "R", "K", "dL",
            "bq_y_mean", "bq_rms", "bq_l1_mean", "varq_y_mean", "varq_rms",
            "varq_sum_mean", "varq_mean", "sd_q_y_over_p_y",
            "bias_logq_y_mean", "ep_bias_logq_mean",
            "var_logq_y_mean", "ep_var_logq_mean",
            "coord_skew", "coord_exkurt",
            "z_pool_skew", "z_pool_exkurt", "w_pool_skew", "w_pool_exkurt")

    def __init__(self, n_tokens, want_per_token=False):
        self.n = n_tokens
        self.k = None                 # subset size, set on first add
        self.cells = 0                # (n, j) pairs seen
        self.tokens = 0
        self.s = {}                   # running sums, keyed by name
        self.mx = {}                   # running maxima
        # coord_*: per-coordinate Fisher G1/G2, averaged over coordinates.
        # These are scale invariant, so they are the SAME for the z and the w
        # standardizations -- the two differ only once values are pooled across
        # coordinates, which is what pool_* below measures.
        self.shape = {"coord_g1": 0.0, "coord_g2": 0.0, "coord_n": 0}
        self.pool = {}
        self.per_token = {} if want_per_token else None

    def _add(self, key, val):
        self.s[key] = self.s.get(key, 0.0) + float(val)

    def _max(self, key, val):
        self.mx[key] = max(self.mx.get(key, 0.0), float(val))

    def _tok(self, key, sl, arr):
        if self.per_token is None:
            return
        if key not in self.per_token:
            self.per_token[key] = np.full(self.n, np.nan, dtype=np.float64)
        self.per_token[key][sl] = arr

    # ---- the drift / fluctuation block -----------------------------------
    def add_logits(self, sl, k, p, y, b, M2, zero_mask, ep_b, var_e):
        """``b`` is [c,V] the subset mean of Delta; ``M2`` the central second
        sum (so sigma2 = M2/(k-1)); ``ep_b`` and ``var_e`` are [c] scalars."""
        self.k = k
        c = b.shape[0]
        self.cells += b.size
        self.tokens += c
        rows = np.arange(c)

        self._add("b_mean", b.sum())
        self._add("b_rms", (b * b).sum())
        self._max("b_absmax", np.abs(b).max())
        bt = b - ep_b[:, None]
        self._add("bt_rms", (bt * bt).sum())
        self._max("bt_absmax", np.abs(bt).max())

        b_y = b[rows, y]
        self._add("b_y_mean", b_y.sum())
        self._add("b_y_rms", (b_y * b_y).sum())
        self._add("ep_b_mean", ep_b.sum())
        self._tok("b_y", sl, b_y)
        self._tok("ep_b", sl, ep_b)

        # D is exact and linear in the seeds: E_p[b] - b_y.
        D_tok = ep_b - b_y
        self._add("D", D_tok.sum())
        self._tok("D_tok", sl, D_tok)

        if k >= 2:
            sigma2 = M2 / (k - 1.0)
            np.maximum(sigma2, 0.0, out=sigma2)
            sd = np.sqrt(sigma2)
            self._add("sigma2_mean", sigma2.sum())
            self._add("sigma2_rms", (sigma2 * sigma2).sum())
            self._add("sd_mean", sd.sum())
            self._add("sd_rms", sigma2.sum())        # rms(sd)^2 = mean(sigma2)
            self._add("frac_zero_var", zero_mask.sum())

            s2y = sigma2[rows, y]
            self._add("sigma2_y_mean", s2y.sum())
            self._add("sd_y_rms", s2y.sum())
            self._tok("sigma2_y", sl, s2y)

            ep_s2 = (p * sigma2).sum(axis=1)         # sum_j p_j sigma2_j
            self._add("ep_sigma2_mean", ep_s2.sum())
            self._add("var_ep_d_mean", var_e.sum())
            # (*) of the module docstring. Identical to decompose's V.
            V_tok = 0.5 * (ep_s2 - var_e)
            self._add("V", V_tok.sum())
            self._tok("V_tok", sl, V_tok)
            self._tok("ep_sigma2", sl, ep_s2)
            self._tok("var_ep_d", sl, var_e)

            # centered fluctuation: sum_j p_j Var(Delta_tilde_j) = 2V exactly
            ep_s2t = ep_s2 - var_e
            self._add("ep_sigma2t_mean", ep_s2t.sum())
            # sigma2_tilde_j = sigma2_j - 2 Cov(Delta_j, e) + Var(e); the
            # covariance term arrives already folded into M2t by the caller.
        else:
            # S = 1: b == Delta exactly and there is no fluctuation to measure.
            # Q = A = 1/2 Var_p(b) and V = 0 hold by eq:rn-degenerate; both are
            # reported, but every *measured* variance column stays blank.
            pass

        # Q_naive = 1/2 Var_p(b); Q = Q_naive - V/k. Both need Var_p(b) only.
        var_p_b = (p * b * b).sum(axis=1) - ep_b * ep_b
        Qn_tok = 0.5 * var_p_b
        self._add("Q_naive", Qn_tok.sum())
        self._tok("Q_naive_tok", sl, Qn_tok)

    def add_sigma2t(self, sl, k, M2t, zero_mask_t):
        """Centered per-coordinate variance, accumulated separately because the
        caller forms it from an extra cross moment."""
        if k < 2:
            return
        s2t = M2t / (k - 1.0)
        np.maximum(s2t, 0.0, out=s2t)
        self._add("sdt_rms", s2t.sum())
        self._tok("sdt_rms_tok", sl, np.sqrt(s2t.mean(axis=1)))

    # ---- softmax block ---------------------------------------------------
    def add_softmax(self, sl, k, p, y, bq, Fq2, bias_logq, var_logq):
        c = p.shape[0]
        rows = np.arange(c)
        self._add("bq_y_mean", bq[rows, y].sum())
        self._add("bq_rms", (bq * bq).sum())
        self._add("bq_l1_mean", np.abs(bq).sum())
        self._tok("bq_y", sl, bq[rows, y])

        pw = (p * bias_logq).sum(axis=1)
        self._add("bias_logq_y_mean", bias_logq[rows, y].sum())
        self._add("ep_bias_logq_mean", pw.sum())
        self._tok("bias_logq_y", sl, bias_logq[rows, y])
        self._tok("ep_bias_logq", sl, pw)

        if k >= 2:
            varq = Fq2 / (k - 1.0)
            np.maximum(varq, 0.0, out=varq)
            self._add("varq_y_mean", varq[rows, y].sum())
            self._add("varq_rms", (varq * varq).sum())
            self._add("varq_sum_mean", varq.sum(axis=1).sum())
            py = p[rows, y]
            self._add("sd_q_y_over_p_y", (np.sqrt(varq[rows, y]) / py).sum())
            self._tok("varq_y", sl, varq[rows, y])

            self._add("var_logq_y_mean", var_logq[rows, y].sum())
            self._add("ep_var_logq_mean", (p * var_logq).sum(axis=1).sum())
            self._tok("var_logq_y", sl, var_logq[rows, y])

    def add_shape(self, g1_sum, g2_sum, n_used):
        """Per-coordinate Fisher shape, accumulated over coordinates."""
        if n_used <= 0:
            return
        self.shape["coord_n"] += n_used
        if g1_sum is not None:
            self.shape["coord_g1"] += g1_sum
        if g2_sum is not None:
            self.shape["coord_g2"] += g2_sum

    def add_pooled(self, which, vals):
        """Pooled central moments of a standardized array, over all (s, n, j).

        Accumulated as raw power sums about zero, which is legitimate here and
        only here: both standardizations are already centered per coordinate, so
        the values are O(1) and of both signs, and there is no large mean for a
        raw sum to cancel against.
        """
        d = self.pool.setdefault(which, [0, 0.0, 0.0, 0.0, 0.0])
        v = vals.ravel()
        v = v[np.isfinite(v)]
        if v.size == 0:
            return
        d[0] += int(v.size)
        d[1] += float(v.sum())
        v2 = v * v
        d[2] += float(v2.sum())
        d[3] += float((v2 * v).sum())
        d[4] += float((v2 * v2).sum())

    def add_dL(self, sl, dL_tok):
        self._add("dL", dL_tok.sum())
        self._tok("dL_tok", sl, dL_tok)

    # ---- finalise --------------------------------------------------------
    def finish(self):
        k, C, T = self.k, self.cells, self.tokens
        out = {}
        deg = (k is not None and k < 2)

        out["b_mean"] = self.s.get("b_mean", 0.0) / C
        out["b_rms"] = math.sqrt(self.s.get("b_rms", 0.0) / C)
        out["b_absmax"] = self.mx.get("b_absmax")
        out["bt_rms"] = math.sqrt(self.s.get("bt_rms", 0.0) / C)
        out["bt_absmax"] = self.mx.get("bt_absmax")
        for key, denom in (("b_y_mean", T), ("ep_b_mean", T), ("D", T),
                           ("bq_y_mean", T), ("bq_l1_mean", T),
                           ("bias_logq_y_mean", T), ("ep_bias_logq_mean", T),
                           ("dL", T)):
            if key in self.s:
                out[key] = self.s[key] / denom
        out["b_y_rms"] = math.sqrt(self.s.get("b_y_rms", 0.0) / T)
        out["bq_rms"] = math.sqrt(self.s.get("bq_rms", 0.0) / C)

        Qn = self.s.get("Q_naive", 0.0) / T
        out["Q_naive"] = Qn

        if deg:
            # eq:rn-degenerate: xi = 0, so V = 0 and Q = A exactly. Reported as
            # theory (the columns exist and the identity holds); every measured
            # variance column is left absent by simply not being set here.
            out["V"] = 0.0
            out["Q"] = Qn
        else:
            out["V"] = self.s.get("V", 0.0) / T
            out["Q"] = Qn - out["V"] / k
            out["sigma2_mean"] = self.s["sigma2_mean"] / C
            out["sigma2_rms"] = math.sqrt(self.s["sigma2_rms"] / C)
            out["sd_mean"] = self.s["sd_mean"] / C
            out["sd_rms"] = math.sqrt(self.s["sd_rms"] / C)
            out["sigma2t_mean"] = self.s.get("ep_sigma2t_mean", 0.0) / T
            out["sdt_rms"] = math.sqrt(self.s.get("sdt_rms", 0.0) / C)
            out["frac_zero_var"] = self.s["frac_zero_var"] / C
            out["sigma2_y_mean"] = self.s["sigma2_y_mean"] / T
            out["sd_y_rms"] = math.sqrt(self.s["sd_y_rms"] / T)
            out["ep_sigma2_mean"] = self.s["ep_sigma2_mean"] / T
            out["var_ep_d_mean"] = self.s["var_ep_d_mean"] / T
            out["ep_sigma2t_mean"] = self.s["ep_sigma2t_mean"] / T
            for key, denom in (("varq_y_mean", T), ("varq_sum_mean", T),
                               ("sd_q_y_over_p_y", T),
                               ("var_logq_y_mean", T), ("ep_var_logq_mean", T)):
                if key in self.s:
                    out[key] = self.s[key] / denom
            if "varq_rms" in self.s:
                out["varq_rms"] = math.sqrt(self.s["varq_rms"] / C)
                out["varq_mean"] = self.s["varq_sum_mean"] / C
            n = self.shape["coord_n"]
            if n > 0:
                out["coord_skew"] = self.shape["coord_g1"] / n
                if k >= 5:
                    out["coord_exkurt"] = self.shape["coord_g2"] / n
            for which, d in self.pool.items():
                cnt, s1, s2, s3, s4 = d
                if cnt < 4:
                    continue
                m = s1 / cnt
                m2 = s2 / cnt - m * m
                if m2 <= 0:
                    continue
                m3 = s3 / cnt - 3.0 * m * (s2 / cnt) + 2.0 * m ** 3
                m4 = (s4 / cnt - 4.0 * m * (s3 / cnt)
                      + 6.0 * m * m * (s2 / cnt) - 3.0 * m ** 4)
                out[which + "_pool_skew"] = m3 / m2 ** 1.5
                out[which + "_pool_exkurt"] = m4 / (m2 * m2) - 3.0

        if "dL" in out:
            out["K"] = out["dL"] - out["D"]
            out["R"] = out["K"] - out["Q"] - out["V"]

        # A normalisation-free contrast: how much of the perturbation is drift
        # rather than fluctuation. Carries the SR/RN distinction with no ULP
        # denominator to argue about.
        if out.get("sd_rms"):
            out["drift_over_fluct"] = out["b_rms"] / out["sd_rms"]
        if out.get("sdt_rms"):
            out["drift_over_fluct_centered"] = out["bt_rms"] / out["sdt_rms"]
        return out


# --------------------------------------------------------------------------
# the streaming reduction
# --------------------------------------------------------------------------

def cell_stats(cell_dir, ref_dir, mode, t_bits, seed_names=None,
               budget_bytes=1_000_000_000, jack=True, softmax=True, hist=True,
               want_per_token=False, min_var=0.0, ref_logits=None,
               scales=None):
    """Reduce one capture cell to scalars, per-token arrays and histograms.

    Never materialises [N, V, S]: two float32 [S, c, V] blocks and a fixed set
    of float64 [c, V] accumulators are resident, and the chunk is sized so that
    peak RSS is ~1 GB independent of both S and N.
    """
    seeds = (complete_seeds(cell_dir) if seed_names is None
             else list(seed_names))
    if not seeds:
        raise SystemExit(f"no seed*/delta.npy under {cell_dir}")
    S = len(seeds)
    deterministic = (mode == "rn")

    own_ref = ref_logits is None
    if own_ref:
        ref_logits = np.load(os.path.join(ref_dir, "logits.npy"), mmap_mode="r")
    ref_scalars = np.load(os.path.join(ref_dir, "scalars.npz"))
    targets = ref_scalars["targets"]
    N, V = ref_logits.shape

    deltas = [np.load(os.path.join(cell_dir, s, "delta.npy"), mmap_mode="r")
              for s in seeds]
    for s, d in zip(seeds, deltas):
        if d.shape != (N, V):
            raise SystemExit(f"{cell_dir}/{s}: delta is {d.shape}, reference is {(N, V)}")

    # RN takes no randomness: replicas must be bitwise identical, and a
    # variance over replicas of a deterministic map measures nothing. Check the
    # premise rather than quietly averaging over it.
    rn_bitwise = None
    if deterministic and S > 1:
        rn_bitwise = True
        for d in deltas[1:]:
            if not np.array_equal(np.asarray(d), np.asarray(deltas[0])):
                rn_bitwise = False
                break
        if not rn_bitwise:
            raise SystemExit(f"{cell_dir}: RN replications differ; "
                             "these are not replicas of a deterministic map")

    # A deterministic cell has one distribution regardless of how many copies
    # of it are on disk, so it is reduced at k = 1.
    k_eff = 1 if deterministic else S
    use = [0] if deterministic else list(range(S))

    chunk = chunk_budget(V, len(use), budget_bytes)
    full = Acc(N, want_per_token=want_per_token)
    loo = ([Acc(N) for _ in use] if (jack and not deterministic and len(use) >= 3)
           else [])

    # per-seed, per-token scalars, for the free checks against scalars.npz
    e_st = np.empty((len(use), N), dtype=np.float64)     # p^T Delta
    c_st = np.empty((len(use), N), dtype=np.float64)     # log sum_j p_j e^Delta
    dy_st = np.empty((len(use), N), dtype=np.float64)    # Delta_y

    zc = np.zeros(len(Z_EDGES) - 1, dtype=np.int64)
    wc = np.zeros(len(W_EDGES) - 1, dtype=np.int64)
    bnc = np.zeros(len(BN_EDGES) - 1, dtype=np.int64)
    z_uo, w_uo, bn_stats = [0, 0], [0, 0], [0, 0]
    z_absmax, w_absmax, bn_absmax = [0.0], [0.0], [0.0]
    ytail = []          # the N*S target-coordinate values, kept in full
    probability_sum_error = 0.0
    probability_bias_sum_error = 0.0
    # ULP denominator for the drift histogram, from the reference logits
    u_scale = ulp_scale(t_bits, scales) if scales else 1.0

    for i0 in range(0, N, chunk):
        sl = slice(i0, min(i0 + chunk, N))
        c = sl.stop - sl.start
        rows = np.arange(c)
        y = targets[sl].astype(np.intp)
        p = load_p_chunk(ref_logits, sl)                  # [c,V] float64
        logp = np.log(p)

        blk = np.empty((len(use), c, V), dtype=np.float32)
        A1 = np.zeros((c, V), dtype=np.float64)
        Xe = np.zeros((c, V), dtype=np.float64)           # sum_s Delta * e_s
        SQ1 = np.zeros((c, V), dtype=np.float64) if softmax else None
        blk_q = (np.empty((len(use), c, V), dtype=np.float64)
                 if softmax else None)

        for si, s in enumerate(use):
            d = np.asarray(deltas[s][sl], dtype=np.float64)
            blk[si] = d.astype(np.float32)
            e = (p * d).sum(axis=1)
            tt = logp + d
            mx = tt.max(axis=1, keepdims=True)
            cs = (mx[:, 0] + np.log(np.exp(tt - mx).sum(axis=1)))
            e_st[si, sl] = e
            c_st[si, sl] = cs
            dy_st[si, sl] = d[rows, y]
            A1 += d
            Xe += d * e[:, None]
            if softmax:
                # store q - p, not q: the deviation keeps 24 bits of its own
                # magnitude, so the later variance does not lose digits to
                # cancellation against a large mean.
                dq = np.exp(tt - cs[:, None]) - p
                probability_sum_error = max(probability_sum_error,
                    float(np.abs((dq + p).sum(axis=1) - 1).max()))
                blk_q[si] = dq
                SQ1 += dq
            del d, tt

        kk = len(use)
        b = A1 / kk
        ep_b = e_st[:kk, sl].mean(axis=0)
        e_bar = ep_b
        var_e = (e_st[:kk, sl].var(axis=0, ddof=1) if kk >= 2
                 else np.zeros(c))

        # central power sums about the full-subset mean; P1 == 0 identically
        E2 = np.zeros((c, V), dtype=np.float64)
        E3 = np.zeros((c, V), dtype=np.float64)
        E4 = np.zeros((c, V), dtype=np.float64)
        for si in range(kk):
            dev = blk[si].astype(np.float64) - b
            d2 = dev * dev
            E2 += d2
            E3 += d2 * dev
            E4 += d2 * d2
            del dev, d2

        zero_mask = (E2 <= min_var)
        # Cov_s(Delta_j, e) folded in to give the p-centered variance:
        #   sigma2t_j = sigma2_j - 2 Cov(Delta_j, e) + Var(e)
        if kk >= 2:
            cov_de = (Xe - kk * b * e_bar[:, None]) / (kk - 1.0)
            M2t = E2 - 2.0 * (kk - 1.0) * cov_de + (kk - 1.0) * var_e[:, None]
            np.maximum(M2t, 0.0, out=M2t)
            del cov_de
        else:
            M2t = E2

        full.add_logits(sl, kk, p, y, b, E2, zero_mask, ep_b, var_e)
        full.add_sigma2t(sl, kk, M2t, zero_mask)
        full.add_dL(sl, (c_st[:kk, sl] - dy_st[:kk, sl]).mean(axis=0))

        if softmax:
            bq = SQ1 / kk
            probability_bias_sum_error = max(probability_bias_sum_error,
                float(np.abs(bq.sum(axis=1)).max()))
            Fq2 = np.zeros((c, V), dtype=np.float64)
            if kk >= 2:
                for si in range(kk):
                    dv = blk_q[si] - bq
                    Fq2 += dv * dv
                    del dv
            # log q_j - log p_j = Delta_j - c_s, so log q needs no block.
            c_bar = c_st[:kk, sl].mean(axis=0)
            bias_logq = b - c_bar[:, None]
            Flog2 = np.zeros((c, V), dtype=np.float64)
            if kk >= 2:
                # Central sums avoid cancellation and also give each
                # leave-one-out variance in O(c*V), without zero placeholders.
                for si in range(kk):
                    devlog = (blk[si].astype(np.float64) - b
                              - (c_st[si, sl] - c_bar)[:, None])
                    Flog2 += devlog * devlog
                del devlog
                var_logq = Flog2 / (kk - 1.0)
            else:
                var_logq = np.zeros((c, V))
            full.add_softmax(sl, kk, p, y, bq, Fq2, bias_logq, var_logq)
            del bias_logq, var_logq
        else:
            bq = Fq2 = None

        # ---- drift histogram: the one shape product defined at S = 1 -----
        # For RN there is a single realization, so the distribution the question
        # asks about is the spread of the drift across coordinates, not across
        # seeds. b/u is defined for every cell including S = 1, so this is the
        # column that makes the SR and RN rows comparable at all.
        if hist:
            bn = (b / u_scale).ravel()
            idx = np.digitize(bn, BN_EDGES) - 1
            bn_stats[0] += int((idx < 0).sum())
            bn_stats[1] += int((idx >= len(bnc)).sum())
            inb = idx[(idx >= 0) & (idx < len(bnc))]
            bnc += np.bincount(inb, minlength=len(bnc))
            bn_absmax[0] = max(bn_absmax[0], float(np.abs(bn).max()))
            full.add_pooled("bn", bn)
            del bn, idx, inb

        # ---- cross-seed shape, and the two standardizations --------------
        # G1 and G2 below are scale invariant, so they are identical for the
        # per-coordinate and the token-pooled standardization; only the *pooled*
        # moments differ, and those are what separate a clipped view from an
        # unclipped one. See the module docstring.
        #
        # All four central sums must belong to the SAME variable. The shape is
        # reported for the p-centered deviation Delta_tilde, because that is the
        # loss-relevant, shift-invariant part and the object the KL bound's
        # Gaussian model is stated on -- so E2t/E3t/E4t are accumulated here
        # from devt directly. (M2t, derived algebraically above, equals E2t
        # exactly and is used for the standardization.)
        if hist and kk >= 4:
            s2t = M2t / (kk - 1.0)
            ok = (s2t > min_var)
            sdt = np.sqrt(np.where(ok, s2t, 1.0))
            sbar = np.sqrt(np.maximum(s2t.mean(axis=1), 1e-300))
            E2t = np.zeros((c, V), dtype=np.float64)
            E3t = np.zeros((c, V), dtype=np.float64)
            E4t = np.zeros((c, V), dtype=np.float64)
            for si in range(kk):
                dev = blk[si].astype(np.float64) - b
                devt = dev - (dev * p).sum(axis=1)[:, None]
                d2 = devt * devt
                E2t += d2
                E3t += d2 * devt
                E4t += d2 * d2
                zv = np.where(ok, devt / sdt, np.nan)
                wv = devt / sbar[:, None]
                full.add_pooled("z", zv)
                full.add_pooled("w", wv)
                zf = zv[np.isfinite(zv)]
                if zf.size:
                    z_absmax[0] = max(z_absmax[0], float(np.abs(zf).max()))
                w_absmax[0] = max(w_absmax[0], float(np.abs(wv).max()))
                idx = np.digitize(zf, Z_EDGES) - 1
                z_uo[0] += int((idx < 0).sum())
                z_uo[1] += int((idx >= len(zc)).sum())
                inb = idx[(idx >= 0) & (idx < len(zc))]
                zc += np.bincount(inb, minlength=len(zc))
                jdx = np.digitize(wv.ravel(), W_EDGES) - 1
                w_uo[0] += int((jdx < 0).sum())
                w_uo[1] += int((jdx >= len(wc)).sum())
                inb = jdx[(jdx >= 0) & (jdx < len(wc))]
                wc += np.bincount(inb, minlength=len(wc))
                ytail.append(zv[rows, y].copy())
                del dev, devt, d2, zv, wv
            g1, g2, nu = fisher_shape(E2t, E3t, E4t, kk, ok)
            full.add_shape(g1, g2, nu)

            # Leave-one-out shape, still O(c*V) per omitted seed: the central
            # sums about the subset mean are literal subtractions of that seed's
            # contribution, then re-centered by central_from_shifted.
            if loo:
                for oi, acc in enumerate(loo):
                    km = kk - 1
                    if km < 4:
                        continue
                    dev = blk[oi].astype(np.float64) - b
                    devt = dev - (dev * p).sum(axis=1)[:, None]
                    d2 = devt * devt
                    P1 = -devt
                    _, M2m, M3m, M4m = central_from_shifted(
                        P1, E2t - d2, E3t - d2 * devt, E4t - d2 * d2, km)
                    np.maximum(M2m, 0.0, out=M2m)
                    g1m, g2m, num = fisher_shape(M2m, M3m, M4m, km,
                                                 (M2m > min_var))
                    acc.add_shape(g1m, g2m, num)
                    del dev, devt, d2, P1, M2m, M3m, M4m
            del s2t, sdt, sbar, ok, E2t, E3t, E4t

        # ---- leave-one-out, O(c*V) per omitted seed ----------------------
        for oi, acc in enumerate(loo):
            km = kk - 1
            dev_i = blk[oi].astype(np.float64) - b
            P1 = -dev_i
            d2 = dev_i * dev_i
            P2 = E2 - d2
            P3 = E3 - d2 * dev_i
            P4 = E4 - d2 * d2
            m, M2m, M3m, M4m = central_from_shifted(P1, P2, P3, P4, km)
            b_m = b + m
            keep = [j for j in range(kk) if j != oi]
            ep_b_m = e_st[keep, sl].mean(axis=0)
            var_e_m = e_st[keep, sl].var(axis=0, ddof=1) if km >= 2 else np.zeros(c)
            np.maximum(M2m, 0.0, out=M2m)
            acc.add_logits(sl, km, p, y, b_m, M2m, (M2m <= min_var),
                           ep_b_m, var_e_m)
            if km >= 2:
                Xe_m = Xe - blk[oi].astype(np.float64) * e_st[oi, sl][:, None]
                cov_m = (Xe_m - km * b_m * ep_b_m[:, None]) / (km - 1.0)
                M2t_m = M2m - 2.0 * (km - 1.0) * cov_m + (km - 1.0) * var_e_m[:, None]
                np.maximum(M2t_m, 0.0, out=M2t_m)
                acc.add_sigma2t(sl, km, M2t_m, (M2t_m <= min_var))
                # Shape is NOT computed here: M3m/M4m are central sums of the
                # raw deviation, and the shape is reported for the p-centered
                # one. It is accumulated in the shape block above, from E2t/E3t/
                # E4t, so that all four moments belong to the same variable.
                del Xe_m, cov_m, M2t_m
            acc.add_dL(sl, (c_st[keep, sl] - dy_st[keep, sl]).mean(axis=0))
            if softmax:
                bq_m = (SQ1 - blk_q[oi]) / km
                Fq2_m = np.zeros((c, V), dtype=np.float64)
                if km >= 2:
                    # O(1) passes, not O(km): the leave-one-out central sum is
                    # the full one minus this seed's contribution, re-centred.
                    # Re-summing over `keep` here cost S*(S-1) passes over
                    # [c, V] -- 56 at S = 8 but 992 at S = 32, which dominated
                    # everything else in the loop.
                    dvq = blk_q[oi] - bq
                    Fq2_m = m2_leave_one_out(Fq2, dvq, km)
                    del dvq
                c_bar_m = c_st[keep, sl].mean(axis=0)
                bias_logq_m = b_m - c_bar_m[:, None]
                devlog = dev_i - (c_st[oi, sl] - c_bar)[:, None]
                var_logq_m = m2_leave_one_out(Flog2, devlog, km) / (km - 1.0)
                np.maximum(var_logq_m, 0.0, out=var_logq_m)
                del devlog
                acc.add_softmax(sl, km, p, y, bq_m, Fq2_m, bias_logq_m, var_logq_m)
                del bq_m, Fq2_m, bias_logq_m, var_logq_m
            del dev_i, P1, d2, P2, P3, P4, M2m, M3m, M4m, b_m

        del blk, A1, Xe, E2, E3, E4, M2t, p, logp
        if softmax:
            del blk_q, SQ1

    res = dict(scalars=full.finish(), per_token=full.per_token or {},
               seeds=seeds, k_eff=k_eff, S_on_disk=S,
               deterministic=deterministic, rn_bitwise=rn_bitwise,
               n_scored=int(N), vocab=int(V))

    # jackknife: decompose.jackknife's convention, including its two refusals
    res["se"] = {}
    if loo:
        ests = [a.finish() for a in loo]
        kk = len(loo)
        for key in Acc.KEYS:
            vals = [e.get(key) for e in ests]
            if any(v is None for v in vals):
                continue
            a = np.asarray(vals, dtype=np.float64)
            res["se"]["se_" + key] = float(
                np.sqrt((kk - 1) / kk * ((a - a.mean()) ** 2).sum()))

    # histogram products
    if hist:
        ztot = int(zc.sum())
        wtot = int(wc.sum())
        res["hist"] = dict(z_counts=zc, z_edges=Z_EDGES, w_counts=wc,
                           w_edges=W_EDGES, bn_counts=bnc, bn_edges=BN_EDGES,
                           z_under=z_uo[0], z_over=z_uo[1],
                           w_under=w_uo[0], w_over=w_uo[1],
                           bn_under=bn_stats[0], bn_over=bn_stats[1])
        zq = hist_quantiles(zc, Z_EDGES, ztot, under=z_uo[0], over=z_uo[1])
        wq = hist_quantiles(wc, W_EDGES, wtot, under=w_uo[0], over=w_uo[1])
        for q, v in zq.items():
            res["scalars"]["z_" + QNAMES[q]] = v
        for q, v in wq.items():
            res["scalars"]["w_" + QNAMES[q]] = v
        res["scalars"]["z_absmax"] = z_absmax[0] or None
        res["scalars"]["w_absmax"] = w_absmax[0] or None
        res["scalars"]["z_under"] = z_uo[0]
        res["scalars"]["z_over"] = z_uo[1]
        res["scalars"]["w_under"] = w_uo[0]
        res["scalars"]["w_over"] = w_uo[1]
        bntot = int(bnc.sum())
        for q, v in hist_quantiles(bnc, BN_EDGES, bntot,
                                   under=bn_stats[0], over=bn_stats[1]).items():
            res["scalars"]["bn_" + QNAMES[q]] = v
        res["scalars"]["bn_absmax"] = bn_absmax[0] or None
        res["scalars"]["bn_under"] = bn_stats[0]
        res["scalars"]["bn_over"] = bn_stats[1]
        res["scalars"]["u_scale"] = u_scale
        res["scalars"]["z_ks_vs_normal"] = ks_vs_normal(zc, Z_EDGES, ztot)
        res["scalars"]["w_ks_vs_normal"] = ks_vs_normal(wc, W_EDGES, wtot)
        if ytail:
            yv = np.concatenate(ytail)
            yv = yv[np.isfinite(yv)]
            if yv.size:
                for q in QUANTILES:
                    res["scalars"]["zy_" + QNAMES[q]] = float(np.quantile(yv, q))
                m2 = float(((yv - yv.mean()) ** 2).mean())
                if m2 > 0:
                    m3 = float(((yv - yv.mean()) ** 3).mean())
                    m4 = float(((yv - yv.mean()) ** 4).mean())
                    res["scalars"]["zy_skew"] = m3 / m2 ** 1.5
                    res["scalars"]["zy_exkurt"] = m4 / (m2 * m2) - 3.0
    if not res["deterministic"] and k_eff >= 2:
        res["scalars"]["z_clip_bound"] = (k_eff - 1) / math.sqrt(k_eff)

    res["checks"] = _stored_checks(cell_dir, [seeds[i] for i in use],
                                   e_st[:len(use)], c_st[:len(use)],
                                   dy_st[:len(use)], ref_dir)
    if softmax:
        res["checks"].update(probability_sum_error=probability_sum_error,
                             probability_bias_sum_error=probability_bias_sum_error)
    if own_ref:
        del ref_logits
    return res


# --------------------------------------------------------------------------
# checks against what the capture already stored
# --------------------------------------------------------------------------

def _stored_checks(cell_dir, seeds, e_st, c_st, dy_st, ref_dir):
    """Compare this script's per-seed per-token scalars with scalars.npz.

    Free (no arrays), and it catches the class of bug no aggregate can: a seed
    ordering that does not match `decompose.load_scalars`, a window off by one,
    a reference from a different token set. Tolerance is 5e-5 relative, as
    kl_bounds.py uses, because Delta is stored in binary32 while the capture
    accumulated these same scalars in float64 from the float32 logits.
    """
    out = {}
    try:
        scal = load_scalars(cell_dir, seeds)
    except Exception as exc:                       # noqa: BLE001
        out["stored_check_error"] = str(exc)
        return out

    def rel(a, b):
        d = np.abs(a - b)
        scale = np.maximum(1.0, np.abs(b))
        return float((d / scale).max())

    out["ep_d_abserr"] = rel(e_st, scal["Ep_d"])
    out["d_y_abserr"] = rel(dy_st, scal["d_y"])
    out["dL_abserr"] = rel(c_st - dy_st, scal["dL"])
    # K = dL - D, with D = Ep_d - d_y per seed; also c_s - e_s exactly.
    K_stored = scal["dL"] - (scal["Ep_d"] - scal["d_y"])
    out["K_abserr"] = rel(c_st - e_st, K_stored)

    nonfinite = 0
    for s in seeds:
        mpath = os.path.join(cell_dir, s, "meta.json")
        if os.path.exists(mpath):
            with open(mpath) as f:
                meta = json.load(f)
            nonfinite += int(meta.get("nonfinite_delta", 0) or 0)
    out["nonfinite_delta"] = nonfinite

    rmeta_path = os.path.join(ref_dir, "meta.json")
    if os.path.exists(rmeta_path):
        with open(rmeta_path) as f:
            rmeta = json.load(f)
        mismatch = []
        for s in seeds:
            mpath = os.path.join(cell_dir, s, "meta.json")
            if not os.path.exists(mpath):
                continue
            with open(mpath) as f:
                meta = json.load(f)
            for key in ("num_windows", "window_idx", "n_scored",
                        "context_length", "max_tokens", "vocab"):
                if key in rmeta and key in meta and rmeta[key] != meta[key]:
                    mismatch.append(f"{s}:{key}")
        out["meta_mismatch"] = ",".join(sorted(set(mismatch)))
    return out


def gram_cross_check(cell_dir, ref_dir, seeds, res, deterministic):
    """Compare D, Q, V against decompose.terms() on the cached Gram.

    Read-only: it uses `gram.npz` if `decompose.py report` has already built it
    and reports blanks otherwise. This script must never build a Gram -- that is
    O(S^2 N V) and is decompose.py's job. Because the identities of the module
    docstring are exact, the expected agreement is ~1e-12 and a larger value
    means this script is wrong.
    """
    path = os.path.join(cell_dir, "gram.npz")
    if not os.path.exists(path):
        return {}
    try:
        with np.load(path) as z:
            cached = [str(x) for x in z["seeds"]]
            gram = z["gram"]
        if len(set(cached)) != len(cached) or not set(seeds).issubset(cached):
            return {"gram_check_skipped": "requested seeds absent or duplicated in cache"}
        scal = load_scalars(cell_dir, cached)
        ref = aggregate(terms(gram, scal, [cached.index(s) for s in seeds], deterministic))
    except Exception as exc:                       # noqa: BLE001
        return {"gram_check_skipped": str(exc)}
    out = {}
    for key in ("D", "Q", "V"):
        mine = res["scalars"].get(key)
        if mine is None:
            continue
        out[key + "_decompose_abserr"] = abs(float(mine) - float(ref[key]))
    return out


# --------------------------------------------------------------------------
# row assembly
# --------------------------------------------------------------------------

IDENT = ("site", "mode", "t", "seeds", "seed_names", "n_scored", "vocab",
         "delta_dtype", "variance_status", "variance_theory",
         "rn_bitwise_identical")
SCALE = ("lam_rms", "lam_ulp_mean", "lam_absmax", "ulp", "ulp_exact")
DRIFT = ("b_mean", "b_rms", "b_absmax", "b_y_mean", "b_y_rms", "ep_b_mean",
         "bt_rms", "bt_absmax", "b_rms_ulp", "bt_rms_ulp", "b_y_mean_ulp")
FLUCT = ("sigma2_mean", "sigma2_rms", "sd_mean", "sd_rms", "sigma2_y_mean",
         "sd_y_rms", "sigma2t_mean", "sdt_rms", "ep_sigma2_mean",
         "var_ep_d_mean", "ep_sigma2t_mean", "sd_rms_ulp", "sdt_rms_ulp",
         "sd_y_rms_ulp", "frac_zero_var", "drift_over_fluct",
         "drift_over_fluct_centered")
TERMS = ("D", "Q_naive", "Q", "V", "R", "K", "dL")
SOFTM = ("bq_y_mean", "bq_rms", "bq_l1_mean", "varq_y_mean", "varq_rms",
         "varq_sum_mean", "varq_mean", "sd_q_y_over_p_y", "bias_logq_y_mean",
         "ep_bias_logq_mean", "var_logq_y_mean", "ep_var_logq_mean")
SHAPE = (("coord_skew", "coord_exkurt", "z_clip_bound",
          "z_pool_skew", "z_pool_exkurt", "w_pool_skew", "w_pool_exkurt",
          "bn_pool_skew", "bn_pool_exkurt",
          "z_absmax", "w_absmax", "bn_absmax", "u_scale",
          "z_under", "z_over", "w_under", "w_over", "bn_under", "bn_over",
          "z_ks_vs_normal", "w_ks_vs_normal", "zy_skew", "zy_exkurt")
         + tuple("z_" + QNAMES[q] for q in QUANTILES)
         + tuple("w_" + QNAMES[q] for q in QUANTILES)
         + tuple("bn_" + QNAMES[q] for q in QUANTILES)
         + tuple("zy_" + QNAMES[q] for q in QUANTILES))
CHECKS = ("ep_d_abserr", "d_y_abserr", "dL_abserr", "K_abserr",
          "probability_sum_error", "probability_bias_sum_error",
          "nonfinite_delta", "meta_mismatch", "stored_check_error",
          "QV_sum_abserr", "ep_sigma2t_vs_2V_abserr", "bq_sum_identity",
          "D_decompose_abserr", "Q_decompose_abserr", "V_decompose_abserr",
          "gram_check_skipped")
PROV = ("root", "chunk", "budget_bytes", "hist_bin_width_z",
        "hist_bin_width_w", "script_version", "elapsed_s")
SE_KEYS = tuple("se_" + k for k in Acc.KEYS)
FIELDS = IDENT + SCALE + DRIFT + FLUCT + TERMS + SOFTM + SHAPE + SE_KEYS + CHECKS + PROV


def to_row(res, site, t, mode, scales, root, elapsed, budget_bytes, chunk,
           delta_dtype):
    sc = dict(res["scalars"])
    row = {k: "" for k in FIELDS}
    k = res["k_eff"]
    degenerate = res["deterministic"] or k < 2

    row.update(site=site, mode=mode, t=t, seeds=k,
               seed_names=";".join(res["seeds"]),
               n_scored=res["n_scored"], vocab=res["vocab"],
               delta_dtype=delta_dtype,
               variance_status=("degenerate_S1" if degenerate else "measured"),
               variance_theory=("0 by eq:rn-degenerate" if res["deterministic"]
                                else ""),
               rn_bitwise_identical=("" if res["rn_bitwise"] is None
                                     else int(res["rn_bitwise"])))
    row.update(lam_rms=scales["lam_rms"], lam_ulp_mean=scales["lam_ulp_mean"],
               lam_absmax=scales["lam_absmax"],
               ulp=ulp_scale(t, scales), ulp_exact=ulp_scale(t, scales, "exact"))

    u = ulp_scale(t, scales)
    for name in DRIFT + FLUCT + TERMS + SOFTM + SHAPE:
        if name in sc and sc[name] is not None:
            row[name] = sc[name]
    # normalised columns
    for src, dst in (("b_rms", "b_rms_ulp"), ("bt_rms", "bt_rms_ulp"),
                     ("b_y_mean", "b_y_mean_ulp"), ("sd_rms", "sd_rms_ulp"),
                     ("sdt_rms", "sdt_rms_ulp"), ("sd_y_rms", "sd_y_rms_ulp")):
        if sc.get(src) is not None:
            row[dst] = float(sc[src]) / u

    for key, val in res["se"].items():
        if key in row:
            row[key] = val
    for key, val in res["checks"].items():
        if key in row:
            row[key] = val

    # internal identities
    if sc.get("V") is not None and sc.get("Q") is not None and k:
        # E[Q_naive] = Q + V/S, and the estimators are defined so the sample
        # identity Q_naive == Q + V/S holds exactly, not just in expectation.
        row["QV_sum_abserr"] = abs((sc["Q"] + sc["V"] / k) - sc["Q_naive"])
    if sc.get("ep_sigma2t_mean") is not None and sc.get("V") is not None:
        row["ep_sigma2t_vs_2V_abserr"] = abs(sc["ep_sigma2t_mean"] - 2.0 * sc["V"])
    row["bq_sum_identity"] = "sum_j (mean_s q_j - p_j) == 0 identically"

    row.update(root=root, chunk=chunk, budget_bytes=budget_bytes,
               hist_bin_width_z=float(Z_EDGES[1] - Z_EDGES[0]),
               hist_bin_width_w=float(W_EDGES[1] - W_EDGES[0]),
               script_version=SCRIPT_VERSION, elapsed_s=round(elapsed, 2))
    return row


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def _write_rows(path, rows):
    """Atomic: the sbatch cleanup does not remove a half-written CSV, and
    `merge` must never ingest one."""
    tmp = path + ".partial"
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


def _delta_dtype(cell_dir, seeds):
    d = np.load(os.path.join(cell_dir, seeds[0], "delta.npy"), mmap_mode="r")
    return str(d.dtype)


def _run_cells(cells, root, args, ref_logits=None, scales=None):
    ref_dir = os.path.join(root, "reference")
    if ref_logits is None:
        ref_logits = np.load(os.path.join(ref_dir, "logits.npy"), mmap_mode="r")
    if scales is None:
        scales = logit_scales(ref_logits)
    rows = []
    for site, t, mode, cdir in cells:
        all_seeds = complete_seeds(cdir, shape=ref_logits.shape)
        on_disk = cell_seeds(cdir)
        if len(all_seeds) != len(on_disk):
            print(f"  note: {site} t{t} {mode}: {len(on_disk) - len(all_seeds)} "
                  f"of {len(on_disk)} seed dir(s) incomplete (no meta.json or "
                  f"short delta.npy); excluded", file=sys.stderr)
        seeds = all_seeds[:args.max_seeds] if args.max_seeds else all_seeds
        if not seeds:
            print(f"  skip: {site} t{t} {mode}: no complete seeds",
                  file=sys.stderr)
            continue
        dt = _delta_dtype(cdir, seeds)
        if args.strict and dt != "float32":
            print(f"strict: {site} t{t} {mode} stores Delta as {dt}; shape "
                  f"statistics are not emitted for a narrower dtype",
                  file=sys.stderr)
        t0 = time.time()
        res = cell_stats(cdir, ref_dir, mode, t, seed_names=seeds,
                         budget_bytes=args.budget_bytes,
                         jack=not args.no_jackknife,
                         softmax=not args.no_softmax,
                         hist=not args.no_hist and dt == "float32",
                         want_per_token=bool(args.per_token_dir),
                         min_var=args.min_var, ref_logits=ref_logits,
                         scales=scales)
        elapsed = time.time() - t0
        chunk = chunk_budget(res["vocab"], res["k_eff"], args.budget_bytes)
        row = to_row(res, site, t, mode, scales, root, elapsed,
                     args.budget_bytes, chunk, dt)
        if not args.no_gram:
            row.update({k: v for k, v in gram_cross_check(
                cdir, ref_dir, res["seeds"], res, res["deterministic"]).items()
                if k in row})
        rows.append(row)
        print(f"  {site:12s} t{t:02d} {mode:2s} S={res['k_eff']:2d}  "
              f"D={row['D']:+.6g}  V={row['V'] if row['V'] != '' else 'n/a':>10}  "
              f"b_rms/u={row['b_rms_ulp']:.4g}  "
              f"{'sd_rms/u=%.4g' % row['sd_rms_ulp'] if row['sd_rms_ulp'] != '' else 'sd: n/a'}"
              f"  [{elapsed:.0f}s]", flush=True)
        if args.per_token_dir:
            os.makedirs(args.per_token_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(args.per_token_dir, f"{site}_t{t:02d}_{mode}.npz"),
                **{k: v for k, v in res["per_token"].items()})
        if args.hist_dir and "hist" in res:
            os.makedirs(args.hist_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(args.hist_dir, f"{site}_t{t:02d}_{mode}.npz"),
                **res["hist"])
    return rows


def _result_line(rows, elapsed):
    if not rows:
        print(f"Result -> Cells: 0 | Time: {elapsed:.2f}s")
        return
    ts = sorted({int(r["t"]) for r in rows})
    modes = sorted({r["mode"] for r in rows})
    seeds = sorted({int(r["seeds"]) for r in rows})
    print(f"Result -> Cells: {len(rows)} | Bits: {min(ts):02d}-{max(ts):02d} | "
          f"Modes: {','.join(modes)} | Seeds: {min(seeds)}-{max(seeds)} | "
          f"Tokens: {rows[0]['n_scored']} | Time: {elapsed:.2f}s")


def cmd_cell(args):
    t0 = time.time()
    root = args.root
    cdir = os.path.join(root, args.site, f"t{args.t:02d}", args.mode)
    if not os.path.isdir(cdir):
        raise SystemExit(f"no such cell: {cdir}")
    rows = _run_cells([(args.site, args.t, args.mode, cdir)], root, args)
    _write_rows(args.cell_csv, rows)
    print(f"wrote {args.cell_csv}")
    _result_line(rows, time.time() - t0)


def cmd_report(args):
    t0 = time.time()
    root = args.root
    cells = discover(root)
    if args.sites:
        cells = [c for c in cells if c[0] in args.sites]
    if args.t:
        cells = [c for c in cells if c[1] in args.t]
    if args.modes:
        cells = [c for c in cells if c[2] in args.modes]
    if not cells:
        raise SystemExit("no cells matched")
    print(f"{len(cells)} cell(s) under {root}")
    rows = _run_cells(cells, root, args)
    if args.csv:
        _write_rows(args.csv, rows)
        print(f"wrote {args.csv}")
    if args.long_csv:
        _write_long(args.long_csv, rows)
        print(f"wrote {args.long_csv}")
    print()
    print("blank in a variance column: one deterministic realization, so there "
          "is no fluctuation to measure (eq:rn-degenerate).")
    print("This is NOT a measured variance of zero. See variance_status.")
    _result_line(rows, time.time() - t0)


def _write_long(path, rows):
    """(site, mode, t, statistic, value, se) -- what a plot or a LaTeX table
    wants, against the wide file which is the archival one."""
    tmp = path + ".partial"
    stats = [k for k in FIELDS
             if k not in IDENT + PROV + CHECKS and not k.startswith("se_")]
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site", "mode", "t", "seeds", "statistic", "value", "se"])
        for r in rows:
            for k in stats:
                if r.get(k) == "" or r.get(k) is None:
                    continue
                w.writerow([r["site"], r["mode"], r["t"], r["seeds"], k,
                            r[k], r.get("se_" + k, "")])
    os.replace(tmp, path)


def cmd_merge(args):
    paths = sorted(glob.glob(args.cells))
    if not paths:
        raise SystemExit(f"no CSVs matched {args.cells}")
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    rows.sort(key=lambda r: (r["site"], int(r["t"]), r["mode"]))
    _write_rows(args.csv, rows)
    print(f"merged {len(paths)} file(s), {len(rows)} row(s) -> {args.csv}")
    if args.long_csv:
        _write_long(args.long_csv, rows)
        print(f"wrote {args.long_csv}")
    print(f"Result -> Cells: {len(rows)} | Merged: {len(paths)} | Time: 0.00s")


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------

def _naive(lam, deltas, targets, p=None):
    """A deliberately naive [S,N,V] reference implementation.

    Exists so the streaming path can be checked against something written
    without any of its optimisations: no chunking, no shifted power sums, no
    leave-one-out algebra, and the softmax formed directly rather than through
    log q - log p = Delta - c.
    """
    S = len(deltas)
    lam = np.asarray(lam, dtype=np.float64)
    d = np.stack([np.asarray(x, dtype=np.float64) for x in deltas])   # [S,N,V]
    lsm = log_softmax_np(lam)
    p = np.exp(lsm)
    N, V = lam.shape
    rows = np.arange(N)
    y = np.asarray(targets, dtype=np.intp)

    b = d.mean(axis=0)
    s2 = d.var(axis=0, ddof=1) if S > 1 else None
    e = np.einsum("nj,snj->sn", p, d)
    ep_b = e.mean(axis=0)
    var_e = e.var(axis=0, ddof=1) if S > 1 else np.zeros(N)

    q = np.stack([np.exp(log_softmax_np(lam + d[s])) for s in range(S)])
    logq = np.stack([log_softmax_np(lam + d[s]) for s in range(S)])
    bq = q.mean(axis=0) - p
    varq = q.var(axis=0, ddof=1) if S > 1 else None
    bias_logq = logq.mean(axis=0) - lsm
    var_logq = logq.var(axis=0, ddof=1) if S > 1 else None

    dL = (-logq[:, rows, y] + lsm[rows, y]).mean(axis=0)
    D = (ep_b - b[rows, y]).mean()

    out = dict(b=b, s2=s2, e=e, ep_b=ep_b, var_e=var_e, bq=bq, varq=varq,
               bias_logq=bias_logq, var_logq=var_logq, p=p, lsm=lsm,
               D=float(D), dL=float(dL.mean()),
               b_rms=float(np.sqrt((b * b).mean())),
               b_y_mean=float(b[rows, y].mean()),
               ep_b_mean=float(ep_b.mean()),
               bq_y_mean=float(bq[rows, y].mean()),
               bq_l1_mean=float(np.abs(bq).sum(axis=1).mean()),
               bias_logq_y_mean=float(bias_logq[rows, y].mean()),
               ep_bias_logq_mean=float((p * bias_logq).sum(axis=1).mean()))
    if S > 1:
        Qn = 0.5 * ((p * b * b).sum(axis=1) - ep_b * ep_b)
        V = 0.5 * ((p * s2).sum(axis=1) - var_e)
        out.update(sigma2_mean=float(s2.mean()),
                   sd_rms=float(np.sqrt(s2.mean())),
                   sigma2_y_mean=float(s2[rows, y].mean()),
                   V=float(V.mean()), Q_naive=float(Qn.mean()),
                   Q=float(Qn.mean() - V.mean() / S),
                   varq_y_mean=float(varq[rows, y].mean()),
                   var_logq_y_mean=float(var_logq[rows, y].mean()),
                   ep_var_logq_mean=float((p * var_logq).sum(axis=1).mean()))
    return out


def _synth(rng, N, V, S, b_vec, L, shift=None, dist="normal", lam_sd=3.0):
    lam = rng.normal(0.0, lam_sd, size=(N, V))
    targets = rng.integers(0, V, size=N).astype(np.int32)
    deltas = []
    for _ in range(S):
        if dist == "normal":
            zz = rng.normal(size=(N, V))
        elif dist == "rademacher":
            zz = rng.integers(0, 2, size=(N, V)) * 2.0 - 1.0
        elif dist == "uniform":
            zz = rng.uniform(-math.sqrt(3.0), math.sqrt(3.0), size=(N, V))
        elif dist == "exponential":
            zz = rng.exponential(size=(N, V)) - 1.0
        else:
            raise ValueError(dist)
        d = b_vec[None, :] + zz @ L.T
        if shift is not None:
            d = d + shift[:, None]
        deltas.append(d)
    return lam, targets, deltas


def _write_tree(tmp, lam, targets, deltas, site="lm_head", t=7, mode="sr",
                dtype=np.float32):
    from decompose import reduce_perturbation, reduce_reference
    ref = os.path.join(tmp, "reference")
    os.makedirs(ref, exist_ok=True)
    lam32 = lam.astype(np.float32)
    np.save(os.path.join(ref, "logits.npy"), lam32)
    nll, coll = reduce_reference(lam32, targets)
    np.savez(os.path.join(ref, "scalars.npz"), targets=targets, nll=nll,
             collision=coll)
    with open(os.path.join(ref, "meta.json"), "w") as f:
        json.dump(dict(site="none", precision=24, mode="rn", n_scored=len(targets),
                       vocab=lam.shape[1], num_windows=1, window_idx=0,
                       context_length=256, max_tokens=len(targets) + 1), f)
    ref_mm = np.load(os.path.join(ref, "logits.npy"), mmap_mode="r")
    for i, d in enumerate(deltas, start=1):
        cd = os.path.join(tmp, site, f"t{t:02d}", mode, f"seed{i}")
        os.makedirs(cd, exist_ok=True)
        # Recover Delta exactly when dtype is float64: keep the reference at
        # float32 (as in a real capture) but form the perturbed logits at the
        # delta dtype, so Delta = pert - lam32 is representable.
        pert = (lam32.astype(dtype) + d.astype(dtype))
        out = np.lib.format.open_memmap(os.path.join(cd, "delta.npy"), mode="w+",
                                        dtype=dtype, shape=lam.shape)
        cols, nf = reduce_perturbation(ref_mm, pert, targets, out)
        out.flush(); del out
        np.savez(os.path.join(cd, "scalars.npz"), targets=targets, **cols)
        with open(os.path.join(cd, "meta.json"), "w") as f:
            json.dump(dict(site=site, precision=t, mode=mode, seed=str(i),
                           n_scored=len(targets), vocab=lam.shape[1],
                           num_windows=1, window_idx=0, context_length=256,
                           max_tokens=len(targets) + 1, nonfinite_delta=nf), f)
    return ref


def cmd_selftest(args):
    rng = np.random.default_rng(20260910)
    N, V = 24, 64
    fails = []

    def check(name, got, want, tol, rel=False):
        if got is None or want is None:
            fails.append(f"{name}: got {got}, want {want}")
            return
        d = abs(got - want)
        if rel:
            d /= max(1e-300, abs(want))
        ok = d <= tol
        print(f"  {'ok  ' if ok else 'FAIL'} {name:42s} {got:+.10g} vs "
              f"{want:+.10g}  ({'rel' if rel else 'abs'} {d:.2e} <= {tol:.0e})")
        if not ok:
            fails.append(name)

    # A deliberately NON-diagonal Sigma: 0.4 I + 0.05 11^T, so tr(H Sigma) is a
    # real trace and no diagonal-covariance shortcut can hide a bug.
    Sigma = 0.4 * np.eye(V) + 0.05 * np.ones((V, V))
    L = np.linalg.cholesky(Sigma)
    b_vec = rng.normal(0.0, 0.2, size=V)

    tmpbase = args.dir or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), f"dist-selftest-{os.getpid()}")

    for S in (1, 2, 4, 5, 8, 32):
        print(f"\n--- S = {S} " + "-" * 50)
        lam, targets, deltas = _synth(rng, N, V, S, b_vec, L)
        tmp = os.path.join(tmpbase, f"S{S}")
        os.makedirs(tmp, exist_ok=True)
        mode = "rn" if S == 1 else "sr"
        ref = _write_tree(tmp, lam, targets, deltas, mode=mode)
        ref_mm = np.load(os.path.join(ref, "logits.npy"), mmap_mode="r")
        scales = logit_scales(ref_mm)
        nv = _naive(ref_mm, [np.load(os.path.join(tmp, "lm_head", "t07", mode,
                                                  f"seed{i}", "delta.npy"))
                             for i in range(1, S + 1)], targets)
        cdir = os.path.join(tmp, "lm_head", "t07", mode)

        # (1) streaming == naive, at chunk sizes that do not divide N = 24
        for cb in (V * (8 * S + 136) * 1, V * (8 * S + 136) * 5,
                   V * (8 * S + 136) * 7, V * (8 * S + 136) * 24):
            res = cell_stats(cdir, ref, mode, 7, budget_bytes=cb, jack=False,
                             ref_logits=ref_mm, scales=scales)
            sc = res["scalars"]
            ch = chunk_budget(V, res["k_eff"], cb)
            check(f"S={S} chunk={ch:2d} b_rms", sc["b_rms"], nv["b_rms"], 1e-11, True)
            check(f"S={S} chunk={ch:2d} D", sc["D"], nv["D"], 1e-11, True)
            if S > 1:
                check(f"S={S} chunk={ch:2d} V", sc["V"], nv["V"], 1e-11, True)

        res = cell_stats(cdir, ref, mode, 7, budget_bytes=10**9, jack=True,
                         ref_logits=ref_mm, scales=scales)
        sc, se = res["scalars"], res["se"]

        # (4) D is exact and linear
        check(f"S={S} D exact", sc["D"], nv["D"], 1e-12, True)
        check(f"S={S} dL exact", sc["dL"], nv["dL"], 1e-11, True)
        check(f"S={S} b_y_mean", sc["b_y_mean"], nv["b_y_mean"], 1e-11, True)
        check(f"S={S} ep_b_mean", sc["ep_b_mean"], nv["ep_b_mean"], 1e-11, True)

        # (8) softmax identities
        check(f"S={S} bq_y_mean", sc["bq_y_mean"], nv["bq_y_mean"], 1e-9, True)
        check(f"S={S} bq_l1_mean", sc["bq_l1_mean"], nv["bq_l1_mean"], 1e-9, True)
        check(f"S={S} bias_logq_y == -dL", sc["bias_logq_y_mean"], -nv["dL"], 1e-11, True)
        K = nv["dL"] - nv["D"]
        check(f"S={S} ep_bias_logq == -K", sc["ep_bias_logq_mean"], -K, 1e-11, True)

        if S == 1:
            # (14) the degenerate path: b == Delta exactly, no measured variance
            d0 = np.load(os.path.join(cdir, "seed1", "delta.npy"))
            check("S=1 b == Delta", sc["b_rms"],
                  float(np.sqrt((d0.astype(np.float64) ** 2).mean())), 1e-11, True)
            for key in ("sigma2_mean", "sd_rms", "varq_y_mean", "coord_skew"):
                bad = sc.get(key) is not None
                print(f"  {'FAIL' if bad else 'ok  '} S=1 {key:38s} "
                      f"{'present!' if bad else 'absent (correct)'}")
                if bad:
                    fails.append(f"S=1 {key} present")
            if se:
                fails.append("S=1 jackknife not empty")
                print("  FAIL S=1 jackknife should be empty")
            else:
                print("  ok   S=1 jackknife empty (correct)")
            continue

        # (3)(5)(6) variance and the two term identities
        check(f"S={S} sigma2_mean", sc["sigma2_mean"], nv["sigma2_mean"], 1e-11, True)
        check(f"S={S} sd_rms", sc["sd_rms"], nv["sd_rms"], 1e-11, True)
        check(f"S={S} sigma2_y_mean", sc["sigma2_y_mean"], nv["sigma2_y_mean"], 1e-11, True)
        check(f"S={S} V", sc["V"], nv["V"], 1e-11, True)
        check(f"S={S} Q_naive", sc["Q_naive"], nv["Q_naive"], 1e-11, True)
        check(f"S={S} Q", sc["Q"], nv["Q"], 1e-11, True)
        check(f"S={S} Q_naive - Q == V/S", sc["Q_naive"] - sc["Q"],
              sc["V"] / S, 1e-12, True)
        check(f"S={S} ep_sigma2t == 2V", sc["ep_sigma2t_mean"],
              2.0 * sc["V"], 1e-10, True)
        check(f"S={S} varq_y_mean", sc["varq_y_mean"], nv["varq_y_mean"], 1e-9, True)
        check(f"S={S} var_logq_y_mean", sc["var_logq_y_mean"],
              nv["var_logq_y_mean"], 1e-9, True)

        # (7) agreement with decompose.terms() on an in-memory Gram
        from decompose import build_gram
        seeds_g, gram = build_gram(cdir, ref)
        scal = load_scalars(cdir, seeds_g)
        ref_terms = aggregate(terms(gram, scal, list(range(S)), mode == "rn"))
        # Q and V come from delta.npy on both sides, so they agree to
        # floating-point noise. D does not: decompose reads it from the stored
        # scalars, which the capture accumulated in float64 from the float32
        # *logits*, while this script recomputes it from the float32-rounded
        # Delta. The gap is the storage dtype of Delta and nothing else -- it is
        # what ep_d_abserr measures directly.
        check(f"S={S} Q == decompose.terms", sc["Q"], ref_terms["Q"], 1e-10, True)
        check(f"S={S} V == decompose.terms", sc["V"], ref_terms["V"], 1e-10, True)
        check(f"S={S} D == decompose.terms", sc["D"], ref_terms["D"], 1e-6, True)

        # (11) the clipping bound. Only observable at S >= 4, where the shape
        #      block runs; the column itself is a structural bound at any S.
        bound = (S - 1) / math.sqrt(S)
        check(f"S={S} z_clip_bound column", sc["z_clip_bound"], bound, 1e-12, True)
        if S >= 4:
            got = sc.get("z_absmax")
            ok = got is not None and got <= bound * (1 + 1e-9)
            print(f"  {'ok  ' if ok else 'FAIL'} S={S} |z| <= (S-1)/sqrt(S)"
                  f"{'':17s} {got:.6f} <= {bound:.4f}")
            if not ok:
                fails.append(f"S={S} clip bound")

        # (13b) the O(S) leave-one-out algebra must give exactly what an
        #       explicit recomputation on each S-1 subset gives. This is what
        #       covers the softmax path, whose central sum is reconstructed by
        #       m2_leave_one_out rather than re-summed.
        if S >= 4:
            names = [f"seed{i}" for i in range(1, S + 1)]
            brute = {}
            for key in ("varq_y_mean", "var_logq_y_mean", "ep_var_logq_mean",
                        "sd_rms", "V", "R", "K", "coord_skew"):
                vals = []
                for i in range(S):
                    sub = [n for j, n in enumerate(names) if j != i]
                    r = cell_stats(cdir, ref, mode, 7, seed_names=sub,
                                   jack=False, ref_logits=ref_mm, scales=scales)
                    vals.append(r["scalars"].get(key))
                if any(v is None for v in vals):
                    continue
                a = np.asarray(vals, dtype=np.float64)
                brute[key] = float(np.sqrt((S - 1) / S
                                           * ((a - a.mean()) ** 2).sum()))
            for key, want in brute.items():
                check(f"S={S} se_{key} O(S) == brute force",
                      se.get("se_" + key), want, 1e-9, True)

        # (13) jackknife convention: for the linear D, the delete-one SE equals
        #      std(D_s, ddof=1)/sqrt(S) identically.
        if S >= 3:
            p_ref = nv["p"]
            rows = np.arange(N)
            yv = targets.astype(np.intp)
            Ds = np.array([float(((p_ref * np.asarray(d)).sum(axis=1)
                                  - np.asarray(d)[rows, yv]).mean())
                           for d in [np.load(os.path.join(cdir, f"seed{i}",
                                                          "delta.npy"))
                                     for i in range(1, S + 1)]])
            check(f"S={S} se_D == std/sqrt(S)", se.get("se_D"),
                  float(Ds.std(ddof=1) / math.sqrt(S)), 1e-9, True)

    # (16) shift invariance.
    #
    # delta L is invariant to Delta -> Delta + c*1, so every centered, softmax
    # and term column must be too. The invariance is exact in exact arithmetic,
    # but the centered second moment is formed as a difference
    # (E2 - 2(k-1)Cov + (k-1)Var_e) whose parts grow like c^2, so a common mode
    # far larger than the signal costs digits to cancellation. The test
    # therefore does two things: it checks the invariance tightly at a realistic
    # common mode (comparable to Delta itself, which is the actual situation),
    # and it checks that the residual GROWS WITH the shift -- which is what
    # distinguishes cancellation from a genuine asymmetry.
    print("\n--- shift invariance " + "-" * 44)
    S = 8
    lam, targets, deltas = _synth(rng, N, V, S, b_vec, L)
    KEYS = ("D", "Q", "V", "bt_rms", "sdt_rms", "coord_skew",
            "bias_logq_y_mean", "ep_bias_logq_mean", "varq_y_mean")
    tmpA = os.path.join(tmpbase, "shiftA"); os.makedirs(tmpA, exist_ok=True)
    # float64 on purpose: at the production float32 a shift of size c costs
    # ~c*6e-8 of absolute storage precision, which would swamp the effect under
    # test. The invariance itself does not depend on the dtype.
    refA = _write_tree(tmpA, lam, targets, deltas, dtype=np.float64)
    scA = logit_scales(np.load(os.path.join(refA, "logits.npy"), mmap_mode="r"))
    a = cell_stats(os.path.join(tmpA, "lm_head", "t07", "sr"), refA, "sr", 7,
                   jack=False, scales=scA)["scalars"]
    resid = {}
    # D, bias_logq and varq come out EXACTLY invariant (residual 0.0): they are
    # linear in the seeds or formed through log q - log p = Delta - c, where the
    # common mode cancels symbolically. Q, V, sdt_rms and coord_skew go through
    # the centered second moment, which is a difference of parts growing like
    # c^2, so they carry ~1e-6 of relative cancellation on quantities that are
    # themselves small differences. That is immaterial next to the ~1e-3
    # relative seed standard errors these terms actually carry.
    for scale, tol in ((0.5, 1e-6), (50.0, None)):
        tmpB = os.path.join(tmpbase, f"shiftB{scale}")
        os.makedirs(tmpB, exist_ok=True)
        sh = rng.normal(0.0, scale, size=N)
        refB = _write_tree(tmpB, lam, targets, [d + sh[:, None] for d in deltas],
                           dtype=np.float64)
        b_ = cell_stats(os.path.join(tmpB, "lm_head", "t07", "sr"), refB, "sr", 7,
                        jack=False, scales=scA)["scalars"]
        rs = []
        for key in KEYS:
            r = abs(a[key] - b_[key]) / max(1e-300, abs(a[key]))
            rs.append(r)
            if tol is not None:
                check(f"shift(c~{scale}) invariant {key}", a[key], b_[key],
                      tol, True)
        resid[scale] = max(rs)
        moved = abs(a["b_rms"] - b_["b_rms"]) / abs(a["b_rms"])
        ok = moved > 1e-3
        print(f"  {'ok  ' if ok else 'FAIL'} shift(c~{scale}) b_rms DOES move"
              f"{'':21s} rel {moved:.3e} > 1e-3")
        if not ok:
            fails.append(f"shift test vacuous at c~{scale}")
    grew = resid[50.0] > 10.0 * resid[0.5]
    print(f"  {'ok  ' if grew else 'FAIL'} residual scales with the shift"
          f"{'':17s} {resid[0.5]:.2e} -> {resid[50.0]:.2e} (cancellation, "
          f"not asymmetry)")
    if not grew:
        fails.append("residual did not scale with the shift")

    # (10) known non-Gaussian shapes.
    #
    # lam_sd is deliberately tiny and V large, so p is near-uniform and the
    # p-centering subtracts a mean over V ~ iid draws (sd 1/sqrt(V) = 3%)
    # instead of one dominant coordinate. Only then does the *centered*
    # variable this script reports share the shape of the raw law, so only then
    # is a comparison against the analytic value meaningful. With a peaked p the
    # centered variable is genuinely less skewed than the raw one -- that is a
    # property of Delta_tilde and not an estimator error.
    print("\n--- known shapes (S = 32, near-uniform p) " + "-" * 23)
    for dist, want_sk, want_ek in (("normal", 0.0, 0.0),
                                   ("rademacher", 0.0, -2.0),
                                   ("uniform", 0.0, -1.2),
                                   ("exponential", 2.0, 6.0)):
        lam, targets, deltas = _synth(rng, 8, 1024, 32, np.zeros(1024),
                                      np.eye(1024), dist=dist, lam_sd=1e-3)
        tmp = os.path.join(tmpbase, f"shape-{dist}"); os.makedirs(tmp, exist_ok=True)
        r = _write_tree(tmp, lam, targets, deltas)
        sca = logit_scales(np.load(os.path.join(r, "logits.npy"), mmap_mode="r"))
        sc = cell_stats(os.path.join(tmp, "lm_head", "t07", "sr"), r, "sr", 7,
                        jack=False, scales=sca)["scalars"]

        # An independent, unstreamed computation of the same two estimators on
        # the same p-centered deviations. This is what validates the code.
        dd = np.stack([np.load(os.path.join(tmp, "lm_head", "t07", "sr",
                                            f"seed{i}", "delta.npy"))
                       .astype(np.float64) for i in range(1, 33)])
        pp = np.exp(log_softmax_np(np.load(os.path.join(r, "logits.npy"))
                                   .astype(np.float64)))
        dv = dd - dd.mean(axis=0)
        dvt = dv - np.einsum("snj,nj->sn", dv, pp)[:, :, None]
        k = 32
        m2 = (dvt ** 2).mean(axis=0)
        m3 = (dvt ** 3).mean(axis=0)
        m4 = (dvt ** 4).mean(axis=0)
        g1 = m3 / m2 ** 1.5
        G1 = math.sqrt(k * (k - 1.0)) / (k - 2.0) * g1
        g2 = m4 / (m2 * m2) - 3.0
        G2 = (k - 1.0) / ((k - 2.0) * (k - 3.0)) * ((k + 1.0) * g2 + 6.0)
        check(f"{dist:12s} coord_skew vs direct", sc["coord_skew"],
              float(G1.mean()), 1e-9, True)
        check(f"{dist:12s} coord_exkurt vs direct", sc["coord_exkurt"],
              float(G2.mean()), 1e-9, True)

        # Against the ANALYTIC value, only where Fisher's corrections are
        # unbiased: they are derived under normality, so at k = 32 a strongly
        # non-Gaussian law still reads low. Assert the direction is
        # unambiguous there instead of a value the estimator cannot deliver.
        if dist == "normal":
            check(f"{dist:12s} coord_skew ~ analytic", sc["coord_skew"],
                  want_sk, 0.05)
            check(f"{dist:12s} coord_exkurt ~ analytic", sc["coord_exkurt"],
                  want_ek, 0.10)
        else:
            got_sk, got_ek = sc["coord_skew"], sc["coord_exkurt"]
            ok = (abs(got_sk) < 0.05 if want_sk == 0.0
                  else got_sk > 0.5 * want_sk)
            ok2 = (got_ek < 0.5 * want_ek if want_ek < 0
                   else got_ek > 0.3 * want_ek)
            print(f"  {'ok  ' if ok and ok2 else 'FAIL'} {dist:12s} direction "
                  f"(analytic {want_sk:+.1f}/{want_ek:+.1f}): "
                  f"skew {got_sk:+.3f}, exkurt {got_ek:+.3f}"
                  f"  [biased low at k=32, by construction]")
            if not (ok and ok2):
                fails.append(f"{dist} shape direction")

    # (18) RN with S > 1: bitwise identical, and still no fluctuation
    print("\n--- RN multi-seed " + "-" * 47)
    lam, targets, deltas = _synth(rng, N, V, 1, b_vec, L)
    tmp = os.path.join(tmpbase, "rn3"); os.makedirs(tmp, exist_ok=True)
    r = _write_tree(tmp, lam, targets, [deltas[0]] * 3, mode="rn")
    sca = logit_scales(np.load(os.path.join(r, "logits.npy"), mmap_mode="r"))
    res = cell_stats(os.path.join(tmp, "lm_head", "t07", "rn"), r, "rn", 7,
                     jack=False, scales=sca)
    ok = res["rn_bitwise"] is True and res["k_eff"] == 1 \
        and res["scalars"].get("sigma2_mean") is None
    print(f"  {'ok  ' if ok else 'FAIL'} RN x3: bitwise={res['rn_bitwise']}, "
          f"k_eff={res['k_eff']}, variance absent="
          f"{res['scalars'].get('sigma2_mean') is None}")
    if not ok:
        fails.append("RN multi-seed handling")

    print()
    if fails:
        print(f"SELFTEST FAILED: {len(fails)} check(s)")
        for f in fails[:20]:
            print("  -", f)
        raise SystemExit(1)
    print("all checks passed")
    print("Result -> Cells: 0 | Selftest: pass | Time: 0.00s")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def analysis_flags(p):
        p.add_argument("--budget-bytes", type=int, default=1_000_000_000,
                       help="resident-memory target for the chunk loop")
        p.add_argument("--no-jackknife", action="store_true")
        p.add_argument("--no-softmax", action="store_true")
        p.add_argument("--no-hist", action="store_true")
        p.add_argument("--no-gram", action="store_true",
                       help="skip the read-only cross-check against a cached "
                            "gram.npz")
        p.add_argument("--max-seeds", type=int, default=None,
                       help="use only the first K seeds, for a seed-budget check")
        p.add_argument("--min-var", type=float, default=0.0,
                       help="coordinates with centered variance <= this are "
                            "masked out of the shape statistics")
        p.add_argument("--strict", action="store_true",
                       help="warn when Delta is stored narrower than float32")
        # Deliberately NOT --out-dir: see the module docstring.
        p.add_argument("--per-token-dir", default=None)
        p.add_argument("--hist-dir", default=None)

    c = sub.add_parser("cell", help="one cell -> one CSV row")
    c.add_argument("--root", required=True)
    c.add_argument("--site", required=True)
    c.add_argument("--t", type=int, required=True)
    c.add_argument("--mode", required=True, choices=["sr", "rn"])
    c.add_argument("--cell-csv", required=True)
    analysis_flags(c)
    c.set_defaults(func=cmd_cell)

    r = sub.add_parser("report", help="every discovered cell under a root")
    r.add_argument("--root", required=True)
    r.add_argument("--sites", nargs="*", default=None)
    r.add_argument("--t", nargs="*", type=int, default=None)
    r.add_argument("--modes", nargs="*", default=None)
    r.add_argument("--csv", default=None)
    r.add_argument("--long-csv", default=None)
    analysis_flags(r)
    r.set_defaults(func=cmd_report)

    m = sub.add_parser("merge", help="per-cell CSVs -> one tidy CSV")
    m.add_argument("--cells", required=True, help="glob")
    m.add_argument("--csv", required=True)
    m.add_argument("--long-csv", default=None)
    m.set_defaults(func=cmd_merge)

    s = sub.add_parser("selftest", help="synthetic, known b and Sigma")
    s.add_argument("--dir", default=None)
    s.set_defaults(func=cmd_selftest)
    return ap


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
