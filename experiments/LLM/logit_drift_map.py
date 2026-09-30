#!/usr/bin/env python3
"""Per-logit expected perturbation b_hat[r,j] and where it cancels.

The manuscript reports a pooled sample drift whose mean is statistically
indistinguishable from zero at the head.  A pooled mean over N*V token-
coordinate pairs can hide structure: individual logits may drift, with the
drift cancelling across coordinates within a token, across tokens at a fixed
coordinate, or both.  This script reduces the raw captures to the quantities
needed to see that, without ever materialising b_hat outside the cluster.

b_hat[r,j] = mean_s Delta[r,j]^(s) over S seeds; its per-element standard
error se[r,j] = sd_s(Delta[r,j])/sqrt(S) sets the finite-seed resolution.
Pooling levels reported, each with an exact seed-level standard error
(seed means are the independent replicates; coordinates within a seed are
not independent, so the SE is always taken over seeds):

  level 0  individual logits        b_hat[r,j]
  level 1a per token, over coords   m_r = mean_j b_hat[r,j]
  level 1b per coordinate, over toks c_j = mean_r b_hat[r,j]
  level 2  pooled                   mean_{r,j} b_hat[r,j]

For each level the RMS of the estimate is reported next to the RMS the same
estimator would show under pure seed noise (drift exactly zero), so the two
can be compared directly: excess of the former over the latter is resolved
drift, equality is noise.

Coordinates are also grouped by the reference probability p_r,j, since the
loss sees only p-weighted contrasts: a drift that is uniform in j is
invisible to the softmax, one that is anticorrelated with p is not.

Usage: python3 logit_drift_map.py <site> <t> <S> [--out DIR] [--root DIR]
writes <out>/logit_drift_<site>_t<t>.npz (SR, S seeds, plus the RN run).
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.environ.get("HEAD_DECOMP_ROOT", "head_decomposition/256/tok1024")
PBIN_EDGES = np.array([-np.inf, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, np.inf])
Z_EDGES = np.linspace(-8.0, 8.0, 161)
SUBSAMPLE = 200_000
QUANTS = np.array([0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def load_reference(root):
    ref = np.load(f"{root}/reference/logits.npy").astype(np.float64)
    z = np.load(f"{root}/reference/scalars.npz")
    targets = z["targets"].astype(np.int64)
    p = ref - ref.max(axis=1, keepdims=True)
    np.exp(p, out=p)
    p /= p.sum(axis=1, keepdims=True)
    return p, targets


def pooling_levels(b, se, p, targets, pbin, n_pbin):
    """Signed means at each pooling level, plus the matching noise scales."""
    N, V = b.shape
    out = {}
    out["token_mean"] = b.mean(axis=1)
    out["token_rms"] = np.sqrt((b * b).mean(axis=1))
    out["coord_mean"] = b.mean(axis=0)
    out["coord_rms"] = np.sqrt((b * b).mean(axis=0))
    out["pooled_mean"] = np.array(b.mean())
    out["pooled_rms"] = np.array(np.sqrt((b * b).mean()))
    out["pooled_absmean"] = np.array(np.abs(b).mean())
    # p-weighted (loss-visible) per-token drift and the target coordinate
    out["token_pmean"] = np.einsum("nv,nv->n", p, b)
    out["token_target"] = b[np.arange(N), targets]
    # coordinates grouped by reference probability
    pb_sum = np.zeros(n_pbin)
    pb_abs = np.zeros(n_pbin)
    pb_sq = np.zeros(n_pbin)
    pb_cnt = np.zeros(n_pbin)
    pb_psum = np.zeros(n_pbin)     # sum of p over the bin (its share of softmax mass)
    pb_pb = np.zeros(n_pbin)       # sum of p*b_hat: the bin's share of p^T b_hat
    for k in range(n_pbin):
        mask = pbin == k
        if not mask.any():
            continue
        vals = b[mask]
        pvals = p[mask]
        pb_sum[k] = vals.sum()
        pb_abs[k] = np.abs(vals).sum()
        pb_sq[k] = (vals * vals).sum()
        pb_cnt[k] = vals.size
        pb_psum[k] = pvals.sum()
        pb_pb[k] = (pvals * vals).sum()
    out["pbin_sum"] = pb_sum
    out["pbin_abs_sum"] = pb_abs
    out["pbin_sq_sum"] = pb_sq
    out["pbin_count"] = pb_cnt
    out["pbin_p_sum"] = pb_psum
    out["pbin_pb_sum"] = pb_pb
    if se is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(se > 0, b / se, 0.0)
        out["z_hist"] = np.histogram(z, bins=Z_EDGES)[0]
        out["z_absmax"] = np.array(np.abs(z).max())
        out["z_frac_gt2"] = np.array((np.abs(z) > 2.0).mean())
        out["z_frac_gt3"] = np.array((np.abs(z) > 3.0).mean())
        out["z_quantiles"] = np.quantile(z, QUANTS)
        # noise-only RMS of each pooled estimator: E[mean of noise]^2 = mean(se^2)/n
        out["se_rms"] = np.array(np.sqrt((se * se).mean()))
        out["token_noise_rms"] = np.sqrt((se * se).mean(axis=1) / V)
        out["coord_noise_rms"] = np.sqrt((se * se).mean(axis=0) / N)
        out["pooled_noise_rms"] = np.array(np.sqrt((se * se).mean() / (N * V)))
    return out


def run(site, t, seeds, root, out_dir):
    p, targets = load_reference(root)
    N, V = p.shape
    log(f"reference {N}x{V}")
    with np.errstate(divide="ignore"):
        logp = np.log10(np.maximum(p, 1e-300))
    pbin = np.digitize(logp, PBIN_EDGES[1:-1])
    n_pbin = len(PBIN_EDGES) - 1

    S = len(seeds)
    S1 = np.zeros((N, V), dtype=np.float64)
    S2 = np.zeros((N, V), dtype=np.float64)
    seed_token_mean = np.zeros((N, S))
    seed_coord_mean = np.zeros((V, S))
    seed_grand = np.zeros(S)
    seed_pbin_mean = np.zeros((n_pbin, S))
    pbin_flat = pbin.ravel()
    pbin_count = np.bincount(pbin_flat, minlength=n_pbin).astype(np.float64)
    for si, seed in enumerate(seeds):
        d = np.load(f"{root}/{site}/t{t:02d}/sr/seed{seed}/delta.npy").astype(np.float64)
        S1 += d
        S2 += d * d
        seed_token_mean[:, si] = d.mean(axis=1)
        seed_coord_mean[:, si] = d.mean(axis=0)
        seed_grand[si] = d.mean()
        seed_pbin_mean[:, si] = (np.bincount(pbin_flat, weights=d.ravel(),
                                             minlength=n_pbin)
                                 / np.maximum(pbin_count, 1.0))
        del d
        log(f"  seed {seed} done")

    b = S1 / S
    var = (S2 - S1 * S1 / S) / (S - 1)
    np.maximum(var, 0.0, out=var)
    se = np.sqrt(var / S)
    del S2, var

    res = {f"sr_{k}": v for k, v in
           pooling_levels(b, se, p, targets, pbin, n_pbin).items()}
    # exact seed-level standard errors for the pooled estimators
    res["sr_token_mean_se"] = seed_token_mean.std(axis=1, ddof=1) / np.sqrt(S)
    res["sr_coord_mean_se"] = seed_coord_mean.std(axis=1, ddof=1) / np.sqrt(S)
    res["sr_pooled_mean_se"] = np.array(seed_grand.std(ddof=1) / np.sqrt(S))
    res["sr_pbin_mean_se"] = seed_pbin_mean.std(axis=1, ddof=1) / np.sqrt(S)
    res["sr_seed_pbin_mean"] = seed_pbin_mean
    res["sr_seed_grand"] = seed_grand
    rng = np.random.default_rng(20260914)
    idx = rng.choice(N * V, size=min(SUBSAMPLE, N * V), replace=False)
    res["sr_sub_b"] = b.ravel()[idx].astype(np.float32)
    res["sr_sub_se"] = se.ravel()[idx].astype(np.float32)
    res["sr_sub_logp"] = logp.ravel()[idx].astype(np.float32)
    res["sr_b_quantiles"] = np.quantile(b, QUANTS)
    del b, se

    drn = None
    for rn_path in (f"{root}/{site}/t{t:02d}/rn/seed1/delta.npy",
                    f"{root}/{site}/t{t:02d}/rn/delta.npy"):
        try:
            drn = np.load(rn_path).astype(np.float64)
            break
        except FileNotFoundError:
            continue
    if drn is None:
        log(f"  no RN capture under {root}/{site}/t{t:02d}/rn")
    if drn is not None:
        res.update({f"rn_{k}": v for k, v in
                    pooling_levels(drn, None, p, targets, pbin, n_pbin).items()})
        res["rn_sub_b"] = drn.ravel()[idx].astype(np.float32)
        del drn
        log("  rn done")

    res["site"] = np.array(site)
    res["t"] = np.array(t)
    res["seeds"] = np.array(seeds)
    res["n_scored"] = np.array(N)
    res["vocab"] = np.array(V)
    res["pbin_edges"] = PBIN_EDGES
    res["z_edges"] = Z_EDGES
    res["quantiles"] = QUANTS
    res["sub_index"] = idx.astype(np.int64)
    res["root"] = np.array(root)

    path = f"{out_dir}/logit_drift_{site}_t{t}.npz"
    np.savez_compressed(path, **res)
    log(f"wrote {path}")
    summary = {
        "site": site, "t": t, "S": S,
        "pooled_mean": float(res["sr_pooled_mean"]),
        "pooled_mean_se": float(res["sr_pooled_mean_se"]),
        "pooled_rms": float(res["sr_pooled_rms"]),
        "pooled_noise_rms": float(res["sr_pooled_noise_rms"]),
        "se_rms": float(res["sr_se_rms"]),
        "token_mean_rms": float(np.sqrt((res["sr_token_mean"] ** 2).mean())),
        "token_noise_rms": float(np.sqrt((res["sr_token_noise_rms"] ** 2).mean())),
        "coord_mean_rms": float(np.sqrt((res["sr_coord_mean"] ** 2).mean())),
        "coord_noise_rms": float(np.sqrt((res["sr_coord_noise_rms"] ** 2).mean())),
        "z_frac_gt2": float(res["sr_z_frac_gt2"]),
        "z_frac_gt3": float(res["sr_z_frac_gt3"]),
        "rn_pooled_mean": float(res["rn_pooled_mean"]) if "rn_pooled_mean" in res else None,
        "rn_pooled_rms": float(res["rn_pooled_rms"]) if "rn_pooled_rms" in res else None,
    }
    print(json.dumps(summary))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("site")
    ap.add_argument("t", type=int)
    ap.add_argument("S", type=int)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    run(args.site, args.t, list(range(1, args.S + 1)), args.root, args.out)


if __name__ == "__main__":
    main()
