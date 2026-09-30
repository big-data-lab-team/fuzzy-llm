#!/usr/bin/env python3
"""Per-coordinate seed-std of the p-centered logit fluctuation, for
Figure 4 row (iii) ("centered std(xi_SR)").

Companion to logit_drift_map.py (rows i/ii), run on the cluster where
delta.npy lives. Row (iii) is currently a placeholder in
scripts/plot_drift_3row.py: a single sqrt(2*V_SR) marker per (site, t),
pending "full per-coordinate re-extraction from cluster". This script
supplies that re-extraction.

Centering: each seed capture's scalars.npz already stores
Ep_d[r] = p[r]^T Delta[r,:] (verified by direct comparison against the
reference softmax to ~1e-7, i.e. float32 roundoff). That is exactly the
per-token uniform-shift component the softmax cannot see, and exactly
what the Appendix D identity (H = diag(p) - p p^T annihilates the
uniform part) is stated on. Centering by Ep_d -- not by an unweighted
vocab mean -- is what makes the pooled p-weighted RMS of the result
equal sqrt(2*V_SR) from tab:decomposition, so the new boxplot data and
the old placeholder marker are the same quantity at two granularities
(full distribution vs. a single summary scalar), not two different
statistics.

Delta_tilde[r,j,s] = Delta[r,j,s] - Ep_d[r,s]
sigma_tilde[r,j]   = std_s(Delta_tilde[r,j,s])   (ddof=1, S seeds)

Streaming two-accumulator variance (S1, S2 running sum / sum-of-squares
over seeds, shape (N,V) float64) avoids holding all S delta.npy arrays
in memory at once, matching logit_drift_map.py's memory profile.

For a sanity check against the existing V_SR table, this script also
computes the exact pooled quantity 2*V = mean_r sum_j p[r,j]*sigma_tilde
[r,j]^2 over the *full* (N,V) grid (not the subsample), using the
reference softmax -- this should match tab:decomposition's V_SR to
close precision, since it is algebraically the same identity
distributions.py uses (mean_s x^2 - xbar^2 = ((k-1)/k) var_ddof1).

Usage: python3 logit_fluct_map.py <site> <t> <S> [--out DIR] [--root DIR]
       [--drift-dir DIR]
reads <drift-dir>/logit_drift_<site>_t<t>.npz for n_scored/vocab/sub_index
(so the row-3 boxplot samples the identical (token, coordinate) pairs as
rows 1-2), and writes <out>/logit_fluct_<site>_t<t>.npz.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.environ.get("HEAD_DECOMP_ROOT", "head_decomposition/256/tok1024")
DRIFT_DIR = os.environ.get("DRIFT_DIR", "logit_drift")


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def load_reference(root):
    ref = np.load(f"{root}/reference/logits.npy").astype(np.float64)
    p = ref - ref.max(axis=1, keepdims=True)
    np.exp(p, out=p)
    p /= p.sum(axis=1, keepdims=True)
    return p


def run(site, t, seeds, root, out_dir, drift_dir):
    with np.load(f"{drift_dir}/logit_drift_{site}_t{t}.npz") as z:
        n_scored = int(z["n_scored"])
        vocab = int(z["vocab"])
        sub_index = z["sub_index"]

    S = len(seeds)
    N, V = n_scored, vocab
    S1 = np.zeros((N, V), dtype=np.float64)
    S2 = np.zeros((N, V), dtype=np.float64)
    for seed in seeds:
        base = f"{root}/{site}/t{t:02d}/sr/seed{seed}"
        d = np.load(f"{base}/delta.npy").astype(np.float64)
        if d.shape != (N, V):
            raise ValueError(f"{base}/delta.npy: shape {d.shape}, expected {(N, V)}")
        with np.load(f"{base}/scalars.npz") as z:
            ep_d = z["Ep_d"].astype(np.float64)
        centered = d - ep_d[:, None]
        S1 += centered
        S2 += centered * centered
        del d, centered
        log(f"  seed {seed} done")

    var = (S2 - S1 * S1 / S) / (S - 1)
    np.maximum(var, 0.0, out=var)
    del S1, S2
    std = np.sqrt(var)

    # Full-grid pooled cross-check against tab:decomposition's V_SR.
    p = load_reference(root)
    two_v_per_token = np.einsum("rv,rv->r", p, var)
    two_v = float(two_v_per_token.mean())
    del p

    idx = sub_index.astype(np.int64)
    sr_fluc_sub = std.ravel()[idx].astype(np.float32)
    del std, var

    res = {
        "site": np.array(site),
        "t": np.array(t),
        "seeds": np.array(seeds),
        "n_scored": np.array(N),
        "vocab": np.array(V),
        "sub_index": idx,
        "root": np.array(root),
        "sr_fluc_sub": sr_fluc_sub,
        "sr_fluc_pooled_rms": np.array(float(np.sqrt(np.mean(sr_fluc_sub.astype(np.float64) ** 2)))),
        "sr_fluc_two_v": np.array(two_v),
        "sr_fluc_sqrt_two_v": np.array(float(np.sqrt(max(two_v, 0.0)))),
    }

    path = f"{out_dir}/logit_fluct_{site}_t{t}.npz"
    np.savez_compressed(path, **res)
    log(f"wrote {path}")

    summary = {
        "site": site, "t": t, "S": S,
        "sub_rms": float(res["sr_fluc_pooled_rms"]),
        "two_v": two_v,
        "sqrt_two_v": float(res["sr_fluc_sqrt_two_v"]),
    }
    print(json.dumps(summary))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("site")
    ap.add_argument("t", type=int)
    ap.add_argument("S", type=int)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=".")
    ap.add_argument("--drift-dir", default=DRIFT_DIR)
    args = ap.parse_args()
    run(args.site, args.t, list(range(1, args.S + 1)), args.root, args.out, args.drift_dir)


if __name__ == "__main__":
    main()
