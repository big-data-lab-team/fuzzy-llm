#!/usr/bin/env python3
"""Per-token tr(H_r), D/Q/V_r, and the W/C cancellation behind V_r.

H_r = diag(p_r) - p_r p_r^T at each scored token r; tr(H_r) = 1 - sum_j p_r,j^2
(the "collision probability" complement, computed from the reference logits
alone, so it is the same at every site and precision for a given token).

Streams delta.npy one seed at a time (each [N,V] float32, ~205MB) rather than
holding all S seeds in memory. D_r, Q_r, V_r use the same bias-corrected
finite-sample estimators as distributions.py (A, B, S/(S-1) correction);
averaged over tokens they reproduce results/quadratic_loss_summary.csv and
results/distributions_s32.csv to the precision reported there.

W_r := sum_j p_r,j Sigma_jj(r) (p-weighted raw per-coordinate seed variance)
and C_r := Var_s(p_r^T Delta_r) (variance of the scalar the loss actually
projects onto) satisfy 2*V_r = W_r - C_r exactly -- checked numerically below
to machine precision before trusting any correlation built from them. A
naive per-token summary that reports W_r (or sqrt(W_r)) alone without C_r
can be badly misleading: at some sites C_r cancels nearly all of W_r, so
V_r behaves like a small residual of two much larger, nearly equal
quantities rather than tracking W_r's own trend.

Usage: python3 token_fisher_analysis.py <site> <t> <S> [--root DIR] [--out DIR]
writes <out>/token_fisher_<site>_t<t>.csv with columns trH,D,Q,V,sigma,W,C.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.environ.get("HEAD_DECOMP_ROOT", "head_decomposition/256/tok1024")


def run(site, t, seeds, root=ROOT, out_dir="."):
    ref = np.load(f"{root}/reference/logits.npy").astype(np.float64)
    z = np.load(f"{root}/reference/scalars.npz")
    targets = z["targets"].astype(np.int64)
    N, V = ref.shape
    p = ref - ref.max(axis=1, keepdims=True)
    np.exp(p, out=p)
    p /= p.sum(axis=1, keepdims=True)
    trH = 1.0 - (p * p).sum(axis=1)

    S1 = np.zeros((N, V), dtype=np.float64)
    S2 = np.zeros((N, V), dtype=np.float64)
    Tproj = np.zeros((N, len(seeds)), dtype=np.float64)

    for si, seed in enumerate(seeds):
        d = np.load(f"{root}/{site}/t{t:02d}/sr/seed{seed}/delta.npy").astype(np.float64)
        S1 += d
        S2 += d * d
        Tproj[:, si] = np.einsum("nv,nv->n", p, d)
        del d
        print(f"  seed {seed} done", file=sys.stderr)

    S = len(seeds)
    corr = S / (S - 1)
    b_hat = S1 / S
    m_r = np.einsum("nv,nv->n", p, b_hat)            # p . b_hat
    D_r = m_r - b_hat[np.arange(N), targets]

    p_bhat_sq = np.einsum("nv,nv->n", p, b_hat * b_hat)
    varp_bhat = p_bhat_sq - m_r ** 2
    B_r = 0.5 * varp_bhat

    mean_s_sq_pdelta = (Tproj ** 2).mean(axis=1)
    sum_p_meanS2 = np.einsum("nv,nv->n", p, S2 / S)
    A_r = 0.5 * (sum_p_meanS2 - mean_s_sq_pdelta)

    Q_r = (S * B_r - A_r) / (S - 1)
    V_r = S * (A_r - B_r) / (S - 1)

    W_r = corr * (sum_p_meanS2 - p_bhat_sq)          # bias-corrected sum_j p_j Sigma_jj
    C_r = corr * (mean_s_sq_pdelta - m_r ** 2)        # bias-corrected Var_s(p.Delta)
    check = np.abs((W_r - C_r) / 2 - V_r)
    print(f"  identity check max|((W-C)/2)-V| = {check.max():.3e}", file=sys.stderr)

    sigma_r = np.sqrt(np.maximum(W_r, 0.0))          # = sqrt(W_r); kept for continuity

    out = np.stack([trH, D_r, Q_r, V_r, sigma_r, W_r, C_r], axis=1)
    path = os.path.join(out_dir, f"token_fisher_{site}_t{t}.csv")
    np.savetxt(path, out, delimiter=",", header="trH,D,Q,V,sigma,W,C", comments="")
    print(f"wrote {path}", file=sys.stderr)
    print(f"  mean W={W_r.mean():.6g} mean C={C_r.mean():.6g} "
          f"C/W(agg)={C_r.mean()/W_r.mean():.6f} mean(C/W)={ (C_r/W_r).mean():.6f}",
          file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Per-token tr(H_r), D/Q/V_r and W/C for one cell.")
    ap.add_argument("site")
    ap.add_argument("t", type=int)
    ap.add_argument("S", type=int, help="number of SR seeds (1..S)")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    run(args.site, args.t, list(range(1, args.S + 1)), args.root, args.out)
