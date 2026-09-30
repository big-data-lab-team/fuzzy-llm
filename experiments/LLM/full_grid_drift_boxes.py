#!/usr/bin/env python3
"""Compute exact full-grid box summaries for the drift--fluctuation figure.

Run where the 32 seed-level delta.npy and scalars.npz captures are stored.
Each output contains six-number summaries (1st, 25th, 50th, 75th, 99th
percentiles, mean) over all N*V token--coordinate pairs. No pairs are sampled.
"""
import argparse
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('HEAD_DECOMP_ROOT', 'head_decomposition/256/tok1024'))
QUANTILES = [0.01, 0.25, 0.50, 0.75, 0.99]


def box(values):
    return np.r_[np.quantile(values, QUANTILES), values.mean()]


def run(site, t, seeds, root, out):
    base = root / site / f't{t:02d}'
    first = np.load(base / 'sr' / f'seed{seeds[0]}' / 'delta.npy', mmap_mode='r')
    shape = first.shape
    del first
    n, vocab = shape
    sums = np.zeros(shape, dtype=np.float64)
    squares = np.zeros(shape, dtype=np.float64)
    centered_sums = np.zeros(shape, dtype=np.float64)
    centered_squares = np.zeros(shape, dtype=np.float64)

    for seed in seeds:
        seed_dir = base / 'sr' / f'seed{seed}'
        delta = np.load(seed_dir / 'delta.npy').astype(np.float64)
        if delta.shape != shape:
            raise ValueError(f'{seed_dir}: shape {delta.shape}, expected {shape}')
        with np.load(seed_dir / 'scalars.npz') as z:
            ep_delta = z['Ep_d'].astype(np.float64)
        if ep_delta.shape != (n,):
            raise ValueError(f'{seed_dir}: Ep_d shape {ep_delta.shape}, expected {(n,)}')
        sums += delta
        squares += delta * delta
        delta -= ep_delta[:, None]
        centered_sums += delta
        centered_squares += delta * delta
        print(f'{site} t={t}: seed {seed}/{seeds[-1]}', flush=True)

    s = len(seeds)
    drift = sums / s
    sr_rms_box = box(np.sqrt(squares / s))
    sr_drift_box = box(np.abs(drift))
    sr_cent_box = box(drift - drift.mean(axis=1, keepdims=True))
    centered_variance = (centered_squares - centered_sums * centered_sums / s) / (s - 1)
    np.maximum(centered_variance, 0.0, out=centered_variance)
    sr_fluc_box = box(np.sqrt(centered_variance))

    rn_path = base / 'rn' / 'seed1' / 'delta.npy'
    if not rn_path.exists():
        rn_path = base / 'rn' / 'delta.npy'
    rn = np.load(rn_path).astype(np.float64)
    if rn.shape != shape:
        raise ValueError(f'{rn_path}: shape {rn.shape}, expected {shape}')
    rn_rms_box = box(np.abs(rn))
    rn_cent_box = box(rn - rn.mean(axis=1, keepdims=True))

    out.mkdir(parents=True, exist_ok=True)
    path = out / f'full_grid_drift_{site}_t{t}.npz'
    np.savez_compressed(path, site=site, t=t, seeds=np.asarray(seeds),
                        n_scored=n, vocab=vocab, n_points=n * vocab,
                        sr_rms_box=sr_rms_box, rn_rms_box=rn_rms_box,
                        sr_drift_box=sr_drift_box,
                        sr_cent_box=sr_cent_box, rn_cent_box=rn_cent_box,
                        sr_fluc_box=sr_fluc_box)
    print(f'wrote {path} ({n * vocab:,} pairs)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('site', choices=['lm_head', 'mlp_c_proj'])
    parser.add_argument('t', type=int)
    parser.add_argument('--seeds', type=int, default=32)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, default=Path('.'))
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error('--seeds must be at least 2')
    run(args.site, args.t, list(range(1, args.seeds + 1)), args.root, args.out)


if __name__ == '__main__':
    main()
