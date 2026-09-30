"""Combine the per-cell captures of ``capture_logits.py`` into the four terms of

    E_m[dL] = D_m + Q_m + V_m + R_m                         (eq:head-full-comparison)

per token, and report their token averages D, Q, V, R.

The sufficient statistic
------------------------
Three of the four terms are p-weighted quadratic forms of vectors that differ
only in which realizations they average over, so a single S x S matrix per token
determines all of them at *every* seed subset at once. With
Delta^(s) = b + xi^(s) and

    G_{s s'} := 1/2 Cov_p(Delta^(s), Delta^(s'))
              = 1/2 [ E_p(Delta^(s) Delta^(s')) - E_p(Delta^(s)) E_p(Delta^(s')) ],

a subset K of size k has mean vector b_K = mean_{s in K} Delta^(s) and

    A_K := 1/2 mean_{s in K} Var_p(Delta^(s)) = (1/k)  sum_{s in K}    G_{ss}
    B_K := 1/2 Var_p(b_K)                    = (1/k^2) sum_{s,s' in K} G_{s s'}

by bilinearity of Cov_p -- the same bilinearity eq:head-step-var-split uses.
So the whole seed-budget study of a cell is a sum over submatrices of one
[N, S, S] array; no vector has to be re-read to change the seed count.

The finite-seed correction, and why it is not optional
------------------------------------------------------
The estimators the decomposition is written with are *not* A_K and B_K.
Since E_m[xi] = 0 and the seeds are independent draws,

    E[A_K] = Q + V,        E[B_K] = Q + V/k,

so the naive Q-hat = B_K carries a bias of +V/k and the naive V-hat = A_K - B_K
one of -V/k. That bias is not a detail here: the hypothesis under test is that V
dominates, and at k = 5 a dominant V puts 20% of itself inside the very term it
is supposed to dominate. Inverting the two expectations gives the unbiased pair

    Q-hat = (k B_K - A_K) / (k - 1),      V-hat = k (A_K - B_K) / (k - 1),

which is what this script reports (the naive pair is reported alongside, since
their difference is the size of the effect). Note Q-hat + V-hat = A_K either
way, so R -- computed as a residual -- is identical under both, and the
correction only moves mass between Q and V.

RN is a single deterministic realization on this ensemble, so xi = 0, b = Delta,
V = 0 and Q = A exactly; there is no correction to make and none is applied.

Layout consumed
---------------
    <root>/reference/                       logits.npy scalars.npz meta.json
    <root>/<site>/t<NN>/<mode>/seed<S>/     delta.npy  scalars.npz meta.json
"""

import argparse
import itertools
import json
import math
import os
import re
import sys

import numpy as np

R_BOUND_CONST = math.sqrt(6.0) / 54.0   # eq:head-bias-cov, thm:app-compare

# Tokens per chunk in the float64 reduction below. [chunk, V] float64 at V=50257
# is 25 MB per temporary at 64, against 410 MB for the whole [N, V] tensor; the
# array job runs one task per core and the node's memory is the binding
# constraint, not this loop's speed.
CHUNK = 64


def log_softmax_np(x):
    """Row-wise log-softmax in float64, shift-stabilized.

    At t=4 the perturbed logits carry excursions of tens of nats, so the naive
    exp() underflows or overflows depending on the sign; the excess loss there
    is the one number the perturbative decomposition cannot predict and must
    therefore be measured without help from it.
    """
    m = x.max(axis=1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def reduce_reference(logits, targets):
    """Per-token reference quantities: the loss, and the collision probability
    ``sum_j p_j^2`` that eq:fluctuation-closed-form reads V_m through."""
    n = logits.shape[0]
    nll = np.empty(n, dtype=np.float64)
    collision = np.empty(n, dtype=np.float64)
    for i in range(0, n, CHUNK):
        sl = slice(i, min(i + CHUNK, n))
        lam = logits[sl].astype(np.float64)
        lsm = log_softmax_np(lam)
        p = np.exp(lsm)
        nll[sl] = -lsm[np.arange(lsm.shape[0]), targets[sl]]
        collision[sl] = (p * p).sum(axis=1)
    return nll, collision


def reduce_perturbation(ref_logits, pert_logits, targets, out_delta):
    """Stream over tokens, writing Delta and reducing it to per-token scalars.

    Everything is accumulated in float64 from the float32 logits. Delta is
    persisted at whatever dtype ``out_delta`` was opened with, but no derived
    scalar is ever read back from that file: the only quantity ``decompose.py``
    recomputes from the stored vectors is the cross-seed mean, so a narrower
    storage dtype costs precision on Q_m alone and on nothing else.
    """
    n = ref_logits.shape[0]
    cols = {k: np.empty(n, dtype=np.float64)
            for k in ("d_y", "Ep_d", "W", "dL", "l2", "nll_pert")}
    nonfinite = 0
    for i in range(0, n, CHUNK):
        sl = slice(i, min(i + CHUNK, n))
        rows = np.arange(sl.stop - sl.start)
        lam = ref_logits[sl].astype(np.float64)
        pert = pert_logits[sl].astype(np.float64)
        delta = pert - lam
        nonfinite += int((~np.isfinite(delta)).sum())

        out_delta[sl] = delta.astype(out_delta.dtype)

        lsm_ref = log_softmax_np(lam)
        p = np.exp(lsm_ref)
        lsm_pert = log_softmax_np(pert)

        y = targets[sl]
        ep_d = (p * delta).sum(axis=1)
        cols["d_y"][sl] = delta[rows, y]
        cols["Ep_d"][sl] = ep_d
        cols["W"][sl] = (p * delta * delta).sum(axis=1) - ep_d * ep_d
        cols["nll_pert"][sl] = -lsm_pert[rows, y]
        cols["dL"][sl] = -lsm_pert[rows, y] + lsm_ref[rows, y]
        cols["l2"][sl] = np.sqrt((delta * delta).sum(axis=1))
    return cols, nonfinite




def cell_seeds(cell_dir):
    seeds = []
    for name in sorted(os.listdir(cell_dir)):
        m = re.fullmatch(r"seed(\w+)", name)
        if m and os.path.exists(os.path.join(cell_dir, name, "delta.npy")):
            seeds.append(name)
    return seeds


def load_p_chunk(ref_logits, sl):
    lam = ref_logits[sl].astype(np.float64)
    lam -= lam.max(axis=1, keepdims=True)
    np.exp(lam, out=lam)
    lam /= lam.sum(axis=1, keepdims=True)
    return lam


def build_gram(cell_dir, ref_dir, budget_bytes=256_000_000):
    """[N, S, S] of 1/2 Cov_p(Delta^(s), Delta^(s')), streamed over tokens."""
    seeds = cell_seeds(cell_dir)
    if not seeds:
        raise SystemExit(f"no seed*/delta.npy under {cell_dir}")
    ref_logits = np.load(os.path.join(ref_dir, "logits.npy"), mmap_mode="r")
    deltas = [np.load(os.path.join(cell_dir, s, "delta.npy"), mmap_mode="r")
              for s in seeds]
    n, v = ref_logits.shape
    for s, d in zip(seeds, deltas):
        if d.shape != (n, v):
            raise SystemExit(f"{cell_dir}/{s}: delta is {d.shape}, reference is {(n, v)}")

    S = len(seeds)
    chunk = max(4, min(128, int(budget_bytes // (S * v * 8))))
    gram = np.empty((n, S, S), dtype=np.float64)
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        p = load_p_chunk(ref_logits, sl)                       # [c, V]
        blk = np.stack([np.asarray(d[sl], dtype=np.float64)
                        for d in deltas], axis=0)              # [S, c, V]
        for j in range(sl.stop - sl.start):
            m = blk[:, j, :]                                   # [S, V]
            mp = m * p[j]
            mean = mp.sum(axis=1)                              # [S] = E_p Delta^(s)
            gram[i + j] = 0.5 * (mp @ m.T - np.outer(mean, mean))
    return seeds, gram


def load_scalars(cell_dir, seeds):
    out = {}
    for key in ("d_y", "Ep_d", "dL", "l2", "W"):
        out[key] = np.stack([
            np.load(os.path.join(cell_dir, s, "scalars.npz"))[key] for s in seeds])
    return out


def terms(gram, scal, subset, deterministic):
    """Per-token D, Q, V, R for one seed subset. ``gram`` is [N, S, S]."""
    k = len(subset)
    idx = np.asarray(subset)
    sub = gram[:, idx[:, None], idx[None, :]]                  # [N, k, k]
    A = np.einsum("nii->n", sub) / k
    B = sub.sum(axis=(1, 2)) / (k * k)

    D = (scal["Ep_d"][idx] - scal["d_y"][idx]).mean(axis=0)
    dL = scal["dL"][idx].mean(axis=0)

    if deterministic or k == 1:
        Q, V = A, np.zeros_like(A)
        Q_naive, V_naive = A, np.zeros_like(A)
    else:
        Q_naive, V_naive = B, A - B
        Q = (k * B - A) / (k - 1)
        V = k * (A - B) / (k - 1)
    R = dL - D - A          # = dL - D - Q - V, identical for either split
    bound = R_BOUND_CONST * (scal["l2"][idx] ** 3).mean(axis=0)
    return dict(D=D, Q=Q, V=V, R=R, Q_naive=Q_naive, V_naive=V_naive,
                A=A, B=B, dL=dL, R_bound=bound)


def aggregate(t):
    return {k: float(np.mean(v)) for k, v in t.items()}


def jackknife(gram, scal, deterministic, keys=("D", "Q", "V", "R", "dL")):
    """Delete-one-seed standard errors on the aggregate terms.

    The seeds are the unit of replication, so the error bar that matters is
    over them, and the terms are nonlinear in the seed set -- Q and V both mix
    diagonal and off-diagonal Gram entries -- so there is no closed form to
    propagate. Leave-one-out is the right tool and costs S extra evaluations of
    `terms`, all off the cached Gram.

    Returns {} for a deterministic single-realization cell: RN has one seed and
    no sampling error to report, which is not the same as an error of zero.
    """
    S = gram.shape[1]
    if deterministic or S < 3:
        return {}
    est = []
    for i in range(S):
        subset = [j for j in range(S) if j != i]
        est.append(aggregate(terms(gram, scal, subset, deterministic)))
    out = {}
    for k in keys:
        a = np.array([e[k] for e in est])
        out["se_" + k] = float(np.sqrt((S - 1) / S * ((a - a.mean()) ** 2).sum()))
    return out


def discover(root):
    cells = []
    for site in sorted(os.listdir(root)):
        sdir = os.path.join(root, site)
        if site == "reference" or not os.path.isdir(sdir):
            continue
        for tname in sorted(os.listdir(sdir)):
            m = re.fullmatch(r"t(\d+)", tname)
            if not m:
                continue
            for mode in sorted(os.listdir(os.path.join(sdir, tname))):
                cdir = os.path.join(sdir, tname, mode)
                if os.path.isdir(cdir) and cell_seeds(cdir):
                    cells.append((site, int(m.group(1)), mode, cdir))
    return cells


def cmd_gram(args):
    seeds, gram = build_gram(args.cell_dir, args.ref)
    np.savez(os.path.join(args.cell_dir, "gram.npz"),
             gram=gram, seeds=np.array(seeds))
    print(f"wrote {args.cell_dir}/gram.npz  seeds={len(seeds)} tokens={gram.shape[0]}")


def get_gram(cell_dir, ref_dir, rebuild=False):
    """Load the cell's Gram, building and caching it if absent.

    Caching matters: the Gram is the only thing that touches the Delta vectors,
    and re-reading a 32-seed cell is 1.6 GB off shared storage. Once written, the report
    and the seed-budget table both run off a couple of megabytes.
    """
    path = os.path.join(cell_dir, "gram.npz")
    if os.path.exists(path) and not rebuild:
        with np.load(path) as z:
            cached_seeds = [str(s) for s in z["seeds"]]
            if cached_seeds == cell_seeds(cell_dir):
                return cached_seeds, z["gram"]
        # A report may be requested before an array finishes. Its cached Gram
        # must not silently exclude seeds that completed after that report.
        print(f"note: seed membership changed; rebuilding {path}", file=sys.stderr)
    seeds, gram = build_gram(cell_dir, ref_dir)
    try:
        np.savez(path, gram=gram, seeds=np.array(seeds))
    except OSError as exc:      # a read-only or full capture root is not fatal
        print(f"note: could not cache {path}: {exc}", file=sys.stderr)
    return seeds, gram


def cmd_report(args):
    ref_dir = os.path.join(args.root, "reference")
    ref_meta = json.load(open(os.path.join(ref_dir, "meta.json")))
    ref_scalars = np.load(os.path.join(ref_dir, "scalars.npz"))
    print(f"reference: PPL={ref_meta['perplexity']:.4f}  tokens={ref_meta['n_scored']}"
          f"  mean collision sum_j p_j^2 = {ref_scalars['collision'].mean():.4f}")

    rows = []
    for site, t, mode, cdir in discover(args.root):
        if args.sites and site not in args.sites:
            continue
        seeds, gram = get_gram(cdir, ref_dir, rebuild=args.rebuild_gram)
        scal = load_scalars(cdir, seeds)
        deterministic = (mode == "rn")
        per_token = terms(gram, scal, list(range(len(seeds))), deterministic)
        agg = aggregate(per_token)
        agg.update(jackknife(gram, scal, deterministic))
        agg.update(site=site, t=t, mode=mode, seeds=len(seeds))

        # Validation: the identity is exact by construction of R, so what is
        # worth checking is that the token-averaged excess loss still equals the
        # log perplexity ratio the sweep would report -- i.e. that nothing was
        # lost between the capture and here.
        metas = [json.load(open(os.path.join(cdir, s, "meta.json"))) for s in seeds]
        log_ratio = float(np.mean([math.log(m["perplexity"]) for m in metas])
                          - math.log(ref_meta["perplexity"]))
        agg["log_ppl_ratio"] = log_ratio
        agg["identity_abserr"] = abs(agg["dL"] - log_ratio)
        agg["sum_terms_abserr"] = abs(agg["D"] + agg["Q"] + agg["V"] + agg["R"]
                                      - agg["dL"])
        rows.append(agg)
        if args.per_token_dir:
            os.makedirs(args.per_token_dir, exist_ok=True)
            np.savez(os.path.join(args.per_token_dir,
                                  f"{site}_t{t:02d}_{mode}.npz"), **per_token)

    rows.sort(key=lambda r: (r["site"], r["mode"], r["t"]))
    hdr = (f"{'site':<12}{'mode':>4}{'t':>3}{'S':>4}"
           f"{'D':>12}{'Q':>12}{'V':>12}{'R':>12}"
           f"{'E[dL]':>12}{'Q_naive':>12}{'V_naive':>12}{'|R|bound':>12}{'idErr':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['site']:<12}{r['mode']:>4}{r['t']:>3}{r['seeds']:>4}"
              f"{r['D']:>12.4e}{r['Q']:>12.4e}{r['V']:>12.4e}{r['R']:>12.4e}"
              f"{r['dL']:>12.4e}{r['Q_naive']:>12.4e}{r['V_naive']:>12.4e}"
              f"{r['R_bound']:>12.2e}{r['identity_abserr']:>10.1e}")

    se_hdr = (f"{'site':<12}{'mode':>4}{'t':>3}{'S':>4}"
              f"{'se(D)':>12}{'se(Q)':>12}{'se(V)':>12}{'se(R)':>12}{'se(E[dL])':>12}")
    if any("se_D" in r for r in rows):
        print("\njackknife standard errors over seeds "
              "(blank: one deterministic realization, no sampling error to report)")
        print(se_hdr)
        print("-" * len(se_hdr))
        for r in rows:
            if "se_D" not in r:
                continue
            print(f"{r['site']:<12}{r['mode']:>4}{r['t']:>3}{r['seeds']:>4}"
                  f"{r['se_D']:>12.2e}{r['se_Q']:>12.2e}{r['se_V']:>12.2e}"
                  f"{r['se_R']:>12.2e}{r['se_dL']:>12.2e}")

    if args.csv:
        import csv
        keys = ["site", "mode", "t", "seeds", "D", "Q", "V", "R", "dL",
                "se_D", "se_Q", "se_V", "se_R", "se_dL",
                "Q_naive", "V_naive", "A", "B", "R_bound", "log_ppl_ratio",
                "identity_abserr", "sum_terms_abserr"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")

    if args.seed_budget:
        seed_budget_report(args, ref_dir)


def seed_budget_report(args, ref_dir):
    """How D, Q, V stabilize as the seed count grows.

    Every subset is evaluated from the same [N, S, S] Gram, so this is exact
    subsampling of the realized seed pool rather than a bootstrap approximation
    to it, and it costs no re-reading of the Delta vectors.

    Read the ``sd`` columns, not the means. Averaged over all size-k subsets of
    one pool the corrected Q-hat collapses to the mean off-diagonal Gram entry
    for *every* k, and V-hat to the mean diagonal minus that, so their subset
    means do not move with k by construction -- which is the bias correction
    working. What the seed budget buys is the spread, and that is what has to
    be small against the separation between Q, V and R before a cell can be
    called measured. The naive Q column is shown alongside precisely because it
    does move with k: it is chasing V/k, not converging.
    """
    rng = np.random.default_rng(0)
    print("\nseed-budget subsampling (aggregate terms; spread is over subsets)")
    hdr = (f"{'site':<12}{'t':>3}{'k':>4}{'sets':>6}"
           f"{'D':>11}{'sd':>10}{'Q':>11}{'sd':>10}{'V':>11}{'sd':>10}"
           f"{'Qnaive':>11}")
    print(hdr)
    print("-" * len(hdr))
    for site, t, mode, cdir in discover(args.root):
        if mode == "rn" or (args.sites and site not in args.sites):
            continue
        seeds, gram = get_gram(cdir, ref_dir, rebuild=False)
        scal = load_scalars(cdir, seeds)
        S = len(seeds)
        for k in range(2, S + 1):
            # math.comb first: C(32,16) is 6.0e8, and materializing that list
            # to discover it is too long costs gigabytes and minutes. Only
            # enumerate when the enumeration is small enough to want.
            if math.comb(S, k) > args.max_subsets:
                combos = [tuple(rng.choice(S, size=k, replace=False))
                          for _ in range(args.max_subsets)]
            else:
                combos = list(itertools.combinations(range(S), k))
            vals = [aggregate(terms(gram, scal, list(c), False)) for c in combos]
            def ms(key):
                a = np.array([v[key] for v in vals])
                return a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0
            dm, ds = ms("D"); qm, qs = ms("Q"); vm, vs = ms("V")
            qn, _ = ms("Q_naive")
            print(f"{site:<12}{t:>3}{k:>4}{len(combos):>6}"
                  f"{dm:>11.3e}{ds:>10.2e}{qm:>11.3e}{qs:>10.2e}"
                  f"{vm:>11.3e}{vs:>10.2e}{qn:>11.3e}")


def cmd_refcheck(args):
    """Is the reference a fixed origin, and how far is it from IEEE?

    Two different questions, and only the first is a gate.

    ``--dirs`` must agree **bitwise**. These are repeats of the reference
    itself: PRISM at t=24 under RN, under different seeds. RN takes no
    randomness, so any difference means the path is not reproducible and there
    is no fixed origin to measure Delta from. Bitwise is the right bar here, not
    "agrees to the reported precision": at t=13-14 the perturbation itself is
    only a few ulp.

    ``--advisory`` is reported and does not gate. The IEEE backend belongs here.
    It is *not* the reference and must not be: PRISM's rounding path is not
    IEEE's -- measured at CONTEXT=16, PRISM RN at t=24 moves 96% of the logits
    against IEEE, by up to 5.0e-4 -- and the perturbed cells run their
    non-target arithmetic through PRISM, not through IEEE. Referencing IEEE
    would put that whole difference into every Delta as a drift belonging to no
    site. The IEEE run is only a sanity check: the two agree to the reported
    perplexity.

    Neither of these is the check for the SR background. PRISM's SR at t=24 is
    not the identity either -- it stochastically rounds the exact result to
    binary32 -- so an SR cell's Delta always carries the whole network's t=24
    rounding alongside its site's. That is real and cannot be checked away; the
    `null` stage measures it as D, Q, V, R.
    """
    def compare(base, other_dir):
        other = np.load(os.path.join(other_dir, "logits.npy"), mmap_mode="r")
        if other.shape != base.shape:
            print(f"{other_dir}: shape {other.shape} != {base.shape}  MISMATCH")
            return False
        a, b = np.asarray(base), np.asarray(other)
        identical = bool(np.array_equal(a.view(np.int32), b.view(np.int32)))
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        print(f"{other_dir}: bitwise={'yes' if identical else 'NO'}  "
              f"differing entries={int((diff > 0).sum())}/{a.size}  "
              f"max|diff|={diff.max():.3e}")
        return identical

    ref = np.load(os.path.join(args.dirs[0], "logits.npy"), mmap_mode="r")
    ok = True
    print("required bitwise (reference repeats):")
    for d in args.dirs[1:]:
        ok &= compare(ref, d)
    if args.advisory:
        print("advisory (reported, not a gate):")
        for d in args.advisory:
            compare(ref, d)
    print("refcheck", "PASS" if ok else "FAIL")
    if not ok:
        print("Do not run the graded stages: the reference is not reproducible, "
              "so Delta has no fixed origin. Check that every --dirs entry is an "
              "RN capture -- PRISM's SR at t=24 is not the identity and will "
              "never pass this.")
    return 0 if ok else 1


def cmd_selftest(args):
    """Synthetic check of the estimator algebra, no cluster data needed.

    Draws independent seed pools of Delta^(s) = b + xi^(s) with known b and
    known xi covariance, then checks three things against truth:
    the corrected pair recovers Q and V, the naive Q overshoots by exactly V/k,
    and D + Q + V + R reproduces the imposed excess loss. Unbiasedness is a
    statement about repeated pools, so this replicates the pool rather than
    subsetting one -- subsets of a single pool share their fluctuation and
    would agree with a biased estimator just as happily.
    """
    rng = np.random.default_rng(7)
    V_vocab, N, reps = 300, 24, 400
    logits = rng.normal(size=(N, V_vocab)) * 3.0
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)

    b = rng.normal(size=(N, V_vocab)) * 0.02
    sigma = 0.05

    def varp(x, n):
        return float((p[n] * x * x).sum() - (p[n] * x).sum() ** 2)

    Q_true = float(np.mean([0.5 * varp(b[n], n) for n in range(N)]))
    # eq:fluctuation-closed-form, exact for Sigma = sigma^2 I
    V_true = float(np.mean([0.5 * sigma ** 2 * (1.0 - (p[n] ** 2).sum())
                            for n in range(N)]))

    ok = True
    for k in (2, 5, 16):
        Qs, Vs, Qn, res = [], [], [], []
        for _ in range(reps):
            deltas = b[None] + rng.normal(size=(k, N, V_vocab)) * sigma
            gram = np.empty((N, k, k))
            for n in range(N):
                m = deltas[:, n, :]
                mp = m * p[n]
                mean = mp.sum(1)
                gram[n] = 0.5 * (mp @ m.T - np.outer(mean, mean))
            scal = dict(
                d_y=deltas[:, :, 0],
                Ep_d=np.einsum("snv,nv->sn", deltas, p),
                # An excess loss that is exactly second order, so R must vanish.
                dL=np.array([[ (p[n] * deltas[s, n]).sum() - deltas[s, n, 0]
                               + 0.5 * varp(deltas[s, n], n)
                               for n in range(N)] for s in range(k)]),
                l2=np.linalg.norm(deltas, axis=2),
                W=np.zeros((k, N)))
            a = aggregate(terms(gram, scal, list(range(k)), False))
            Qs.append(a["Q"]); Vs.append(a["V"]); Qn.append(a["Q_naive"])
            res.append(a["D"] + a["Q"] + a["V"] + a["R"] - a["dL"])
        Qs, Vs, Qn, res = map(np.array, (Qs, Vs, Qn, res))

        def check(name, sample, truth, tol_sigmas=4.0):
            se = sample.std(ddof=1) / math.sqrt(len(sample))
            dev = abs(sample.mean() - truth)
            good = dev <= tol_sigmas * se + 1e-12
            print(f"  {name:<9} {sample.mean():.6e}  true {truth:.6e}"
                  f"  dev {dev:.2e}  se {se:.2e}  {'ok' if good else 'FAIL'}")
            return good

        print(f"k={k}  (V/k = {V_true / k:.3e}, i.e. {V_true / k / Q_true:.1f}x Q)")
        ok &= check("Q", Qs, Q_true)
        ok &= check("V", Vs, V_true)
        ok &= check("Q_naive", Qn, Q_true + V_true / k)
        max_res = float(np.abs(res).max())
        print(f"  identity  max |D+Q+V+R-E[dL]| = {max_res:.2e}"
              f"  {'ok' if max_res < 1e-12 else 'FAIL'}")
        ok &= max_res < 1e-12
    print("selftest", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_e2e(args):
    """End-to-end check on a synthetic capture tree written to disk.

    Exercises the code paths the cluster runs -- ``reduce_reference``,
    ``reduce_perturbation``, the on-disk layout, ``build_gram``, ``terms`` and
    the report table -- against logits whose perturbation is known exactly.
    It is the substitute for a cluster smoke run on a machine without PRISM,
    and it is what catches a layout or an alignment mistake before 200
    node-hours go into producing arrays that cannot be combined.
    """
    import tempfile
    rng = np.random.default_rng(11)
    N, V, S, sigma = 128, 500, 8, 0.02
    root = args.dir or tempfile.mkdtemp(prefix="decomp-e2e-")
    ref_dir = os.path.join(root, "reference")
    cell = os.path.join(root, "lm_head", "t07", "sr")
    os.makedirs(ref_dir, exist_ok=True)

    ref_logits = (rng.normal(size=(N, V)) * 1.5).astype(np.float32)
    targets = rng.integers(0, V, size=N).astype(np.int32)
    nll, collision = reduce_reference(ref_logits, targets)
    np.save(os.path.join(ref_dir, "logits.npy"), ref_logits)
    np.savez(os.path.join(ref_dir, "scalars.npz"),
             targets=targets, nll=nll, collision=collision)
    ref_ppl = float(np.exp(nll.mean()))
    json.dump(dict(perplexity=ref_ppl, n_scored=N),
              open(os.path.join(ref_dir, "meta.json"), "w"))

    b = rng.normal(size=(N, V)) * 0.004
    for s in range(1, S + 1):
        sd = os.path.join(cell, f"seed{s}")
        os.makedirs(sd, exist_ok=True)
        pert = (ref_logits.astype(np.float64) + b
                + rng.normal(size=(N, V)) * sigma).astype(np.float32)
        out = np.lib.format.open_memmap(os.path.join(sd, "delta.npy"), mode="w+",
                                        dtype=np.float32, shape=(N, V))
        cols, nonfinite = reduce_perturbation(ref_logits, pert, targets, out)
        out.flush(); del out
        np.savez(os.path.join(sd, "scalars.npz"), targets=targets, **cols)
        json.dump(dict(perplexity=float(np.exp(cols["nll_pert"].mean())),
                       nonfinite_delta=nonfinite),
                  open(os.path.join(sd, "meta.json"), "w"))

    seeds, gram = build_gram(cell, ref_dir)
    scal = load_scalars(cell, seeds)
    agg = aggregate(terms(gram, scal, list(range(len(seeds))), False))

    p_ref = np.exp(log_softmax_np(ref_logits.astype(np.float64)))
    Q_true = float(np.mean(0.5 * ((p_ref * b * b).sum(1)
                                  - (p_ref * b).sum(1) ** 2)))
    V_true = float(np.mean(0.5 * sigma ** 2 * (1.0 - (p_ref ** 2).sum(1))))

    ok = True
    resid = abs(agg["D"] + agg["Q"] + agg["V"] + agg["R"] - agg["dL"])
    print(f"identity  |D+Q+V+R-E[dL]| = {resid:.2e}"
          f"  {'ok' if resid < 1e-12 else 'FAIL'}")
    ok &= resid < 1e-12
    # E[dL] must equal the log ratio of the two perplexities, since log PPL is
    # the token-averaged loss (eq:ppl-loss). This is the check that would fail
    # on a window-boundary misalignment.
    metas = [json.load(open(os.path.join(cell, s, "meta.json"))) for s in seeds]
    lr = float(np.mean([math.log(m["perplexity"]) for m in metas]) - math.log(ref_ppl))
    print(f"E[dL]     {agg['dL']:.8e} vs log PPL ratio {lr:.8e}"
          f"  {'ok' if abs(agg['dL'] - lr) < 1e-10 else 'FAIL'}")
    ok &= abs(agg["dL"] - lr) < 1e-10
    print(f"Q         {agg['Q']:.4e}  true {Q_true:.4e}   "
          f"(naive {agg['Q_naive']:.4e}, expected ~{Q_true + V_true / S:.4e})")
    print(f"V         {agg['V']:.4e}  true {V_true:.4e}")
    ok &= abs(agg["V"] - V_true) < 0.15 * V_true
    ok &= abs(agg["Q_naive"] - (Q_true + V_true / S)) < 0.25 * (Q_true + V_true / S)

    ns = argparse.Namespace(root=root, sites=None, csv=None, per_token_dir=None,
                            seed_budget=True, max_subsets=40, rebuild_gram=False)
    cmd_report(ns)

    print("e2e", "PASS" if ok else "FAIL")
    if args.dir is None:
        import shutil
        shutil.rmtree(root)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gram", help="build one cell's [N, S, S] Gram")
    g.add_argument("--cell-dir", required=True)
    g.add_argument("--ref", required=True)
    g.set_defaults(func=cmd_gram)

    r = sub.add_parser("report", help="tabulate D, Q, V, R over a capture root")
    r.add_argument("--root", required=True)
    r.add_argument("--sites", nargs="*", default=None)
    r.add_argument("--csv", default=None)
    r.add_argument("--per-token-dir", default=None)
    r.add_argument("--seed-budget", action="store_true")
    r.add_argument("--max-subsets", type=int, default=120)
    r.add_argument("--rebuild-gram", action="store_true")
    r.set_defaults(func=cmd_report)

    c = sub.add_parser("refcheck", help="bitwise comparison of reference captures")
    c.add_argument("--dirs", nargs="+", required=True,
                   help="reference repeats; must agree bitwise")
    c.add_argument("--advisory", nargs="*", default=None,
                   help="compared and reported, but not a gate (e.g. the IEEE run)")
    c.set_defaults(func=cmd_refcheck)

    e = sub.add_parser("e2e", help="synthetic end-to-end check of the on-disk pipeline")
    e.add_argument("--dir", default=None, help="keep the tree here instead of a temp dir")
    e.set_defaults(func=cmd_e2e)

    s = sub.add_parser("selftest", help="synthetic check of the estimator algebra")
    s.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
