"""Capture per-token logits and the scalars used by the paper's loss diagnostics.

Section 4.2 uses the exact two-term identity D + K = dL, with
D = Ep_d - d_y and K = dL - D = KL(p || q). The existing captures already
contain everything needed by scripts/exact_loss.py, without covariance
estimation or another inference run. The quadratic decomposition below is
retained in Appendix B.

Why this exists
---------------
The sweep records hold aggregate perplexities. Every term of

    E_m[dL] = D_m + Q_m + V_m + R_m                         (eq:head-full-comparison)

is instead a p-weighted linear or quadratic form of the *logit perturbation*
Delta_m = lambda_perturbed - lambda_reference at a single token, so none of them
needs the V x V covariance the closed forms are written with: for a single
realization

    E_p[Delta] = p . Delta,     Var_p(Delta) = p . Delta^2 - (p . Delta)^2

are O(V), and the whole decomposition follows from those two scalars plus
Delta_y. This script produces, per scored token and per realization, exactly

    d_y  = Delta_y                 the drift at the observed token
    Ep_d = E_p[Delta]              the p-weighted mean perturbation
    W    = Var_p(Delta)            the p-weighted dispersion
    dL   = L(lambda+Delta) - L(lambda)   the *exact* excess loss, not its
                                          second-order approximation
    l2   = ||Delta||_2             for the third-moment bound on R_m

and persists Delta itself, because one term -- Q_m = 1/2 Var_p(b_m) with
b_m = E_m[Delta_m] -- is a quadratic form of a cross-seed *mean vector* and does
not collapse into per-seed scalars. ``decompose.py`` consumes those vectors.

Run with ``--precision 24`` and no ``--ref`` to produce the reference; run with
``--ref`` pointing at that directory for every perturbed cell.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import eval_utils
from decompose import reduce_reference, reduce_perturbation
from test_percomponent_perplexity import make_qkav_attn, _ATTN_SOURCE_VERSION

SITES = ("attn_c_attn", "attn_c_proj", "mlp_c_fc", "mlp_c_proj",
         "attn_qkav", "lm_head", "attention", "mlp", "none")


def install_site_hooks(model, site, precision, default_precision):
    """Bracket ``site`` at ``precision``. Returns a callable that undoes it.

    The four sub-layer sites and the three group sites are the ones
    test_fine_perplexity.py and test_percomponent_perplexity.py instrument, at
    the same module boundaries and with the same hooks, so a cell captured here
    is the same arithmetic as the corresponding sweep cell.
    """
    if site == "none":
        return lambda: None

    if site == "attn_qkav":
        if transformers_version() != _ATTN_SOURCE_VERSION:
            raise RuntimeError(
                f"attn_qkav copies GPT2Attention._attn from transformers=="
                f"{_ATTN_SOURCE_VERSION}; installed is {transformers_version()}. "
                "Refusing to run against a possibly-changed implementation.")
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
        original = GPT2Attention._attn
        GPT2Attention._attn = make_qkav_attn(precision, None, default_precision, None)

        def undo():
            GPT2Attention._attn = original
        return undo

    blocks = model.transformer.h
    picker = {
        "attn_c_attn": lambda b: b.attn.c_attn,
        "attn_c_proj": lambda b: b.attn.c_proj,
        "mlp_c_fc": lambda b: b.mlp.c_fc,
        "mlp_c_proj": lambda b: b.mlp.c_proj,
        "attention": lambda b: b.attn,
        "mlp": lambda b: b.mlp,
    }
    if site == "lm_head":
        modules = [model.lm_head]
    else:
        modules = [picker[site](b) for b in blocks]
    print(f"Instrumenting {len(modules)} submodule(s)")

    handles = []
    for mod in modules:
        handles.append(mod.register_forward_pre_hook(
            eval_utils.make_pre_hook(precision, None)))
        handles.append(mod.register_forward_hook(
            eval_utils.make_post_hook(default_precision, None)))

    def undo():
        for h in handles:
            h.remove()
    return undo


def transformers_version():
    import transformers
    return transformers.__version__


def backend_seed():
    """The PRISM seed, which arrives through VFC_BACKENDS rather than argv."""
    for tok in os.environ.get("VFC_BACKENDS", "").split():
        if tok.startswith("--seed="):
            return tok.split("=", 1)[1]
    return None


def backend_mode():
    """SR, RN, or IEEE, read off the backend string.

    The reference check runs one capture under ``libinterflop_ieee.so``, which
    carries neither a seed nor a mode; calling that "sr" would put a run that
    does no rounding at all under the stochastic label.
    """
    backends = os.environ.get("VFC_BACKENDS", "")
    if "prism" not in backends:
        return "ieee" if "ieee" in backends else "unknown"
    return "rn" if "--mode=rn" in backends else "sr"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True, choices=SITES,
                    help="Reduction to lower. 'none' captures the reference.")
    ap.add_argument("--precision", type=int, required=True)
    ap.add_argument("--context_length", type=int, default=256)
    ap.add_argument("--max_tokens", type=int, default=None)
    ap.add_argument("--num_windows", type=int, default=1)
    ap.add_argument("--window_idx", type=int, default=0)
    ap.add_argument("--out-dir", required=True,
                    help="Directory for this cell's arrays; created if absent.")
    ap.add_argument("--ref", default=None,
                    help="Reference cell directory. Omit to write a reference.")
    ap.add_argument("--delta-dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--keep-logits", action="store_true",
                    help="Also persist the perturbed logits. Off by default: "
                         "Delta plus the reference determines them exactly.")
    args = ap.parse_args()

    if args.ref is None and args.site != "none":
        raise SystemExit("a reference capture must use --site none")
    # `--site none --ref ...` is the *null cell*: nothing is instrumented, so
    # Delta is whatever this run's own arithmetic differs from the reference by.
    # Under RN that is identically zero; under SR at t=24 it is the background
    # stochastic rounding of the whole network, which every SR cell carries in
    # addition to its site's and which therefore sets the floor below which a
    # term is not measured.

    os.makedirs(args.out_dir, exist_ok=True)

    model, tokenizer, encodings = eval_utils.load_model_and_dataset(
        "distilgpt2", fraction=100,
        num_windows=args.num_windows, window_idx=args.window_idx,
        min_tokens=args.max_tokens)

    max_length = args.context_length
    stride = args.context_length
    max_tokens = args.max_tokens if args.max_tokens is not None else args.context_length
    seq_len = min(encodings.input_ids.size(1), max_tokens)
    n_windows = max(1, -(-seq_len // stride))
    n_scored = seq_len - n_windows
    print(f"\nTotal tokens to evaluate: {seq_len} in {n_windows} window(s) of "
          f"{max_length}; {n_scored} scored positions")

    default_precision = 24
    undo = install_site_hooks(model, args.site, args.precision, default_precision)

    print(f"\n--- Capturing: site={args.site} | precision={args.precision} | "
          f"mode={backend_mode()} | seed={backend_seed()} ---")
    start = time.time()
    ppl, logits, targets = eval_utils.evaluate_perplexity_with_logits(
        model, encodings, seq_len, stride, max_length, verbose=False)
    elapsed = time.time() - start
    undo()

    logits = logits.numpy()
    targets = targets.numpy().astype(np.int32)

    meta = dict(site=args.site, precision=args.precision, mode=backend_mode(),
                seed=backend_seed(), context_length=args.context_length,
                max_tokens=max_tokens, num_windows=args.num_windows,
                window_idx=args.window_idx, seq_len=seq_len,
                n_scored=int(n_scored), vocab=int(logits.shape[1]),
                perplexity=ppl, elapsed_s=elapsed,
                vfc_backends=os.environ.get("VFC_BACKENDS", ""))

    if args.ref is None:
        nll, collision = reduce_reference(logits, targets)
        np.save(os.path.join(args.out_dir, "logits.npy"), logits)
        np.savez(os.path.join(args.out_dir, "scalars.npz"),
                 targets=targets, nll=nll, collision=collision)
        # Validation 1: the captured slice must reproduce the perplexity the
        # sweep reports. This is the check that catches a window-boundary
        # off-by-one, which no later stage could detect; misalignment moves the
        # perplexity by far more than the 1e-5 relative that float32 loss
        # accumulation in the harness does, so the bound leaves that room.
        recomputed = float(np.exp(nll.mean()))
        meta["ppl_from_logits"] = recomputed
        meta["ppl_from_logits_relerr"] = abs(recomputed - ppl) / ppl
        print(f"check: perplexity from captured logits {recomputed:.6f} vs "
              f"harness {ppl:.6f} (rel {meta['ppl_from_logits_relerr']:.2e})")
        assert meta["ppl_from_logits_relerr"] < 1e-3, "logit capture is misaligned"
    else:
        ref_logits = np.load(os.path.join(args.ref, "logits.npy"), mmap_mode="r")
        ref_scalars = np.load(os.path.join(args.ref, "scalars.npz"))
        ref_targets = ref_scalars["targets"]
        if ref_logits.shape != logits.shape:
            raise SystemExit(f"reference is {ref_logits.shape}, this cell is "
                             f"{logits.shape}: different token sets")
        if not np.array_equal(ref_targets, targets):
            raise SystemExit("reference and perturbed runs scored different tokens")

        delta_path = os.path.join(args.out_dir, "delta.npy")
        out_delta = np.lib.format.open_memmap(
            delta_path, mode="w+", dtype=np.dtype(args.delta_dtype),
            shape=logits.shape)
        cols, nonfinite = reduce_perturbation(ref_logits, logits, targets, out_delta)
        out_delta.flush()
        del out_delta

        np.savez(os.path.join(args.out_dir, "scalars.npz"),
                 targets=targets, **cols)
        if args.keep_logits:
            np.save(os.path.join(args.out_dir, "logits.npy"), logits)

        meta["nonfinite_delta"] = nonfinite
        # Validation 2: the token-averaged exact excess loss must equal the log
        # ratio of the two measured perplexities, since log PPL *is* the
        # token-averaged loss (eq:ppl-loss). This ties every downstream term to
        # the number tab:percomponent reports.
        ref_ppl = float(np.exp(ref_scalars["nll"].mean()))
        lhs = float(cols["dL"].mean())
        rhs = float(np.log(ppl) - np.log(ref_ppl))
        meta.update(ref_perplexity=ref_ppl, mean_dL=lhs, log_ppl_ratio=rhs,
                    dL_check_abserr=abs(lhs - rhs))
        print(f"check: mean per-token dL {lhs:.8f} vs log(PPL/PPL_ref) "
              f"{rhs:.8f} (abs {abs(lhs - rhs):.2e})")
        assert abs(lhs - rhs) < 1e-4 * max(1.0, abs(rhs)), "excess loss is inconsistent"
        if nonfinite:
            print(f"WARNING: {nonfinite} non-finite Delta entries")

    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    # `Bits:` rather than `Precision:`, as test_mixed_perplexity.py writes
    # `Sites:`: run_config.sh needs a `Result ->`
    # line to tell a finished task from a dead one, but these captures are not
    # sweep records and must never become rows in sweep_records.csv. The capture
    # root is outside every tree parse_sweep_logs.py ingests; this is the second
    # line of defence, and it fails loudly rather than producing a wrong record.
    print(f"Result -> Site: {args.site} | Bits: {args.precision:2d} | "
          f"Mode: {backend_mode()} | Seed: {backend_seed()} | "
          f"Perplexity: {ppl:8.2f} | Tokens: {n_scored} | Time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
