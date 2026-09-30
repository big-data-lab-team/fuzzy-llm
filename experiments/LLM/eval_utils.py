"""Utility functions for LLM evaluations (dataset loading, perplexity computation, and precision hooks)."""

import math
import os
import urllib.request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WIKITEXT_URL = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt"
LOCAL_CACHE_FILE = "wikitext-2-test.txt"

# Precision and rounding-mode control comes from fuzzy_torch, which drives
# PRISM's C API directly. It replaces the omp_ext extension this harness used
# to carry: PRISM now reconciles a process-wide change with every running
# thread itself, so there is nothing left to broadcast from here.
#
# The import is fatal rather than a warning. Without it the per-module hooks
# below are no-ops, the run silently degrades to full precision, and the
# resulting perplexity is indistinguishable from a real measurement in the logs.
from fuzzy_torch import RN, SR, set_precision, set_rounding_mode  # noqa: F401
from fuzzy_torch import assert_effective, get_precision, get_rounding_mode  # noqa: F401


def make_pre_hook(target_precision, target_mode=None):
    """Creates a pre-hook to drop precision and optionally change rounding mode (0:SR, 1:RN) before the module runs."""
    def hook(module, input):
        set_precision(target_precision)
        if target_mode is not None:
            set_rounding_mode(target_mode)
    return hook

def make_post_hook(default_precision, default_mode=None):
    """Creates a post-hook to restore precision and optionally rounding mode (0:SR, 1:RN) after the module runs."""
    def hook(module, input, output):
        set_precision(default_precision)
        if default_mode is not None:
            set_rounding_mode(default_mode)
    return hook

def load_wikitext_slice(tokenizer, fraction=100, num_windows=1, window_idx=0,
                        min_tokens=None):
    """Loads a slice of the WikiText-2 dataset from a local cache or downloads it.

    Two slicing modes, and they are not interchangeable:

    ``num_windows=1`` (the default) keeps the historical behaviour exactly --
    the leading ``len(lines) // fraction`` lines, i.e. the first 1% of the file.
    Every sweep in the paper was scored this way.

    ``num_windows=k`` with ``window_idx=i`` takes the i-th of k disjoint,
    evenly spaced line blocks, ``lines[i*step:(i+1)*step]`` for
    ``step = len(lines) // k``. This is what the multi-window robustness study
    varies. The blocks are far larger than the token budget -- at k=8 the
    smallest yields about 29k tokens against a 1024-token budget -- so the
    caller's ``max_tokens`` truncation is what actually fixes the sample, and
    window 0 reproduces the ``fraction=100`` sample token for token: 1% of the
    file is ~1900 tokens, more than the budget, and BPE is prefix-consistent.

    ``min_tokens`` asserts the slice can fill the caller's budget. Without it a
    short window is scored at fewer tokens and reports a perplexity that is not
    comparable to the others, with nothing in the log to say so.
    """
    # Check local cache first
    if os.path.exists(LOCAL_CACHE_FILE):
        print(f"Loading dataset slice from local cache '{LOCAL_CACHE_FILE}'...")
        with open(LOCAL_CACHE_FILE, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print("Downloading dataset slice from remote URL...")
        try:
            text = urllib.request.urlopen(WIKITEXT_URL).read().decode("utf-8")
            # Write to local cache for future runs
            with open(LOCAL_CACHE_FILE, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise e

    lines = text.split("\n")
    if num_windows > 1:
        if not 0 <= window_idx < num_windows:
            raise ValueError(
                f"window_idx={window_idx} out of range for num_windows={num_windows}")
        step = len(lines) // num_windows
        if step < 1:
            raise ValueError(f"num_windows={num_windows} exceeds {len(lines)} lines")
        slice_lines = lines[window_idx * step:(window_idx + 1) * step]
    else:
        # Take the specified fraction of the lines (default 1%)
        slice_lines = lines[:max(1, len(lines) // fraction)]

    encodings = tokenizer("\n\n".join(slice_lines), return_tensors="pt")

    available = encodings.input_ids.size(1)
    if min_tokens is not None and available < min_tokens:
        raise ValueError(
            f"slice yields {available} tokens, short of the {min_tokens} requested "
            f"(num_windows={num_windows}, window_idx={window_idx}); scoring it would "
            f"report a perplexity over fewer tokens than the rest of the sweep")
    return encodings

def evaluate_perplexity(model, encodings, seq_len, stride, max_length, verbose=False):
    """Evaluates perplexity of the model over the given encodings in chunks."""
    ppl, _, _ = _evaluate(model, encodings, seq_len, stride, max_length,
                          verbose=verbose, collect_logits=False)
    return ppl


def evaluate_perplexity_with_logits(model, encodings, seq_len, stride, max_length,
                                    verbose=False):
    """Same scoring as :func:`evaluate_perplexity`, also returning the logits.

    Returns ``(ppl, logits, targets)``. ``logits`` is ``[N, V]`` float32 over
    exactly the ``N = seq_len - n_windows`` positions the perplexity averages,
    and ``targets[i]`` is the token id that position predicts. The first
    position of every window is not scored -- it has no preceding context
    inside the window -- which is where the ``- n_windows`` comes from, and why
    this returns the scored slice rather than the raw ``[1, seq_len, V]``
    tensor: an off-by-one at a window boundary would silently pair a logit
    vector with the wrong token and there is nothing downstream that could
    catch it.

    The logits are the *perturbed* ones when the caller has precision hooks
    installed, which is the whole point: the per-token decomposition of
    eq:head-full-comparison needs the logit perturbation itself, not the
    aggregate it averages to.
    """
    return _evaluate(model, encodings, seq_len, stride, max_length,
                     verbose=verbose, collect_logits=True)


def _evaluate(model, encodings, seq_len, stride, max_length, verbose=False,
              collect_logits=False):
    """Shared body. Perplexity is computed identically either way, so a run
    with logit capture reports the same number as the sweep it is paired
    with."""
    nlls = []

    n_windows = max(1, -(-seq_len // stride))
    logits_out = None
    targets_out = None
    if collect_logits:
        n_scored = seq_len - n_windows
        vocab = model.config.vocab_size
        logits_out = torch.empty((n_scored, vocab), dtype=torch.float32)
        targets_out = torch.empty((n_scored,), dtype=torch.long)
    filled = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - begin_loc

        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()

        if verbose:
            print(f"Evaluating chunk from {begin_loc} to {end_loc}...", flush=True)

        with torch.no_grad():
            if verbose:
                print("Running forward pass...", flush=True)
            outputs = model(input_ids, labels=target_ids)
            if verbose:
                print("Forward pass complete.", flush=True)

            log_likelihood = outputs.loss * (trg_len - 1)
            nlls.append(log_likelihood)

            if collect_logits:
                take = trg_len - 1
                logits_out[filled:filled + take] = outputs.logits[0, :take, :].float()
                targets_out[filled:filled + take] = input_ids[0, 1:trg_len]
                filled += take

    total_predicted_tokens = seq_len - len(nlls)
    ppl = torch.exp(torch.stack(nlls).sum() / total_predicted_tokens)

    if collect_logits:
        assert filled == total_predicted_tokens, (
            f"captured {filled} scored positions but the perplexity averages "
            f"{total_predicted_tokens}")
    return ppl.item(), logits_out, targets_out

def load_model_and_dataset(model_id="distilgpt2", fraction=100, num_threads=None,
                           num_windows=1, window_idx=0, min_tokens=None):
    """Loads the tokenizer, causal LM model, and WikiText-2 dataset slice.

    ``num_windows``/``window_idx``/``min_tokens`` are passed straight through to
    :func:`load_wikitext_slice`; see there for what they select.
    """
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    encodings = load_wikitext_slice(tokenizer, fraction=fraction,
                                    num_windows=num_windows, window_idx=window_idx,
                                    min_tokens=min_tokens)
    return model, tokenizer, encodings
