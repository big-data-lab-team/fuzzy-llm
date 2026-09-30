"""Test LLM perplexity with DistilGPT-2 on WikiText-2, applying uniform lower
precision across the whole network via the same forward-hook scoping the
site-scoped harness uses.

Setting the precision once through the backend options for the whole process
would also drop the loss and the perplexity aggregation into the instrumented
arithmetic. This script instead hooks ``model.transformer`` and ``model.lm_head`` -- the two submodules that
together are "the network" -- exactly as test_percomponent_perplexity.py hooks
a component group, so the loss and the exp() in eval_utils.evaluate_perplexity
run at t=24 like every other level.
"""

import time
import argparse
import eval_utils


def parse_mode(value):
    if value is None:
        return None
    return {"sr": eval_utils.SR, "rn": eval_utils.RN}[value]


def main():
    parser = argparse.ArgumentParser(description="Global (whole-network) perplexity evaluation.")
    parser.add_argument("--precision", type=int, required=True, help="Simulated significand bitwidth.")
    parser.add_argument("--target_mode", choices=["sr", "rn"], default=None)
    parser.add_argument("--default_mode", choices=["sr", "rn"], default=None)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument(
        "--num_windows",
        type=int,
        default=1,
        help="Split the test file into this many disjoint, evenly spaced line "
             "blocks and score one of them. 1 (the default) keeps the historical "
             "leading-1%% slice.",
    )
    parser.add_argument(
        "--window_idx",
        type=int,
        default=0,
        help="Which window to score, 0-based. Ignored when --num_windows is 1.",
    )
    args = parser.parse_args()

    model, tokenizer, encodings = eval_utils.load_model_and_dataset(
        "distilgpt2", fraction=100,
        num_windows=args.num_windows, window_idx=args.window_idx,
        min_tokens=args.max_tokens)

    max_length = args.context_length
    stride = args.context_length
    max_tokens = args.max_tokens if args.max_tokens is not None else args.context_length
    seq_len = min(encodings.input_ids.size(1), max_tokens)

    n_windows = max(1, -(-seq_len // stride))
    print(f"\nTotal tokens to evaluate: {seq_len} in {n_windows} window(s) of {max_length}")

    default_precision = 24
    target_mode = parse_mode(args.target_mode)
    default_mode = parse_mode(args.default_mode)

    # The network is transformer body + lm_head. Hooking these two submodules
    # (rather than the top-level model) keeps the loss and exp() outside the
    # instrumented scope: both run only after lm_head's post-hook has already
    # restored t=24, the same way a component sweep leaves the loss untouched.
    modules = [model.transformer, model.lm_head]

    print(f"\n--- Evaluating Global | Precision: {args.precision} ---")
    print(f"Instrumenting {len(modules)} submodule(s): transformer, lm_head")

    handles = []
    for mod in modules:
        pre_handle = mod.register_forward_pre_hook(
            eval_utils.make_pre_hook(args.precision, target_mode)
        )
        post_handle = mod.register_forward_hook(
            eval_utils.make_post_hook(default_precision, default_mode)
        )
        handles.extend([pre_handle, post_handle])

    start_time = time.time()
    ppl = eval_utils.evaluate_perplexity(model, encodings, seq_len, stride, max_length, verbose=False)
    elapsed = time.time() - start_time

    for h in handles:
        h.remove()

    print(f"Result -> Global | Window: {args.window_idx:d}/{args.num_windows:d} | Precision: {args.precision:2d} | Perplexity: {ppl:8.2f} | Time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
