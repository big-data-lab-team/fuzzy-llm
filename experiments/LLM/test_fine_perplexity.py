"""Test LLM perplexity with DistilGPT-2 on WikiText-2, applying fine-grained per-component lower precision via Verificarlo."""

import time
import torch
import argparse
import eval_utils


def parse_mode(value):
    if value is None:
        return None
    return {"sr": eval_utils.SR, "rn": eval_utils.RN}[value]


def main():
    parser = argparse.ArgumentParser(description="Fine-grained per-component perplexity evaluation.")
    parser.add_argument(
        "--layer", 
        type=str, 
        required=True, 
        choices=["attn_c_attn", "attn_c_proj", "mlp_c_fc", "mlp_c_proj", "attention", "mlp", "lm_head"],
        help="Target layer/module type to reduce precision."
    )
    parser.add_argument("--precision", type=int, required=True, help="Simulated significand bitwidth.")
    parser.add_argument(
        "--target_mode",
        choices=["sr", "rn"],
        default=None,
        help="Optional rounding mode inside the target module.",
    )
    parser.add_argument(
        "--default_mode",
        choices=["sr", "rn"],
        default=None,
        help="Optional rounding mode restored outside the target module.",
    )
    parser.add_argument("--context_length", type=int, default=256, help="Sliding window length (and stride) for evaluation.")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Total tokens to score. Defaults to context_length, which scores a "
             "single window and gives a high-variance perplexity comparable only "
             "within a sweep level. Set larger (e.g. 1024) to average several windows.",
    )
    parser.add_argument(
        "--block_idx", 
        type=str, 
        default="all", 
        help="Apply to specific block index (e.g. 0-5) or 'all'."
    )
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

    # Determine which modules to instrument
    modules = []
    blocks = model.transformer.h
    
    if args.block_idx == "all":
        block_indices = list(range(len(blocks)))
    elif "-" in str(args.block_idx):
        parts = str(args.block_idx).split("-")
        block_indices = list(range(int(parts[0]), int(parts[1]) + 1))
    elif "," in str(args.block_idx):
        block_indices = [int(x.strip()) for x in str(args.block_idx).split(",")]
    else:
        try:
            block_indices = [int(args.block_idx)]
        except ValueError:
            raise ValueError(f"Invalid block_idx: {args.block_idx}. Must be 'all', range '0-2', comma-separated '0,1', or an integer.")

    for idx in block_indices:
        if idx < 0 or idx >= len(blocks):
            raise IndexError(f"Block index {idx} out of range (0-{len(blocks)-1}).")
        block = blocks[idx]
        
        if args.layer == "attn_c_attn":
            modules.append(block.attn.c_attn)
        elif args.layer == "attn_c_proj":
            modules.append(block.attn.c_proj)
        elif args.layer == "mlp_c_fc":
            modules.append(block.mlp.c_fc)
        elif args.layer == "mlp_c_proj":
            modules.append(block.mlp.c_proj)
        elif args.layer == "attention":
            modules.append(block.attn)
        elif args.layer == "mlp":
            modules.append(block.mlp)

    if args.layer == "lm_head":
        modules.append(model.lm_head)

    print(f"\n--- Evaluating Layer: {args.layer} | Block: {args.block_idx} | Precision: {args.precision} ---")
    print(f"Instrumenting {len(modules)} submodule(s)")

    default_precision = 24
    target_mode = parse_mode(args.target_mode)
    default_mode = parse_mode(args.default_mode)
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
    
    # Clean up hooks
    for h in handles:
        h.remove()

    print(f"Result -> Layer: {args.layer} | Block: {args.block_idx} | Window: {args.window_idx:d}/{args.num_windows:d} | Precision: {args.precision:2d} | Perplexity: {ppl:8.2f} | Time: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
