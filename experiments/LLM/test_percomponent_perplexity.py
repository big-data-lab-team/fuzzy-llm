"""Test LLM perplexity with DistilGPT-2 on WikiText-2, applying per-component lower precision via Verificarlo."""

import time
import types
import torch
from torch import nn
import argparse
import transformers
import eval_utils

# GPT2Attention._attn is copied below (unmodified control flow, precision
# bracketing added only around the two matmuls) to isolate QK^T and AV from
# c_attn/c_proj, which the existing "attention" group already sweeps together
# with these ops and cannot tell apart. There is no submodule boundary here to
# hook -- softmax(QK^T)V happens inside one method -- so this pins the
# transformers version it was copied from and refuses to run against another.
_ATTN_SOURCE_VERSION = "4.40.0"


def parse_mode(value):
    if value is None:
        return None
    return {"sr": eval_utils.SR, "rn": eval_utils.RN}[value]


def make_qkav_attn(precision, target_mode, default_precision, default_mode):
    """Copy of GPT2Attention._attn with precision bracketing around QK^T and AV only.

    Softmax, masking, and dropout run at ``default_precision`` (24), exactly as
    every other site's non-target arithmetic does; only the two matmuls run at
    the swept precision.
    """
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        eval_utils.set_precision(precision)
        if target_mode is not None:
            eval_utils.set_rounding_mode(target_mode)
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        eval_utils.set_precision(default_precision)
        if default_mode is not None:
            eval_utils.set_rounding_mode(default_mode)

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        eval_utils.set_precision(precision)
        if target_mode is not None:
            eval_utils.set_rounding_mode(target_mode)
        attn_output = torch.matmul(attn_weights, value)
        eval_utils.set_precision(default_precision)
        if default_mode is not None:
            eval_utils.set_rounding_mode(default_mode)

        return attn_output, attn_weights

    return _attn


def main():
    parser = argparse.ArgumentParser(description="Per-component perplexity evaluation.")
    parser.add_argument("--group", type=str, required=True,
                         choices=["attention", "mlp", "lm_head", "attn_qkav"])
    parser.add_argument("--precision", type=int, required=True)
    parser.add_argument("--target_mode", choices=["sr", "rn"], default=None)
    parser.add_argument("--default_mode", choices=["sr", "rn"], default=None)
    parser.add_argument("--context_length", type=int, default=256, help="Sliding window length (and stride) for evaluation")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Total tokens to score. Defaults to context_length, which scores a "
             "single window and gives a high-variance perplexity comparable only "
             "within a sweep level. Set larger (e.g. 1024) to average several windows.",
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

    group_name = args.group
    precision = args.precision
    default_precision = 24
    target_mode = parse_mode(args.target_mode)
    default_mode = parse_mode(args.default_mode)

    print(f"\n--- Evaluating Group: {group_name} | Precision: {precision} ---")

    if group_name == "attn_qkav":
        if transformers.__version__ != _ATTN_SOURCE_VERSION:
            raise RuntimeError(
                f"attn_qkav copies GPT2Attention._attn from transformers=="
                f"{_ATTN_SOURCE_VERSION}; installed is {transformers.__version__}. "
                "Refusing to run against a possibly-changed implementation."
            )
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
        original_attn = GPT2Attention._attn
        GPT2Attention._attn = make_qkav_attn(precision, target_mode, default_precision, default_mode)
        print(f"Monkeypatched GPT2Attention._attn on all {len(model.transformer.h)} block(s)")
        handles = []
    else:
        groups = {
            "attention": [block.attn for block in model.transformer.h],
            "mlp": [block.mlp for block in model.transformer.h],
            "lm_head": [model.lm_head],
        }
        modules = groups[group_name]
        print(f"Instrumenting {len(modules)} submodule(s)")

        handles = []
        for mod in modules:
            pre_handle = mod.register_forward_pre_hook(
                eval_utils.make_pre_hook(precision, target_mode)
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
    if group_name == "attn_qkav":
        GPT2Attention._attn = original_attn

    print(f"Result -> Group: {group_name} | Window: {args.window_idx:d}/{args.num_windows:d} | Precision: {precision:2d} | Perplexity: {ppl:8.2f} | Time: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
