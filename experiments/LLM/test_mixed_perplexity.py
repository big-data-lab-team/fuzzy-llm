"""Perplexity of DistilGPT-2 on WikiText-2 under a per-site precision and rounding assignment.

Evaluates one mixed configuration of tab:accuracy-matched-precision, e.g.

    python3 test_mixed_perplexity.py \
        --sites attn_c_attn=8:rn,attn_c_proj=6:sr,mlp_c_fc=8:rn,mlp_c_proj=6:sr,lm_head=11:rn

Each of the four block projections is reduced in all six blocks, and the
language-model head separately; every other operation runs at t=24 under the
--default_mode rounding rule. The SR seed comes from VFC_BACKENDS.
"""

import argparse
import time

import eval_utils

BLOCK_SITES = ("attn_c_attn", "attn_c_proj", "mlp_c_fc", "mlp_c_proj")
SITES = BLOCK_SITES + ("lm_head",)
MODES = {"sr": eval_utils.SR, "rn": eval_utils.RN}


def parse_sites(text):
    """``"attn_c_attn=8:rn,mlp_c_proj=6:sr,..."`` -> ``{site: (t, mode)}``, all five sites."""
    config = {}
    for item in filter(None, (s.strip() for s in text.split(","))):
        name, spec = item.split("=", 1)
        precision, mode = spec.split(":", 1)
        if name not in SITES or mode not in MODES:
            raise argparse.ArgumentTypeError(f"bad site assignment {item!r}")
        config[name] = (int(precision), mode)
    missing = [s for s in SITES if s not in config]
    if missing:
        raise argparse.ArgumentTypeError(f"no assignment for {', '.join(missing)}")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sites", type=parse_sites, required=True,
                        help="SITE=T:MODE for each of " + ", ".join(SITES) + ".")
    parser.add_argument("--default_mode", choices=["sr", "rn"], default="rn",
                        help="Rounding mode of the operations outside the five sites.")
    parser.add_argument("--context_length", type=int, default=256,
                        help="Sliding window length (and stride) for evaluation.")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Number of tokens of the test slice to score.")
    args = parser.parse_args()

    model, tokenizer, encodings = eval_utils.load_model_and_dataset("distilgpt2", fraction=100)
    seq_len = min(encodings.input_ids.size(1), args.max_tokens)
    default_precision, default_mode = 24, MODES[args.default_mode]
    config = args.sites
    label = ",".join(f"{s}={config[s][0]}:{config[s][1]}" for s in SITES)
    print(f"\nEvaluating sites: {label}")
    print(f"  All other operations: {default_precision}-bit {args.default_mode.upper()}")

    modules = [(name, getattr(block.attn if name.startswith("attn") else block.mlp, name.split("_", 1)[1]))
               for block in model.transformer.h for name in BLOCK_SITES]
    modules.append(("lm_head", model.lm_head))
    handles = []
    for name, module in modules:
        precision, mode = config[name]
        handles.append(module.register_forward_pre_hook(eval_utils.make_pre_hook(precision, MODES[mode])))
        handles.append(module.register_forward_hook(eval_utils.make_post_hook(default_precision, default_mode)))
    print(f"\nInstrumented {len(handles) // 2} layers.")

    start = time.time()
    ppl = eval_utils.evaluate_perplexity(model, encodings, seq_len, stride=args.context_length,
                                         max_length=args.context_length, verbose=False)
    elapsed = time.time() - start
    for h in handles:
        h.remove()

    print(f"\nResult -> Sites: {label} | Perplexity: {ppl:.4f} | Time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
