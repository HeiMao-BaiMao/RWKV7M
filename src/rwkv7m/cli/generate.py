import argparse

import jax

from ..api import create_runtime, generate_text
from ..io import load_model_safetensors
from ..tokenizer import RWKVTokenizer
from .config import parse_args_with_config
from .eval_binidx import resolve_checkpoint_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate text from an rwkv7m safetensors checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir or model.safetensors file")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", choices=["read_screening_only", "read_write"], default="read_screening_only")
    parser.add_argument("--continuation-only", action="store_true")
    parser.add_argument("--vocab-file", default=None)
    return parse_args_with_config(parser, argv)


def main(argv=None):
    args = parse_args(argv)
    params, config, _ = load_model_safetensors(resolve_checkpoint_file(args.checkpoint))
    if config is None:
        raise ValueError("checkpoint safetensors is missing rwkv7m config metadata")
    runtime = create_runtime(jax.random.PRNGKey(args.seed), config, batch_size=1)
    runtime.variables = {"params": params}
    tokenizer = RWKVTokenizer(args.vocab_file)
    text = generate_text(
        runtime,
        args.prompt,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        rng_key=jax.random.PRNGKey(args.seed),
        phase=args.phase,
        include_prompt=not args.continuation_only,
    )
    print(text)


if __name__ == "__main__":
    main()
