import argparse
import math
from pathlib import Path

import jax
import jax.numpy as jnp

from ..api import create_runtime
from ..data import create_binidx_dataset
from ..io import load_model_safetensors
from ..model import ModelConfig, ScreeningConfig
from ..model.screened_rwkv import cross_entropy_loss
from .config import parse_args_with_config


def default_bank_ids(n_slots):
    if n_slots == 1:
        return (0,)
    if n_slots == 2:
        return (0, 2)
    short = max(1, n_slots // 2)
    mid = max(1, (n_slots - short) // 2)
    return tuple([0] * short + [1] * mid + [2] * (n_slots - short - mid))


def build_config(args):
    if args.use_screening:
        screened_layers = tuple(args.screened_layers)
        if not screened_layers and args.n_layers > 1:
            screened_layers = (args.n_layers // 2,)
        screening = ScreeningConfig(
            d_model=args.d_model,
            d_slot=args.d_slot,
            d_k=args.d_k,
            d_v=args.d_v,
            n_slots=args.n_slots,
            screened_layers=screened_layers,
            bank_ids=default_bank_ids(args.n_slots),
            use_write_screening=args.phase == "read_write",
        )
    else:
        screening = ScreeningConfig()
    return ModelConfig(
        d_model=args.d_model,
        d_ffn=args.d_ffn,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        head_size=args.head_size,
        vocab_size=args.vocab_size,
        max_seq_len=args.ctx_len,
        dtype=args.dtype,
        use_screening=args.use_screening,
        screening=screening,
    )


def resolve_checkpoint_file(checkpoint):
    path = Path(checkpoint)
    if path.is_dir():
        return path / "model.safetensors"
    return path


def evaluate_binidx(args):
    if args.checkpoint:
        params, config, _ = load_model_safetensors(resolve_checkpoint_file(args.checkpoint))
        if config is None:
            raise ValueError("checkpoint safetensors is missing rwkv7m config metadata")
    else:
        config = build_config(args)
        params = None

    dataset = create_binidx_dataset(
        args.data_file,
        ctx_len=args.ctx_len,
        batch_size=args.batch_size,
        magic_prime=args.magic_prime,
        epoch_steps=args.steps,
    )
    runtime = create_runtime(
        jax.random.PRNGKey(args.seed),
        config,
        batch_size=args.batch_size,
    )
    if params is not None:
        runtime.variables = {"params": params}

    @jax.jit
    def eval_step(input_ids, target_ids, mask, rwkv_state, screen_state):
        logits, _, _, stats = runtime.model.apply(
            runtime.variables,
            input_ids,
            rwkv_state,
            screen_state,
            phase=args.phase,
            deterministic=True,
        )
        return cross_entropy_loss(logits, target_ids, mask), stats

    losses = []
    try:
        for step in range(args.steps):
            batch = dataset.get_batch(step)
            loss, _ = eval_step(
                batch["input_ids"],
                batch["target_ids"],
                batch["mask"],
                runtime.initial_rwkv_state,
                runtime.initial_screen_state,
            )
            losses.append(float(loss))
            if args.print_every and (step % args.print_every == 0 or step == args.steps - 1):
                print(f"eval step={step} loss={losses[-1]:.6f}")
    finally:
        dataset.close()

    mean_loss = sum(losses) / len(losses)
    return {
        "steps": args.steps,
        "tokens": args.steps * args.batch_size * args.ctx_len,
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate rwkv7m next-token loss on binidx data.")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint dir or model.safetensors file")
    parser.add_argument("--ctx-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--magic-prime", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", choices=["read_screening_only", "read_write"], default="read_screening_only")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ffn", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--head-size", type=int, default=32)
    parser.add_argument("--no-screening", dest="use_screening", action="store_false")
    parser.set_defaults(use_screening=True)
    parser.add_argument("--d-slot", type=int, default=64)
    parser.add_argument("--d-k", type=int, default=32)
    parser.add_argument("--d-v", type=int, default=32)
    parser.add_argument("--n-slots", type=int, default=4)
    parser.add_argument("--screened-layers", type=int, nargs="*", default=[])
    parser.add_argument("--print-every", type=int, default=1)
    return parse_args_with_config(parser, argv)


def main(argv=None):
    args = parse_args(argv)
    metrics = evaluate_binidx(args)
    print(
        "eval "
        f"steps={metrics['steps']} tokens={metrics['tokens']} "
        f"loss={metrics['loss']:.6f} perplexity={metrics['perplexity']:.6f}"
    )


if __name__ == "__main__":
    main()
