import argparse
import copy
from dataclasses import replace
import time

import jax

from ..api import create_train_runtime, train_batch
from ..data import create_binidx_dataset
from ..io import load_model_config
from ..model import (
    MODEL_PRESET_NAMES,
    ModelConfig,
    ScreeningConfig,
    model_preset,
)
from .config import (
    add_screening_v2_args,
    add_training_vocab_tiling_args,
    apply_execution_overrides,
    parse_args_with_config,
    screening_v2_kwargs,
)


def default_bank_ids(n_slots):
    if n_slots == 1:
        return (0,)
    if n_slots == 2:
        return (0, 2)
    short = max(1, n_slots // 2)
    mid = max(1, (n_slots - short) // 2)
    return tuple([0] * short + [1] * mid + [2] * (n_slots - short - mid))


def build_config(args, variant):
    use_screening = variant != "baseline"
    if args.model_config is not None or args.model_preset is not None:
        config = copy.deepcopy(
            load_model_config(args.model_config)
            if args.model_config is not None
            else model_preset(args.model_preset)
        )
        if args.ctx_len > config.max_seq_len:
            raise ValueError(
                f"ctx_len={args.ctx_len} exceeds model max_seq_len={config.max_seq_len}"
            )
        config.use_screening = use_screening
        if use_screening:
            config.screening.use_write_screening = variant in (
                "read_write",
                "screening_v2",
            )
            if variant == "screening_v2":
                v2_overrides = screening_v2_kwargs(args)
                v2_overrides["write_mode"] = "competitive_novel"
                config.screening = replace(
                    config.screening,
                    **v2_overrides,
                )
        return apply_execution_overrides(config, args)
    screening = ScreeningConfig()
    if use_screening:
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
            use_write_screening=variant in ("read_write", "screening_v2"),
            **(
                {
                    **screening_v2_kwargs(args),
                    "write_mode": "competitive_novel",
                }
                if variant == "screening_v2"
                else {}
            ),
        )
    config = ModelConfig(
        d_model=args.d_model,
        d_ffn=args.d_ffn,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        head_size=args.head_size,
        vocab_size=args.vocab_size,
        max_seq_len=args.ctx_len,
        dtype=args.dtype,
        remat_blocks=bool(args.remat_blocks),
        sequence_chunk_size=args.sequence_chunk_size,
        head_chunk_size=args.head_chunk_size,
        use_screening=use_screening,
        screening=screening,
    )
    return apply_execution_overrides(config, args)


def run_variant(args, variant):
    phase = (
        "read_write"
        if variant in ("read_write", "screening_v2")
        else "read_screening_only"
    )
    cfg = build_config(args, variant)
    dataset = create_binidx_dataset(
        args.data_file,
        ctx_len=args.ctx_len,
        batch_size=args.batch_size,
        magic_prime=args.magic_prime,
        epoch_steps=args.steps,
    )
    runtime, state = create_train_runtime(
        jax.random.PRNGKey(args.seed),
        cfg,
        batch_size=args.batch_size,
        total_steps=args.steps,
    )
    losses = []
    start = time.perf_counter()
    try:
        for step in range(args.steps):
            batch = dataset.get_batch(step)
            state, metrics = train_batch(state, batch, runtime, phase=phase)
            loss = float(metrics["loss"])
            losses.append(loss)
            if args.print_every and (step % args.print_every == 0 or step == args.steps - 1):
                print(f"{variant} step={step} loss={loss:.6f}")
        # Waiting on the final loss is not sufficient to prove that every
        # optimizer output leaf has completed. Synchronize the complete final
        # state once at the end of this measurement window.
        jax.block_until_ready(state)
    finally:
        dataset.close()
    elapsed = time.perf_counter() - start
    tokens = args.steps * args.batch_size * args.ctx_len
    return {
        "variant": variant,
        "steps": args.steps,
        "tokens": tokens,
        "seconds": elapsed,
        "tokens_per_sec": tokens / elapsed if elapsed > 0 else 0.0,
        "first_loss": losses[0],
        "last_loss": losses[-1],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train small rwkv7m variants on a binidx file.")
    parser.add_argument("--data-file", required=True)
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument("--model-config", default=None)
    model_source.add_argument(
        "--model-preset", choices=MODEL_PRESET_NAMES, default=None
    )
    parser.add_argument("--ctx-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--magic-prime", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "screening", "read_write"],
    )
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    remat_group = parser.add_mutually_exclusive_group()
    remat_group.add_argument(
        "--remat-blocks",
        dest="remat_blocks",
        action="store_true",
    )
    remat_group.add_argument(
        "--no-remat-blocks",
        dest="remat_blocks",
        action="store_false",
    )
    parser.set_defaults(remat_blocks=None)
    chunk_group = parser.add_mutually_exclusive_group()
    chunk_group.add_argument("--sequence-chunk-size", type=int, default=None)
    chunk_group.add_argument("--no-sequence-chunking", action="store_true")
    head_chunk_group = parser.add_mutually_exclusive_group()
    head_chunk_group.add_argument("--head-chunk-size", type=int, default=None)
    head_chunk_group.add_argument("--no-head-chunking", action="store_true")
    add_training_vocab_tiling_args(parser)
    add_screening_v2_args(parser)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ffn", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--head-size", type=int, default=16)
    parser.add_argument("--d-slot", type=int, default=32)
    parser.add_argument("--d-k", type=int, default=16)
    parser.add_argument("--d-v", type=int, default=16)
    parser.add_argument("--n-slots", type=int, default=4)
    parser.add_argument("--screened-layers", type=int, nargs="*", default=[])
    parser.add_argument("--print-every", type=int, default=1)
    return parse_args_with_config(parser, argv)


def main(argv=None):
    args = parse_args(argv)
    results = []
    for variant in args.variants:
        if variant not in {
            "baseline",
            "screening",
            "read_write",
            "screening_v2",
        }:
            raise ValueError(f"unknown variant: {variant}")
        results.append(run_variant(args, variant))

    print("variant,steps,tokens,seconds,tokens_per_sec,first_loss,last_loss")
    for item in results:
        print(
            f"{item['variant']},{item['steps']},{item['tokens']},"
            f"{item['seconds']:.6f},{item['tokens_per_sec']:.2f},"
            f"{item['first_loss']:.6f},{item['last_loss']:.6f}"
        )


if __name__ == "__main__":
    main()
