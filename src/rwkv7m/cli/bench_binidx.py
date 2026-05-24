import argparse
import time

import jax

from ..api import create_train_runtime, train_batch
from ..data import create_binidx_dataset
from ..model import ModelConfig, ScreeningConfig


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
            use_write_screening=variant == "read_write",
        )
    return ModelConfig(
        d_model=args.d_model,
        d_ffn=args.d_ffn,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        head_size=args.head_size,
        vocab_size=args.vocab_size,
        max_seq_len=args.ctx_len,
        dtype=args.dtype,
        use_screening=use_screening,
        screening=screening,
    )


def run_variant(args, variant):
    phase = "read_write" if variant == "read_write" else "read_screening_only"
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
    parser.add_argument("--ctx-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--magic-prime", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", nargs="+", default=["baseline", "screening", "read_write"])
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results = []
    for variant in args.variants:
        if variant not in {"baseline", "screening", "read_write"}:
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
