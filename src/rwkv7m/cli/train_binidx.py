import argparse

import jax

from ..api import create_train_runtime, train_batch
from ..data import create_binidx_dataset
from ..model import ModelConfig, ScreeningConfig


def build_config(args):
    if args.use_screening:
        screened_layers = tuple(args.screened_layers)
        if not screened_layers and args.n_layers > 1:
            screened_layers = (args.n_layers // 2,)
        bank_ids = tuple(args.bank_ids) if args.bank_ids is not None else default_bank_ids(args.n_slots)
        screening = ScreeningConfig(
            d_model=args.d_model,
            d_slot=args.d_slot,
            d_k=args.d_k,
            d_v=args.d_v,
            n_slots=args.n_slots,
            screened_layers=screened_layers,
            bank_ids=bank_ids,
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


def default_bank_ids(n_slots):
    if n_slots <= 0:
        raise ValueError("n_slots must be positive")
    if n_slots == 1:
        return (0,)
    if n_slots == 2:
        return (0, 2)
    short = max(1, n_slots // 2)
    mid = max(1, (n_slots - short) // 2)
    long = n_slots - short - mid
    return tuple([0] * short + [1] * mid + [2] * long)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train rwkv7m on RWKV-LM-V7 .bin/.idx data.")
    parser.add_argument("--data-file", required=True, help="Dataset prefix path without .bin/.idx")
    parser.add_argument("--ctx-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
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
    parser.add_argument("--bank-ids", type=int, nargs="+", default=None)
    parser.add_argument("--screened-layers", type=int, nargs="*", default=[])
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = build_config(args)
    dataset = create_binidx_dataset(
        args.data_file,
        ctx_len=args.ctx_len,
        batch_size=args.batch_size,
        magic_prime=args.magic_prime,
        epoch_steps=args.steps,
    )
    runtime, train_state = create_train_runtime(
        jax.random.PRNGKey(args.seed),
        cfg,
        batch_size=args.batch_size,
        total_steps=args.steps,
    )
    try:
        print(
            f"data_tokens={dataset.data_size} magic_prime={dataset.magic_prime} "
            f"ctx_len={args.ctx_len} batch_size={args.batch_size}"
        )
        for step in range(args.steps):
            batch = dataset.get_batch(step)
            train_state, metrics = train_batch(
                train_state,
                batch,
                runtime,
                phase=args.phase,
                carry_state=False,
            )
            if step % args.print_every == 0 or step == args.steps - 1:
                print(f"step={step} loss={float(metrics['loss']):.6f}")
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
