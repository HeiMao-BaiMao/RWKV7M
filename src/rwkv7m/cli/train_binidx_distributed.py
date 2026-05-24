import argparse

import jax

from ..api import create_train_runtime
from ..distributed import (
    create_host_binidx_dataset,
    initialize_jax_distributed,
    make_1d_mesh,
    replicate_train_objects,
    train_batch_data_parallel,
)
from .config import parse_args_with_config
from .train_binidx import build_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Data-parallel rwkv7m binidx training skeleton for JAX distributed setups."
    )
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--ctx-len", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, default=1)
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
    return parse_args_with_config(parser, argv)


def run_distributed_training(args):
    info = initialize_jax_distributed()
    mesh = make_1d_mesh("data")
    dataset = create_host_binidx_dataset(
        args.data_file,
        ctx_len=args.ctx_len,
        global_batch_size=args.global_batch_size,
        magic_prime=args.magic_prime,
        epoch_steps=max(args.steps, 1),
        process_index=info["process_index"],
        process_count=info["process_count"],
        local_device_count=info["local_device_count"],
    )
    config = build_config(args)
    runtime, train_state = create_train_runtime(
        jax.random.PRNGKey(args.seed),
        config,
        batch_size=args.global_batch_size,
        total_steps=args.steps,
    )
    dist = replicate_train_objects(runtime, train_state, mesh=mesh)
    try:
        print(
            f"process={info['process_index']}/{info['process_count']} "
            f"devices={info['local_device_count']} global_batch={args.global_batch_size} "
            f"process_batch={dataset.layout.process_batch_size}"
        )
        for step in range(args.steps):
            dist, metrics = train_batch_data_parallel(
                dist,
                dataset.get_batch(step),
                dataset.layout,
                phase=args.phase,
            )
            if args.print_every and (step % args.print_every == 0 or step == args.steps - 1):
                print(f"step={int(dist.train_state.step)} loss={float(metrics['loss']):.6f}")
    finally:
        dataset.close()
    return dist


def main(argv=None):
    args = parse_args(argv)
    run_distributed_training(args)


if __name__ == "__main__":
    main()
