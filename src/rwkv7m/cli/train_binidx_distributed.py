import argparse
from pathlib import Path

import jax

from ..api import create_train_runtime
from ..distributed import (
    create_host_binidx_dataset,
    initialize_jax_distributed,
    make_1d_mesh,
    replicate_train_objects,
    train_batch_data_parallel,
)
from ..io import load_train_checkpoint, load_train_checkpoint_metadata, save_train_checkpoint
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
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume", default=None)
    return parse_args_with_config(parser, argv)


def _checkpoint_path(output_dir, step):
    return Path(output_dir) / f"ckpt-{int(step):08d}"


def _save_checkpoint(args, dist, config, step, info):
    if args.output_dir is None or info["process_index"] != 0:
        return None
    checkpoint_dir = _checkpoint_path(args.output_dir, step)
    save_train_checkpoint(
        checkpoint_dir,
        jax.device_get(dist.train_state),
        config,
        rng_key=jax.random.PRNGKey(args.seed),
        dataset_position={"step": int(step)},
        metadata={
            "data_file": args.data_file,
            "ctx_len": args.ctx_len,
            "global_batch_size": args.global_batch_size,
            "process_count": info["process_count"],
            "phase": args.phase,
            "distributed": True,
        },
    )
    print(f"saved checkpoint {checkpoint_dir}")
    return checkpoint_dir


def run_distributed_training(args):
    info = initialize_jax_distributed()
    mesh = make_1d_mesh("data")
    if args.resume:
        config, payload = load_train_checkpoint_metadata(args.resume)
        start_step = int(payload.get("step", 0))
    else:
        config = build_config(args)
        start_step = 0

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
    runtime, train_state = create_train_runtime(
        jax.random.PRNGKey(args.seed),
        config,
        batch_size=args.global_batch_size,
        total_steps=max(start_step + args.steps, 1),
    )
    if args.resume:
        train_state, config, _ = load_train_checkpoint(args.resume, train_state)
        runtime.variables = {"params": train_state.params}
    dist = replicate_train_objects(runtime, train_state, mesh=mesh)
    last_checkpoint = None
    try:
        print(
            f"process={info['process_index']}/{info['process_count']} "
            f"devices={info['local_device_count']} global_batch={args.global_batch_size} "
            f"process_batch={dataset.layout.process_batch_size} start_step={start_step}"
        )
        for local_step in range(args.steps):
            global_step = start_step + local_step
            dist, metrics = train_batch_data_parallel(
                dist,
                dataset.get_batch(global_step),
                dataset.layout,
                phase=args.phase,
            )
            completed_step = int(dist.train_state.step)
            if args.print_every and (
                local_step % args.print_every == 0 or local_step == args.steps - 1
            ):
                print(f"step={completed_step} loss={float(metrics['loss']):.6f}")
            if (
                args.output_dir is not None
                and args.save_every > 0
                and completed_step % args.save_every == 0
            ):
                last_checkpoint = _save_checkpoint(args, dist, config, completed_step, info)

        if args.output_dir is not None:
            final_step = int(dist.train_state.step)
            final_path = _checkpoint_path(args.output_dir, final_step)
            if last_checkpoint != final_path:
                last_checkpoint = _save_checkpoint(args, dist, config, final_step, info)
    finally:
        dataset.close()
    return dist, last_checkpoint


def main(argv=None):
    args = parse_args(argv)
    run_distributed_training(args)


if __name__ == "__main__":
    main()
