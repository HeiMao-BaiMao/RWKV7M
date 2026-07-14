import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from ..distributed import (
    initialize_jax_distributed,
    initialize_nnx_probe,
    make_mesh,
    nnx_state_sharding_summary,
    restore_nnx_probe_checkpoint,
    runtime_version_manifest,
    save_nnx_probe_checkpoint,
    train_nnx_probe_step,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate NNX sharded init/update and Orbax save/restore."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--mode", choices=("create", "restore"), required=True)
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Probe width; defaults to max(8, global device count)",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def run_probe(args):
    initialize_jax_distributed()
    model_axis_size = jax.device_count()
    width = args.width if args.width is not None else max(8, model_axis_size)
    if width <= 0 or width % model_axis_size != 0:
        raise ValueError("width must be positive and divisible by the model axis size")
    mesh = make_mesh(
        ("data", "model"),
        axis_sizes=(1, model_axis_size),
        axis_types=(jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
    )
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if args.mode == "create":
        model, optimizer = initialize_nnx_probe(
            mesh,
            width=width,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    else:
        model, optimizer = restore_nnx_probe_checkpoint(
            checkpoint_dir,
            mesh,
            width=width,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )

    before = nnx_state_sharding_summary(model, optimizer)
    inputs = jnp.arange(2 * width, dtype=jnp.float32).reshape(2, width)
    targets = jnp.zeros_like(inputs)
    with jax.set_mesh(mesh):
        loss = train_nnx_probe_step(model, optimizer, inputs, targets, mesh)
        jax.block_until_ready(nnx.state((model, optimizer)))
    after = nnx_state_sharding_summary(model, optimizer)

    if args.mode == "create":
        save_nnx_probe_checkpoint(checkpoint_dir, model, optimizer)

    return {
        "mode": args.mode,
        "loss": float(loss),
        "optimizer_step": int(optimizer.step.get_value()),
        "mesh": {
            "axis_names": list(mesh.axis_names),
            "shape": dict(mesh.shape),
        },
        "sharding_before": before,
        "sharding_after": after,
        "environment": runtime_version_manifest(),
        "checkpoint_dir": str(checkpoint_dir),
    }


def main(argv=None):
    print(json.dumps(run_probe(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
