"""Locate non-finite parameter gradients in one fixed local NNX train step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from rwkv7m.api import create_train_runtime
from rwkv7m.distributed.checkpoint import (
    load_distributed_checkpoint_metadata,
    restore_distributed_train_state,
)
from rwkv7m.io import load_model_config
from rwkv7m.io.config import model_config_to_dict
from rwkv7m.train.nnx_train import nnx_model_loss


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-batch", type=Path, required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional distributed checkpoint whose parameters should be "
            "diagnosed."
        ),
    )
    parser.add_argument(
        "--variant",
        choices=("baseline", "screening", "read_write"),
        default="read_write",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sequence-chunk-size",
        type=int,
        default=None,
        help=(
            "Override recurrent chunking for diagnosis; use 0 to disable "
            "chunking without changing checkpoint tensor shapes."
        ),
    )
    parser.add_argument(
        "--float32-model",
        action="store_true",
        help=(
            "Run the restored weights and activations in float32 to separate "
            "mixed-precision overflow from the recurrence equations."
        ),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-finite",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.sequence_chunk_size is not None and args.sequence_chunk_size < 0:
        parser.error("--sequence-chunk-size must be non-negative")
    return args


def _path_string(path):
    return "/".join(str(part) for part in path)


def summarize_gradient_state(gradients, *, top_k):
    flat = nnx.to_flat_state(gradients)
    device_statistics = []
    for _, value in flat:
        value_f32 = value.astype(jnp.float32)
        finite = jnp.isfinite(value_f32)
        finite_value = jnp.where(finite, value_f32, 0.0)
        device_statistics.append(
            (
                jnp.sum(~finite),
                jnp.max(jnp.abs(finite_value), initial=0.0),
                jnp.sqrt(jnp.sum(finite_value * finite_value)),
            )
        )
    host_statistics = jax.device_get(device_statistics)
    records = []
    for (path, value), (nonfinite, max_abs, l2_norm) in zip(
        flat,
        host_statistics,
        strict=True,
    ):
        records.append(
            {
                "path": _path_string(path),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "nonfinite_count": int(nonfinite),
                "max_abs_finite": float(max_abs),
                "l2_norm_finite": float(l2_norm),
            }
        )
    nonfinite_records = [
        record for record in records if record["nonfinite_count"] > 0
    ]
    largest = sorted(
        records,
        key=lambda record: record["max_abs_finite"],
        reverse=True,
    )[:top_k]
    return {
        "leaf_count": len(records),
        "nonfinite_leaf_count": len(nonfinite_records),
        "nonfinite_value_count": sum(
            record["nonfinite_count"] for record in nonfinite_records
        ),
        "nonfinite_gradients": nonfinite_records,
        "largest_finite_gradients": largest,
    }


def main(argv=None):
    args = parse_args(argv)
    config = load_model_config(args.model_config)
    config.use_screening = args.variant != "baseline"
    if config.use_screening:
        config.screening.use_write_screening = args.variant == "read_write"
    phase = (
        "read_write"
        if args.variant == "read_write"
        else "read_screening_only"
    )
    with np.load(args.fixed_batch, allow_pickle=False) as archive:
        batch = {
            "input_ids": archive["input_ids"],
            "target_ids": archive["target_ids"],
        }
        if "mask" in archive:
            batch["mask"] = archive["mask"]
    batch_size, token_count = batch["input_ids"].shape
    if token_count > config.max_seq_len:
        raise SystemExit("fixed batch exceeds the model maximum sequence length")
    batch = jax.device_put(batch)
    checkpoint_step = 0
    checkpoint_path = None
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint.resolve()
        checkpoint_metadata = load_distributed_checkpoint_metadata(
            checkpoint_path
        )
        stored_config = model_config_to_dict(checkpoint_metadata.config)
        if stored_config != model_config_to_dict(config):
            raise SystemExit(
                "--model-config does not match the configuration stored in "
                "--checkpoint"
            )
        checkpoint_step = checkpoint_metadata.start_step
    if args.sequence_chunk_size is not None:
        config.sequence_chunk_size = (
            None
            if args.sequence_chunk_size == 0
            else args.sequence_chunk_size
        )
    if args.float32_model:
        config.dtype = "float32"
        config.param_dtype = "float32"
    runtime, train_state = create_train_runtime(
        jax.random.key(args.seed),
        config,
        batch_size=batch_size,
        total_steps=max(checkpoint_step + 2, 2),
    )
    if checkpoint_path is not None:
        train_state, restored = restore_distributed_train_state(
            checkpoint_path,
            train_state,
        )
        checkpoint_step = restored.start_step
    if args.float32_model:
        promoted_params = jax.tree.map(
            lambda value: value.astype(jnp.float32)
            if hasattr(value, "dtype")
            and jnp.issubdtype(value.dtype, jnp.inexact)
            else value,
            nnx.state(train_state.model, nnx.Param),
        )
        nnx.update(train_state.model, promoted_params)
    graphdef, params = nnx.split(train_state.model, nnx.Param)

    def loss_function(active_params):
        model = nnx.merge(graphdef, active_params)
        loss, _ = nnx_model_loss(
            model,
            batch,
            runtime.initial_rwkv_state,
            runtime.initial_screen_state,
            phase=phase,
            deterministic=False,
            include_l2wrap=True,
            training_step=jnp.asarray(checkpoint_step, dtype=jnp.uint32),
        )
        return loss

    loss, gradients = jax.jit(jax.value_and_grad(loss_function))(params)
    jax.block_until_ready((loss, gradients))
    summary = summarize_gradient_state(gradients, top_k=args.top_k)
    payload = {
        "loss": float(loss),
        "loss_finite": bool(jnp.isfinite(loss)),
        "batch_size": int(batch_size),
        "tokens": int(token_count),
        "variant": args.variant,
        "model_config": str(Path(args.model_config)),
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "training_step": int(checkpoint_step),
        "sequence_chunk_size": config.sequence_chunk_size,
        "float32_model": bool(args.float32_model),
        "devices": [
            {
                "platform": device.platform,
                "device_kind": device.device_kind,
                "id": device.id,
            }
            for device in jax.devices()
        ],
        "gradients": summary,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.require_finite and (
        not payload["loss_finite"] or summary["nonfinite_leaf_count"] > 0
    ):
        raise SystemExit("non-finite loss or gradients detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
