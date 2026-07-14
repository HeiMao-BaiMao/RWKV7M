"""Audit the Phase 3 NNX model-parallel execution contract."""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
from flax import nnx

from ..api import create_train_runtime, tiny_config
from ..io import load_model_config
from ..distributed import (
    audit_lowered_collectives,
    initialize_jax_distributed,
    make_mesh,
    place_train_objects,
    runtime_version_manifest,
    train_global_batch_data_parallel,
)
from ..model.nnx_model import NNXShardingConfig
from ..model import MODEL_PRESET_NAMES, model_preset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compile and execute a small NNX RWKV7M model on an Explicit "
            "data x model mesh, then report parameter/state shardings and HLO collectives."
        )
    )
    parser.add_argument("--model-axis-size", type=int, default=None)
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument("--model-config", default=None)
    model_source.add_argument(
        "--model-preset", choices=MODEL_PRESET_NAMES, default=None
    )
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--head-size", type=int, default=8)
    parser.add_argument("--ctx-len", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--screening", action="store_true")
    parser.add_argument("--write-screening", action="store_true")
    parser.add_argument("--vocab-parallel", action="store_true")
    parser.add_argument("--remat-blocks", action="store_true")
    parser.add_argument("--sequence-chunk-size", type=int, default=None)
    parser.add_argument("--head-chunk-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _array_summary(value):
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sharding": str(value.sharding.spec),
        "addressable_shards": len(value.addressable_shards),
    }


def _parameter_summary(model):
    summary = {}
    for path, variable in nnx.to_flat_state(nnx.state(model, nnx.Param)):
        value = variable.get_value()
        summary["/".join(str(part) for part in path)] = _array_summary(value)
    return summary


def run_audit(args):
    initialize_jax_distributed()
    device_count = jax.device_count()
    model_axis_size = args.model_axis_size or device_count
    if model_axis_size <= 0 or device_count % model_axis_size != 0:
        raise ValueError(
            "model-axis-size must be positive and divide the global device count"
        )
    if (
        args.model_config is None
        and args.model_preset is None
        and args.d_model != args.n_heads * args.head_size
    ):
        raise ValueError("d-model must equal n-heads * head-size")
    if args.model_config is not None:
        config = load_model_config(args.model_config)
    elif args.model_preset is not None:
        config = model_preset(args.model_preset)
    else:
        config = tiny_config(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=2 if args.screening else 1,
            n_heads=args.n_heads,
            head_size=args.head_size,
            use_screening=args.screening,
        )
    for name, value in (
        ("d-model", config.d_model),
        ("n-heads", config.n_heads),
    ):
        if value % model_axis_size != 0:
            raise ValueError(f"{name} must be divisible by model-axis-size")

    data_axis_size = device_count // model_axis_size
    mesh = make_mesh(
        ("data", "model"),
        axis_sizes=(data_axis_size, model_axis_size),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
        ),
    )
    if args.model_config is None and args.model_preset is None:
        config.lm_head_init = "variance_scaled"
        config.screening.use_write_screening = args.write_screening
        config.vocab_parallel = args.vocab_parallel
        config.remat_blocks = args.remat_blocks
        config.sequence_chunk_size = args.sequence_chunk_size
        config.head_chunk_size = args.head_chunk_size
    if args.ctx_len > config.max_seq_len:
        raise ValueError("ctx-len exceeds model max_seq_len")
    if (
        config.use_screening
        and config.screening.screened_layers
        and config.screening.d_slot % model_axis_size != 0
    ):
        raise ValueError("screening d_slot must be divisible by model-axis-size")
    if config.vocab_parallel and config.vocab_size % model_axis_size != 0:
        raise ValueError("vocab_size must be divisible by model-axis-size")

    sharding = NNXShardingConfig(mesh)
    runtime, train_state = create_train_runtime(
        jax.random.key(args.seed),
        config,
        batch_size=data_axis_size,
        total_steps=2,
        sharding=sharding,
    )
    dist = place_train_objects(
        runtime,
        train_state,
        mesh=mesh,
        axis_name="data",
        param_axis_name="model",
    )
    ids = jnp.arange(
        data_axis_size * args.ctx_len,
        dtype=jnp.int32,
    ).reshape(data_axis_size, args.ctx_len) % args.vocab_size
    batch_sharding = sharding.named("data", None)
    input_ids = jax.device_put(ids, batch_sharding)
    batch = {
        "input_ids": input_ids,
        "target_ids": jax.device_put((ids + 1) % args.vocab_size, batch_sharding),
        "mask": jax.device_put(jnp.ones_like(ids, dtype=jnp.float32), batch_sharding),
    }

    graphdef, model_state = nnx.split(dist.train_state.model)
    phase = "read_write" if args.write_screening else "read_screening_only"

    @jax.jit
    def forward(state, token_ids, rwkv_state, screen_state):
        model = nnx.merge(graphdef, state)
        logits, new_rwkv, new_screen, _ = model(
            token_ids,
            rwkv_state,
            screen_state,
            phase=phase,
            deterministic=True,
        )
        return logits, new_rwkv, new_screen

    with jax.set_mesh(mesh):
        lowered = forward.lower(
            model_state,
            input_ids,
            dist.initial_rwkv_state,
            dist.initial_screen_state,
        )
        compiled = lowered.compile()
        # Compiled.as_text() exposes the post-SPMD executable HLO. The
        # pre-partition lowering can legitimately contain no collective ops.
        collective_audit = audit_lowered_collectives(compiled)
        logits, rwkv_state, screen_state = compiled(
            model_state,
            input_ids,
            dist.initial_rwkv_state,
            dist.initial_screen_state,
        )
        jax.block_until_ready((logits, rwkv_state, screen_state))
        dist, metrics = train_global_batch_data_parallel(
            dist,
            batch,
            phase=phase,
        )
        jax.block_until_ready(dist.train_state.ready_state())

    parameter_summary = _parameter_summary(dist.train_state.model)
    state_summary = {
        "logits": _array_summary(logits),
        "wkv": _array_summary(rwkv_state[0].wkv),
        "time_mix_x": _array_summary(rwkv_state[0].time_mix_x),
    }
    if screen_state.layers:
        state_summary["screening_slots"] = _array_summary(
            screen_state.layers[0].slots
        )
    has_row_kernel = any(
        item["sharding"] == "P('model', None)"
        for item in parameter_summary.values()
    )
    has_column_kernel = any(
        item["sharding"] == "P(None, 'model')"
        for item in parameter_summary.values()
    )
    return {
        "mesh": {
            "axis_names": list(mesh.axis_names),
            "shape": dict(mesh.shape),
            "axis_types": [str(axis_type) for axis_type in mesh.axis_types],
        },
        "forward": {
            "finite": bool(jnp.all(jnp.isfinite(logits))),
            "collectives": collective_audit.to_dict(),
        },
        "train_step": {
            "loss": float(metrics["loss"]),
            "finite": bool(jnp.isfinite(metrics["loss"])),
            "optimizer_step": int(dist.train_state.step),
        },
        "contract": {
            "has_row_parallel_kernel": has_row_kernel,
            "has_column_parallel_kernel": has_column_kernel,
            "model_parallel_exercised": model_axis_size > 1,
        },
        "states": state_summary,
        "parameters": parameter_summary,
        "environment": runtime_version_manifest(),
    }


def main(argv=None):
    print(json.dumps(run_audit(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
