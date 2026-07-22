"""Locate non-finite parameter gradients in one fixed local NNX train step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading

from flax import nnx
import jax
import jax.numpy as jnp
from jax.experimental import checkify
import numpy as np

from rwkv7m.api import create_train_runtime
from rwkv7m.distributed.checkpoint import (
    load_distributed_checkpoint_metadata,
    restore_distributed_train_state,
)
from rwkv7m.io import load_model_config
from rwkv7m.io.config import model_config_to_dict
from rwkv7m.model import nnx_model as nnx_model_module
from rwkv7m.model.wkv import wkv7_reference
from rwkv7m.train.nnx_train import nnx_model_loss
from rwkv7m.train.train_state import create_learning_rate_schedule


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
        "--screening-eps",
        type=float,
        default=None,
        help=(
            "Override the Screening numerical epsilon after checkpoint "
            "compatibility validation; intended only for fixed-batch "
            "sensitivity diagnosis."
        ),
    )
    parser.add_argument(
        "--screening-norm-eps",
        type=float,
        default=None,
        help=(
            "Override only the v5 vector-normalization epsilon after "
            "checkpoint compatibility validation."
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
    parser.add_argument(
        "--float32-parameters",
        action="store_true",
        help=(
            "Promote restored parameter storage to float32 while preserving "
            "the configured activation/compute dtype. This separates "
            "parameter quantization from full-model compute precision."
        ),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--reference-wkv-layers",
        default="",
        help=(
            "comma-separated zero-based layer indices whose WKV recurrence "
            "should use the portable reference instead of backend auto"
        ),
    )
    parser.add_argument(
        "--capture-wkv-layer",
        type=int,
        default=None,
        help=(
            "capture one layer's Pallas WKV inputs and output cotangents "
            "while using the portable pullback for the enclosing diagnosis"
        ),
    )
    parser.add_argument(
        "--capture-wkv-dir",
        type=Path,
        default=None,
        help="directory for --capture-wkv-layer NPZ files",
    )
    parser.add_argument(
        "--component-gradients",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "also differentiate CE, admission floor, write budget, and "
            "self-index losses independently"
        ),
    )
    parser.add_argument(
        "--checkify-floats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "instrument the total loss/gradient graph with JAX NaN and "
            "division checks and record the first reported error"
        ),
    )
    parser.add_argument(
        "--include-update",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply one diagnostic optimizer update and summarize its norm",
    )
    parser.add_argument(
        "--optimizer-total-steps",
        type=int,
        default=10000,
        help="optimizer horizon used to report LR and perform the optional update",
    )
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
    if args.screening_eps is not None and args.screening_eps <= 0.0:
        parser.error("--screening-eps must be positive")
    if args.screening_norm_eps is not None and args.screening_norm_eps <= 0.0:
        parser.error("--screening-norm-eps must be positive")
    if args.optimizer_total_steps <= 0:
        parser.error("--optimizer-total-steps must be positive")
    if args.float32_model and args.float32_parameters:
        parser.error(
            "--float32-model and --float32-parameters are mutually exclusive"
        )
    try:
        args.reference_wkv_layers = tuple(
            sorted(
                {
                    int(value.strip())
                    for value in args.reference_wkv_layers.split(",")
                    if value.strip()
                }
            )
        )
    except ValueError:
        parser.error("--reference-wkv-layers must contain integer indices")
    if any(index < 0 for index in args.reference_wkv_layers):
        parser.error("--reference-wkv-layers indices must be non-negative")
    if (args.capture_wkv_layer is None) != (args.capture_wkv_dir is None):
        parser.error(
            "--capture-wkv-layer and --capture-wkv-dir must be used together"
        )
    if args.capture_wkv_layer is not None and args.capture_wkv_layer < 0:
        parser.error("--capture-wkv-layer must be non-negative")
    if (
        args.capture_wkv_layer is not None
        and args.capture_wkv_layer in args.reference_wkv_layers
    ):
        parser.error(
            "a captured WKV layer cannot also use --reference-wkv-layers"
        )
    if args.capture_wkv_layer is not None and args.checkify_floats:
        parser.error("WKV capture cannot be combined with --checkify-floats")
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
    group_records = {}
    for group_name, predicate in (
        ("all", lambda _: True),
        ("screening", lambda path: "screening_" in path),
        ("trunk", lambda path: "screening_" not in path),
    ):
        selected = [record for record in records if predicate(record["path"])]
        group_records[group_name] = {
            "leaf_count": len(selected),
            "nonfinite_value_count": sum(
                record["nonfinite_count"] for record in selected
            ),
            "l2_norm_finite": float(
                sum(record["l2_norm_finite"] ** 2 for record in selected)
                ** 0.5
            ),
            "max_abs_finite": max(
                (record["max_abs_finite"] for record in selected),
                default=0.0,
            ),
        }
    return {
        "leaf_count": len(records),
        "nonfinite_leaf_count": len(nonfinite_records),
        "nonfinite_value_count": sum(
            record["nonfinite_count"] for record in nonfinite_records
        ),
        "nonfinite_gradients": nonfinite_records,
        "largest_finite_gradients": largest,
        "groups": group_records,
    }


def _optimizer_config(config):
    return {
        "lr_init": config.lr_init,
        "lr_final": config.lr_final,
        "warmup_steps": config.warmup_steps,
        "lr_schedule": config.lr_schedule,
    }


def _parameter_delta(before, after):
    return jax.tree.map(
        lambda left, right: (
            right.astype(jnp.float32) - left.astype(jnp.float32)
        ),
        before,
        after,
    )


def _writable_device_copy(value):
    return jax.device_put(np.array(jax.device_get(value), copy=True))


def _make_optimizer_state_writable(optimizer):
    """Detach restored NumPy buffers before an in-place NNX update.

    Flax deserialization may retain read-only host arrays. Forward and
    gradient diagnostics can consume them, but ``Optimizer.update`` mutates
    the step variable and therefore requires independently owned buffers.
    """

    state = nnx.state(optimizer)
    copied = jax.tree.map(
        lambda value: _writable_device_copy(value)
        if hasattr(value, "dtype")
        else value,
        state,
    )
    nnx.update(optimizer, copied)


def _install_wkv_capture(capture_dir):
    """Capture exact Pallas primals/cotangents with a reference pullback."""

    capture_dir.mkdir(parents=True, exist_ok=True)
    original_wkv7 = nnx_model_module.wkv7
    counter = iter(range(1_000_000))
    lock = threading.Lock()

    def write_capture(*values):
        names = (
            "r",
            "w",
            "k",
            "v",
            "neg_kk",
            "kka",
            "initial_state",
            "y_cotangent",
            "final_state_cotangent",
        )
        with lock:
            index = next(counter)
        payload = {
            name: np.asarray(value, dtype=np.float32)
            for name, value in zip(names, values, strict=True)
        }
        np.savez_compressed(capture_dir / f"wkv-{index:03d}.npz", **payload)

    @jax.custom_vjp
    def captured_wkv(r, w, k, v, neg_kk, kka, initial_state):
        return original_wkv7(
            r,
            w,
            k,
            v,
            neg_kk,
            kka,
            initial_state,
            backend="pallas_gpu_triton",
        )

    def captured_forward(r, w, k, v, neg_kk, kka, initial_state):
        inputs = (r, w, k, v, neg_kk, kka, initial_state)
        outputs = original_wkv7(
            *inputs,
            backend="pallas_gpu_triton",
        )
        return outputs, inputs

    def captured_backward(inputs, cotangents):
        jax.debug.callback(
            write_capture,
            *inputs,
            *cotangents,
            ordered=True,
        )
        _, pullback = jax.vjp(wkv7_reference, *inputs)
        return pullback(cotangents)

    captured_wkv.defvjp(captured_forward, captured_backward)

    def diagnostic_dispatch(
        r,
        w,
        k,
        v,
        neg_kk,
        kka,
        initial_state,
        backend=None,
        interpret=False,
    ):
        if backend == "diagnostic_capture":
            return captured_wkv(r, w, k, v, neg_kk, kka, initial_state)
        return original_wkv7(
            r,
            w,
            k,
            v,
            neg_kk,
            kka,
            initial_state,
            backend=backend,
            interpret=interpret,
        )

    nnx_model_module.wkv7 = diagnostic_dispatch


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
    if args.screening_eps is not None:
        config.screening.eps = args.screening_eps
    if args.screening_norm_eps is not None:
        config.screening.norm_eps = args.screening_norm_eps
    if args.float32_model:
        config.dtype = "float32"
        config.param_dtype = "float32"
    if any(index >= config.n_layers for index in args.reference_wkv_layers):
        raise SystemExit(
            "--reference-wkv-layers contains an index outside the model"
        )
    if (
        args.capture_wkv_layer is not None
        and args.capture_wkv_layer >= config.n_layers
    ):
        raise SystemExit("--capture-wkv-layer is outside the model")
    runtime, train_state = create_train_runtime(
        jax.random.key(args.seed),
        config,
        batch_size=batch_size,
        total_steps=args.optimizer_total_steps,
    )
    if checkpoint_path is not None:
        train_state, restored = restore_distributed_train_state(
            checkpoint_path,
            train_state,
        )
        checkpoint_step = restored.start_step
    if args.float32_model or args.float32_parameters:
        promoted_params = jax.tree.map(
            lambda value: value.astype(jnp.float32)
            if hasattr(value, "dtype")
            and jnp.issubdtype(value.dtype, jnp.inexact)
            else value,
            nnx.state(train_state.model, nnx.Param),
        )
        nnx.update(train_state.model, promoted_params)
    for layer_index in args.reference_wkv_layers:
        layer = getattr(train_state.model, f"layer_{layer_index}")
        block = getattr(layer, f"rwkv_block_{layer_index}")
        block.att.wkv_backend = "reference"
    if args.capture_wkv_layer is not None:
        _install_wkv_capture(args.capture_wkv_dir)
        layer = getattr(
            train_state.model,
            f"layer_{args.capture_wkv_layer}",
        )
        block = getattr(
            layer,
            f"rwkv_block_{args.capture_wkv_layer}",
        )
        block.att.wkv_backend = "diagnostic_capture"
    graphdef, params = nnx.split(train_state.model, nnx.Param)

    def loss_outputs(active_params):
        model = nnx.merge(graphdef, active_params)
        loss, (metrics, _, _) = nnx_model_loss(
            model,
            batch,
            runtime.initial_rwkv_state,
            runtime.initial_screen_state,
            phase=phase,
            deterministic=False,
            include_l2wrap=True,
            training_step=jnp.asarray(checkpoint_step, dtype=jnp.uint32),
        )
        return loss, metrics

    def loss_function(active_params):
        loss, _ = loss_outputs(active_params)
        return loss

    checkify_error = None
    if args.checkify_floats:
        checked_value_and_grad = checkify.checkify(
            jax.value_and_grad(loss_function),
            errors=checkify.float_checks,
        )
        error, (loss, gradients) = jax.jit(checked_value_and_grad)(params)
        checkify_error = error.get()
    else:
        loss, gradients = jax.jit(jax.value_and_grad(loss_function))(params)
    jax.block_until_ready((loss, gradients))
    summary = summarize_gradient_state(gradients, top_k=args.top_k)
    component_summaries = None
    if args.component_gradients:
        component_summaries = {}
        for component_name, metric_name in (
            ("cross_entropy", "loss"),
            ("admission_floor", "admission_floor_loss"),
            ("write_budget", "write_budget_loss"),
            ("self_index", "self_index_loss"),
        ):
            def component_loss(active_params, metric_name=metric_name):
                _, metrics = loss_outputs(active_params)
                return metrics[metric_name]

            component_value, component_gradient = jax.jit(
                jax.value_and_grad(component_loss)
            )(params)
            jax.block_until_ready((component_value, component_gradient))
            component_summaries[component_name] = {
                "value": float(component_value),
                "gradients": summarize_gradient_state(
                    component_gradient,
                    top_k=args.top_k,
                ),
            }

    update_summary = None
    if args.include_update:
        if args.float32_model or args.float32_parameters:
            raise SystemExit(
                "--include-update cannot be combined with a float32 "
                "diagnostic override"
            )
        _make_optimizer_state_writable(train_state.optimizer)
        before_params = jax.tree.map(
            _writable_device_copy,
            nnx.state(train_state.model, nnx.Param),
        )
        train_state.optimizer.update(train_state.model, gradients)
        after_params = nnx.state(train_state.model, nnx.Param)
        parameter_delta = _parameter_delta(before_params, after_params)
        jax.block_until_ready(parameter_delta)
        update_summary = summarize_gradient_state(
            parameter_delta,
            top_k=args.top_k,
        )

    lr_schedule = create_learning_rate_schedule(
        _optimizer_config(config),
        args.optimizer_total_steps,
    )
    payload = {
        "loss": float(loss),
        "loss_finite": bool(jnp.isfinite(loss)),
        "batch_size": int(batch_size),
        "tokens": int(token_count),
        "variant": args.variant,
        "model_config": str(Path(args.model_config)),
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "training_step": int(checkpoint_step),
        "learning_rate": float(lr_schedule(checkpoint_step)),
        "optimizer_total_steps": int(args.optimizer_total_steps),
        "sequence_chunk_size": config.sequence_chunk_size,
        "screening_eps": config.screening.eps,
        "screening_norm_eps": config.screening.norm_eps,
        "reference_wkv_layers": list(args.reference_wkv_layers),
        "captured_wkv_layer": args.capture_wkv_layer,
        "wkv_capture_dir": (
            None
            if args.capture_wkv_dir is None
            else str(args.capture_wkv_dir)
        ),
        "float32_model": bool(args.float32_model),
        "float32_parameters": bool(args.float32_parameters),
        "devices": [
            {
                "platform": device.platform,
                "device_kind": device.device_kind,
                "id": device.id,
            }
            for device in jax.devices()
        ],
        "gradients": summary,
        "component_gradients": component_summaries,
        "checkify_float_error": checkify_error,
        "parameter_update": update_summary,
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
