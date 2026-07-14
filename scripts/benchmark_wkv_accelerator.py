"""Benchmark the production WKV backend against the portable reference.

The script deliberately measures one synchronized call at a time.  This
matches the TPU validation method and avoids reporting queued asynchronous
dispatch as accelerator execution time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from benchmark_common import gc_policy

from rwkv7m.kernels import resolve_wkv_backend
from rwkv7m.model.wkv import wkv7, wkv7_reference


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare Pallas WKV with the lax.scan reference."
    )
    parser.add_argument("--time", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float32"], default="bfloat16"
    )
    parser.add_argument("--backend", default=None)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--initial-state", choices=("random", "zero"), default="random"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--disable-python-gc", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    for name in ("time", "batch", "heads", "head_size", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


def _git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing_summary(milliseconds):
    return {
        "iterations": len(milliseconds),
        "mean_ms": statistics.fmean(milliseconds),
        "median_ms": statistics.median(milliseconds),
        "stdev_ms": (
            statistics.stdev(milliseconds) if len(milliseconds) > 1 else 0.0
        ),
        "min_ms": min(milliseconds),
        "p95_ms": _percentile(milliseconds, 0.95),
        "max_ms": max(milliseconds),
    }


def _measure(function, inputs, *, warmup, iterations):
    for _ in range(warmup):
        jax.block_until_ready(function(*inputs))
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        result = function(*inputs)
        jax.block_until_ready(result)
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return _timing_summary(samples)


def _make_inputs(args):
    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    rng = np.random.default_rng(args.seed)
    vector_shape = (args.time, args.batch, args.heads, args.head_size)
    numpy_vectors = tuple(
        rng.standard_normal(vector_shape, dtype=np.float32) * 0.03
        for _ in range(6)
    )
    vectors = tuple(jax.device_put(value).astype(dtype) for value in numpy_vectors)
    state_values = (
        np.zeros(
            (args.batch, args.heads, args.head_size, args.head_size),
            dtype=np.float32,
        )
        if args.initial_state == "zero"
        else rng.standard_normal(
            (args.batch, args.heads, args.head_size, args.head_size),
            dtype=np.float32,
        )
        * 0.03
    )
    state = jax.device_put(state_values)
    fingerprint = hashlib.sha256(
        b"".join(np.ascontiguousarray(value).tobytes() for value in numpy_vectors)
    ).hexdigest()
    return (*vectors, state), fingerprint


def _forward_error(actual, expected):
    names = ("activation", "final_state")
    return {
        name: float(
            jnp.max(
                jnp.abs(
                    left.astype(jnp.float32) - right.astype(jnp.float32)
                )
            )
        )
        for name, left, right in zip(names, actual, expected, strict=True)
    }


def _gradient_error(actual, expected):
    names = ("r", "w", "k", "v", "neg_kk", "kka", "initial_state")
    result = {}
    for name, left, right in zip(names, actual, expected, strict=True):
        left_f32 = left.astype(jnp.float32)
        right_f32 = right.astype(jnp.float32)
        difference = left_f32 - right_f32
        result[name] = {
            "max_abs": float(jnp.max(jnp.abs(difference))),
            "relative_l2": float(
                jnp.linalg.norm(difference)
                / jnp.maximum(jnp.linalg.norm(right_f32), 1e-12)
            ),
            "allclose": bool(
                jnp.allclose(left_f32, right_f32, rtol=3e-2, atol=3e-2)
            ),
        }
    return result


def main(argv=None):
    args = parse_args(argv)
    selected_backend = resolve_wkv_backend(args.backend)
    if not selected_backend.startswith("pallas_"):
        raise RuntimeError(
            "the comparison backend must be a Pallas accelerator backend, "
            f"got {selected_backend!r}"
        )

    inputs, input_fingerprint = _make_inputs(args)
    pallas_forward = jax.jit(
        lambda *values: wkv7(*values, selected_backend)
    )
    reference_forward = jax.jit(wkv7_reference)
    if selected_backend in ("pallas_gpu_mosaic", "pallas_gpu_triton"):
        from rwkv7m.kernels.wkv_pallas_gpu import (
            wkv7_pallas_gpu_forward_with_aux,
        )

        lowering = (
            "mosaic" if selected_backend == "pallas_gpu_mosaic" else "triton"
        )
        pallas_training_forward = jax.jit(
            lambda *values: wkv7_pallas_gpu_forward_with_aux(
                *values, lowering=lowering
            )
        )
    else:
        from rwkv7m.kernels.wkv_pallas_tpu import (
            wkv7_pallas_tpu_forward_with_aux,
        )

        pallas_training_forward = jax.jit(
            wkv7_pallas_tpu_forward_with_aux
        )

    def objective(function, *values):
        activations, _ = function(*values)
        return jnp.sum(jnp.square(activations.astype(jnp.float32)))

    gradient_argnums = tuple(range(7))
    pallas_objective = jax.jit(
        lambda *values: objective(
            lambda *items: wkv7(*items, selected_backend), *values
        )
    )
    reference_objective = jax.jit(
        lambda *values: objective(wkv7_reference, *values)
    )
    pallas_train = jax.jit(
        jax.value_and_grad(
            pallas_objective,
            argnums=gradient_argnums,
        )
    )
    reference_train = jax.jit(
        jax.value_and_grad(
            reference_objective,
            argnums=gradient_argnums,
        )
    )

    reference_outputs = reference_forward(*inputs)
    pallas_outputs = pallas_forward(*inputs)
    reference_loss, reference_gradients = reference_train(*inputs)
    pallas_loss, pallas_gradients = pallas_train(*inputs)
    jax.block_until_ready(
        (
            reference_outputs,
            pallas_outputs,
            reference_loss,
            reference_gradients,
            pallas_loss,
            pallas_gradients,
        )
    )

    pallas_timing_objective = jax.jit(
        lambda *vectors: objective(
            lambda *items: wkv7(*items, inputs[-1], selected_backend),
            *vectors,
        )
    )
    reference_timing_objective = jax.jit(
        lambda *vectors: objective(
            lambda *items: wkv7_reference(*items, inputs[-1]),
            *vectors,
        )
    )
    pallas_timing_train = jax.jit(
        jax.value_and_grad(
            pallas_timing_objective, argnums=tuple(range(6))
        )
    )
    reference_timing_train = jax.jit(
        jax.value_and_grad(
            reference_timing_objective, argnums=tuple(range(6))
        )
    )
    pallas_vjp_loss, pallas_pullback = jax.vjp(
        pallas_timing_objective, *inputs[:6]
    )
    reference_vjp_loss, reference_pullback = jax.vjp(
        reference_timing_objective, *inputs[:6]
    )
    loss_cotangent = jnp.ones_like(pallas_vjp_loss)
    jax.block_until_ready(
        (
            pallas_pullback(loss_cotangent),
            reference_pullback(jnp.ones_like(reference_vjp_loss)),
        )
    )

    with gc_policy(args.disable_python_gc):
        timings = {
            "pallas_forward": _measure(
                pallas_forward,
                inputs,
                warmup=args.warmup,
                iterations=args.iterations,
            ),
            "reference_forward": _measure(
                reference_forward,
                inputs,
                warmup=args.warmup,
                iterations=args.iterations,
            ),
            "pallas_training_forward": _measure(
                pallas_training_forward,
                inputs,
                warmup=args.warmup,
                iterations=args.iterations,
            ),
            "pallas_backward": _measure(
                pallas_pullback,
                (loss_cotangent,),
                warmup=args.warmup,
                iterations=args.iterations,
            ),
            "reference_backward": _measure(
                reference_pullback,
                (jnp.ones_like(reference_vjp_loss),),
                warmup=args.warmup,
                iterations=args.iterations,
            ),
            "pallas_forward_backward": _measure(
                pallas_timing_train,
                inputs[:6],
                warmup=args.warmup,
                iterations=args.iterations,
            ),
            "reference_forward_backward": _measure(
                reference_timing_train,
                inputs[:6],
                warmup=args.warmup,
                iterations=args.iterations,
            ),
        }
    timings["forward_speedup"] = (
        timings["reference_forward"]["median_ms"]
        / timings["pallas_forward"]["median_ms"]
    )
    timings["forward_backward_speedup"] = (
        timings["reference_forward_backward"]["median_ms"]
        / timings["pallas_forward_backward"]["median_ms"]
    )

    devices = jax.devices()
    payload = {
        "schema_version": 2,
        "benchmark_kind": "wkv_compute_only",
        "framework": "jax_pallas",
        "recorded_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_revision": _git_revision(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jax.lib.__version__,
        "backend": selected_backend,
        "devices": [
            {
                "platform": device.platform,
                "device_kind": device.device_kind,
                "id": device.id,
            }
            for device in devices
        ],
        "shape": {
            "time": args.time,
            "batch": args.batch,
            "heads": args.heads,
            "head_size": args.head_size,
            "dtype": args.dtype,
        },
        "inputs": {
            "numpy_rng": "PCG64",
            "seed": args.seed,
            "sha256_float32_before_bf16_cast": input_fingerprint,
            "initial_state": args.initial_state,
        },
        "method": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "synchronized_each_iteration": True,
            "fixed_inputs": True,
            "inputs_device_resident_before_warmup": True,
            "initial_state": args.initial_state,
            "numpy_rng": "PCG64",
            "python_gc_disabled": args.disable_python_gc,
            "backward_objective": "sum(square(float32(activations)))",
            "timed_gradients": ["r", "w", "k", "v", "a", "b"],
            "excluded": [
                "input_generation",
                "host_to_device_transfer",
                "compilation",
                "logging",
            ],
            "phase_windows": (
                "backward reuses one precomputed VJP residual; combined "
                "forward_backward is measured independently"
            ),
            "forward_fields": {
                "pallas_forward": "inference forward without saved tape",
                "pallas_training_forward": (
                    "forward including saved backward checkpoints and sa tape"
                ),
            },
        },
        "correctness": {
            "forward_max_abs": _forward_error(
                pallas_outputs, reference_outputs
            ),
            "loss_abs": float(jnp.abs(pallas_loss - reference_loss)),
            "gradients": _gradient_error(
                pallas_gradients, reference_gradients
            ),
        },
        "timings": timings,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
