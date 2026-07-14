"""Benchmark the production WKV backend against the portable reference.

The script deliberately measures one synchronized call at a time.  This
matches the TPU validation method and avoids reporting queued asynchronous
dispatch as accelerator execution time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time

import jax
import jax.numpy as jnp

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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
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
    keys = jax.random.split(jax.random.key(args.seed), 7)
    vector_shape = (args.time, args.batch, args.heads, args.head_size)
    vectors = tuple(
        (jax.random.normal(key, vector_shape) * 0.03).astype(dtype)
        for key in keys[:6]
    )
    state = (
        jax.random.normal(
            keys[6],
            (args.batch, args.heads, args.head_size, args.head_size),
        )
        * 0.03
    ).astype(jnp.float32)
    return (*vectors, state)


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

    inputs = _make_inputs(args)
    pallas_forward = jax.jit(
        lambda *values: wkv7(*values, selected_backend)
    )
    reference_forward = jax.jit(wkv7_reference)

    def objective(function, *values):
        activations, _ = function(*values)
        return jnp.sum(jnp.square(activations.astype(jnp.float32)))

    gradient_argnums = tuple(range(7))
    pallas_train = jax.jit(
        jax.value_and_grad(
            lambda *values: objective(
                lambda *items: wkv7(*items, selected_backend), *values
            ),
            argnums=gradient_argnums,
        )
    )
    reference_train = jax.jit(
        jax.value_and_grad(
            lambda *values: objective(wkv7_reference, *values),
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
        "pallas_forward_backward": _measure(
            pallas_train,
            inputs,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "reference_forward_backward": _measure(
            reference_train,
            inputs,
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
        "schema_version": 1,
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
        "method": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "synchronized_each_iteration": True,
            "backward_objective": "sum(square(float32(activations)))",
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
