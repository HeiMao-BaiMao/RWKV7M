"""Benchmark full-logits and vocabulary-tiled training-head compute."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import jax
import jax.numpy as jnp

from rwkv7m.kernels import resolve_training_loss_backend
from rwkv7m.model.losses import cross_entropy_components, l2wrap_components
from rwkv7m.model.training_head import tiled_training_loss_components


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--tile-size", type=int, default=4096)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if min(args.positions, args.hidden_size, args.vocab_size, args.tile_size) <= 0:
        parser.error("all shape arguments must be positive")
    if args.vocab_size > args.tile_size and args.vocab_size % args.tile_size:
        parser.error("--vocab-size must be divisible by --tile-size")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be non-negative and --iterations positive")
    return args


def _summary(samples):
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.fmean(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _measure(function, arguments, *, warmup, iterations):
    for _ in range(warmup):
        jax.block_until_ready(function(*arguments))
    samples = []
    result = None
    for _ in range(iterations):
        started = time.perf_counter_ns()
        result = function(*arguments)
        jax.block_until_ready(result)
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return _summary(samples), result


def main(argv=None):
    args = parse_args(argv)
    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    backend = resolve_training_loss_backend(args.backend)
    hidden_key, kernel_key = jax.random.split(jax.random.key(20260716))
    hidden = (
        jax.random.normal(hidden_key, (args.positions, args.hidden_size)) * 0.02
    ).astype(dtype)
    kernel = (
        jax.random.normal(kernel_key, (args.hidden_size, args.vocab_size)) * 0.02
    ).astype(dtype)
    targets = jnp.arange(args.positions, dtype=jnp.int32) % args.vocab_size
    mask = jnp.ones((args.positions,), dtype=jnp.float32)

    def full_components(active_hidden, active_kernel):
        logits = active_hidden @ active_kernel
        ce_total, ce_count = cross_entropy_components(logits, targets, mask)
        l2_total, l2_count = l2wrap_components(logits)
        return ce_total, ce_count, l2_total, l2_count

    def tiled_components(active_hidden, active_kernel):
        return tiled_training_loss_components(
            active_hidden,
            active_kernel,
            targets,
            mask,
            args.tile_size,
            backend,
        )

    def objective(function, active_hidden, active_kernel):
        ce_total, ce_count, l2_total, l2_count = function(
            active_hidden, active_kernel
        )
        return ce_total / ce_count + l2_total / l2_count

    full_forward = jax.jit(full_components)
    tiled_forward = jax.jit(tiled_components)
    full_forward_backward = jax.jit(
        jax.value_and_grad(
            lambda active_hidden, active_kernel: objective(
                full_components, active_hidden, active_kernel
            ),
            argnums=(0, 1),
        )
    )
    tiled_forward_backward = jax.jit(
        jax.value_and_grad(
            lambda active_hidden, active_kernel: objective(
                tiled_components, active_hidden, active_kernel
            ),
            argnums=(0, 1),
        )
    )
    arguments = (hidden, kernel)
    full_outputs = full_forward(*arguments)
    tiled_outputs = tiled_forward(*arguments)
    full_loss_and_gradients = full_forward_backward(*arguments)
    tiled_loss_and_gradients = tiled_forward_backward(*arguments)
    jax.block_until_ready(
        (
            full_outputs,
            tiled_outputs,
            full_loss_and_gradients,
            tiled_loss_and_gradients,
        )
    )
    forward_error = max(
        float(jnp.max(jnp.abs(left - right)))
        for left, right in zip(full_outputs, tiled_outputs, strict=True)
    )
    gradient_error = max(
        float(
            jnp.max(
                jnp.abs(left.astype(jnp.float32) - right.astype(jnp.float32))
            )
        )
        for left, right in zip(
            full_loss_and_gradients[1],
            tiled_loss_and_gradients[1],
            strict=True,
        )
    )
    timings = {}
    for name, function in (
        ("full_forward", full_forward),
        ("tiled_forward", tiled_forward),
        ("full_forward_backward", full_forward_backward),
        ("tiled_forward_backward", tiled_forward_backward),
    ):
        timings[name], _ = _measure(
            function,
            arguments,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    timings["forward_speedup"] = (
        timings["full_forward"]["median_ms"]
        / timings["tiled_forward"]["median_ms"]
    )
    timings["forward_backward_speedup"] = (
        timings["full_forward_backward"]["median_ms"]
        / timings["tiled_forward_backward"]["median_ms"]
    )
    element_bytes = jnp.dtype(dtype).itemsize
    report = {
        "benchmark": "rwkv7m_training_head_accelerator",
        "jax": jax.__version__,
        "jaxlib": jax.lib.__version__,
        "platform": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "backend": backend,
        "shape": {
            "positions": args.positions,
            "hidden_size": args.hidden_size,
            "vocab_size": args.vocab_size,
            "tile_size": min(args.tile_size, args.vocab_size),
            "dtype": args.dtype,
        },
        "largest_logit_tensor_bytes": {
            "full": args.positions * args.vocab_size * element_bytes,
            "tiled": (
                args.positions
                * min(args.tile_size, args.vocab_size)
                * element_bytes
            ),
        },
        "parity": {
            "maximum_component_absolute_error": forward_error,
            "maximum_gradient_absolute_error": gradient_error,
        },
        "method": {
            "fixed_device_resident_inputs": True,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "synchronization": "all output leaves after every call",
            "compilation_excluded": True,
        },
        "timings": timings,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
