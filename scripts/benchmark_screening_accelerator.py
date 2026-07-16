"""Compare the Pallas screening recurrence with its portable reference."""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from benchmark_common import gc_policy, timing_summary

from rwkv7m.kernels.screening_backend import resolve_screening_backend
from rwkv7m.model.screening_recurrence import (
    ScreeningRecurrenceConfig,
    screening_recurrence,
    screening_recurrence_reference,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--slot-size", type=int, default=128)
    parser.add_argument("--key-size", type=int, default=64)
    parser.add_argument("--value-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--disable-python-gc", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    for name in (
        "time",
        "batch",
        "slots",
        "slot_size",
        "key_size",
        "value_size",
        "iterations",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


def _config():
    return ScreeningRecurrenceConfig(
        write_enabled=True,
        use_value_unit_norm=True,
        use_leaky_warmup=False,
        leaky_alpha=0.0,
        leaky_gamma=8.0,
        use_age_mask=False,
        age_ref=32.0,
        age_sigma=8.0,
        write_rel_floor=1e-3,
        usage_ema_decay=0.99,
        tanh_norm_cap=1.0,
        eps=1e-6,
    )


def _inputs(args):
    rng = np.random.default_rng(args.seed)

    def normal(shape, *, dtype=jnp.bfloat16, scale=0.1):
        values = rng.standard_normal(shape, dtype=np.float32) * scale
        return jax.device_put(values).astype(dtype)

    q_shape = (args.time, args.batch, args.key_size)
    state_prefix = (args.batch, args.slots)
    q_read = normal(q_shape, dtype=jnp.float32)
    q_write = normal(q_shape, dtype=jnp.float32)
    q_read /= jnp.linalg.norm(q_read, axis=-1, keepdims=True) + 1e-6
    q_write /= jnp.linalg.norm(q_write, axis=-1, keepdims=True) + 1e-6
    return (
        q_read,
        q_write,
        normal((args.time, *state_prefix, args.slot_size)),
        normal((args.time, *state_prefix, args.key_size)),
        normal((args.time, *state_prefix, args.value_size)),
        normal((args.time, *state_prefix, args.key_size)),
        normal((*state_prefix, args.slot_size), dtype=jnp.float32),
        normal((*state_prefix, args.key_size)),
        normal((*state_prefix, args.value_size)),
        normal((*state_prefix, args.key_size)),
        jnp.zeros(state_prefix, dtype=jnp.float32),
        jnp.zeros(state_prefix, dtype=jnp.float32),
        jnp.linspace(0.005, 0.05, args.slots, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )


def _loss(function, config, *inputs):
    u, slots, ages, usage, _, _ = function(*inputs, config)
    return (
        jnp.sum(u.astype(jnp.float32) ** 2)
        + 0.01 * jnp.sum(slots**2)
        + 0.001 * jnp.sum(ages**2)
        + 0.01 * jnp.sum(usage**2)
    )


def _measure(function, arguments, *, warmup, iterations):
    for _ in range(warmup):
        jax.block_until_ready(function(*arguments))
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        result = function(*arguments)
        jax.block_until_ready(result)
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return timing_summary(samples)


def _errors(actual, expected, names):
    errors = {}
    for name, left, right in zip(names, actual, expected, strict=True):
        difference = left.astype(jnp.float32) - right.astype(jnp.float32)
        errors[name] = {
            "max_abs": float(jnp.max(jnp.abs(difference))),
            "relative_l2": float(
                jnp.linalg.norm(difference)
                / jnp.maximum(jnp.linalg.norm(right.astype(jnp.float32)), 1e-12)
            ),
        }
    return errors


def main(argv=None):
    args = parse_args(argv)
    backend = resolve_screening_backend(args.backend)
    if not backend.startswith("pallas_"):
        raise RuntimeError(f"expected a Pallas accelerator backend, got {backend}")
    inputs = _inputs(args)
    config = _config()
    pallas_forward = jax.jit(
        lambda *values: screening_recurrence(*values, config, backend)
    )
    reference_forward = jax.jit(
        lambda *values: screening_recurrence_reference(*values, config)
    )
    argnums = tuple(range(15))
    pallas_train = jax.jit(
        jax.value_and_grad(
            lambda *values: _loss(screening_recurrence, config, *values),
            argnums=argnums,
        )
    )
    reference_train = jax.jit(
        jax.value_and_grad(
            lambda *values: _loss(
                screening_recurrence_reference, config, *values
            ),
            argnums=argnums,
        )
    )

    pallas_outputs = pallas_forward(*inputs)
    reference_outputs = reference_forward(*inputs)
    pallas_loss, pallas_gradients = pallas_train(*inputs)
    reference_loss, reference_gradients = reference_train(*inputs)
    jax.block_until_ready(
        (pallas_outputs, reference_outputs, pallas_gradients, reference_gradients)
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

    report = {
        "backend": backend,
        "device": str(jax.devices()[0]),
        "jax_version": jax.__version__,
        "shape": {
            "time": args.time,
            "batch": args.batch,
            "slots": args.slots,
            "slot_size": args.slot_size,
            "key_size": args.key_size,
            "value_size": args.value_size,
            "vector_dtype": "bfloat16",
            "state_dtype": "float32",
        },
        "measurement": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "synchronized_each_iteration": True,
            "python_gc_disabled": args.disable_python_gc,
        },
        "output_errors": _errors(
            pallas_outputs,
            reference_outputs,
            ("u", "slots", "ages", "usage", "statistics", "update_squared"),
        ),
        "gradient_errors": _errors(
            pallas_gradients,
            reference_gradients,
            (
                "q_read",
                "q_write",
                "delta_slots",
                "delta_read_keys",
                "delta_values",
                "delta_write_keys",
                "initial_slots",
                "initial_read_keys",
                "initial_values",
                "initial_write_keys",
                "initial_ages",
                "initial_usage",
                "mu",
                "tau_read",
                "tau_write",
            ),
        ),
        "loss_difference": float(jnp.abs(pallas_loss - reference_loss)),
        "timings": timings,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
