"""Compare Pallas and portable WKV pullbacks on a captured training call."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rwkv7m.model.wkv import wkv7, wkv7_reference


_VECTOR_NAMES = ("r", "w", "k", "v", "neg_kk", "kka")
_GRADIENT_NAMES = (*_VECTOR_NAMES, "initial_state")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--backend", default="pallas_gpu_triton")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-parity",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args(argv)


def _summary(value):
    value = value.astype(jnp.float32)
    finite = jnp.isfinite(value)
    finite_value = jnp.where(finite, value, 0.0)
    return {
        "nonfinite_count": int(jnp.sum(~finite)),
        "max_abs_finite": float(
            jnp.max(jnp.abs(finite_value), initial=0.0)
        ),
        "l2_norm_finite": float(jnp.linalg.norm(finite_value)),
    }


def _comparison(actual, expected):
    actual_f32 = actual.astype(jnp.float32)
    expected_f32 = expected.astype(jnp.float32)
    both_finite = jnp.isfinite(actual_f32) & jnp.isfinite(expected_f32)
    difference = jnp.where(both_finite, actual_f32 - expected_f32, 0.0)
    return {
        "pallas": _summary(actual),
        "reference": _summary(expected),
        "max_abs_finite_difference": float(
            jnp.max(jnp.abs(difference), initial=0.0)
        ),
        "relative_l2_finite_difference": float(
            jnp.linalg.norm(difference)
            / jnp.maximum(
                jnp.linalg.norm(jnp.where(both_finite, expected_f32, 0.0)),
                1e-12,
            )
        ),
    }


def main(argv=None):
    args = parse_args(argv)
    with np.load(args.capture, allow_pickle=False) as archive:
        inputs = tuple(
            jax.device_put(archive[name]).astype(jnp.bfloat16)
            for name in _VECTOR_NAMES
        ) + (jax.device_put(archive["initial_state"]).astype(jnp.float32),)
        cotangents = (
            jax.device_put(archive["y_cotangent"]).astype(jnp.bfloat16),
            jax.device_put(archive["final_state_cotangent"]).astype(
                jnp.float32
            ),
        )

    def evaluate(function, values):
        outputs, pullback = jax.vjp(function, *values)
        return outputs, pullback(cotangents)

    pallas = jax.jit(
        lambda *values: evaluate(
            lambda *items: wkv7(*items, backend=args.backend),
            values,
        )
    )(*inputs)
    reference = jax.jit(
        lambda *values: evaluate(wkv7_reference, values)
    )(*inputs)
    jax.block_until_ready((pallas, reference))
    pallas_outputs, pallas_gradients = pallas
    reference_outputs, reference_gradients = reference
    payload = {
        "capture": str(args.capture),
        "backend": args.backend,
        "shape": {
            "time": inputs[0].shape[0],
            "batch": inputs[0].shape[1],
            "heads": inputs[0].shape[2],
            "head_size": inputs[0].shape[3],
        },
        "inputs": {
            name: _summary(value)
            for name, value in zip(
                (*_GRADIENT_NAMES, "y_cotangent", "final_state_cotangent"),
                (*inputs, *cotangents),
                strict=True,
            )
        },
        "outputs": {
            "activation": _comparison(
                pallas_outputs[0], reference_outputs[0]
            ),
            "final_state": _comparison(
                pallas_outputs[1], reference_outputs[1]
            ),
        },
        "gradients": {
            name: _comparison(actual, expected)
            for name, actual, expected in zip(
                _GRADIENT_NAMES,
                pallas_gradients,
                reference_gradients,
                strict=True,
            )
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.require_parity and any(
        value["pallas"]["nonfinite_count"] > 0
        for value in payload["gradients"].values()
    ):
        raise SystemExit("captured Pallas WKV pullback is non-finite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
