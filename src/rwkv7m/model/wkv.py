"""Pallas-first WKV recurrence API with a portable reference path."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from rwkv7m.kernels.wkv_backend import resolve_wkv_backend
from rwkv7m.kernels.wkv_ffi import require_wkv_ffi_backend

from .rwkv_core import wkv_step


_SUPPORTED_INPUT_DTYPES = (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32))


def _validate_wkv_inputs(r, w, k, v, neg_kk, kka, initial_state):
    vectors = {
        "r": r,
        "w": w,
        "k": k,
        "v": v,
        "neg_kk": neg_kk,
        "kka": kka,
    }
    if initial_state.ndim != 4:
        raise ValueError(
            "initial_state must have shape [batch, heads, rows, columns]"
        )
    if initial_state.shape[-1] != initial_state.shape[-2]:
        raise ValueError("initial_state WKV rows and columns must be equal")
    if initial_state.dtype != jnp.float32:
        raise TypeError(
            f"initial_state must be float32, got {initial_state.dtype}"
        )
    if r.ndim != 4:
        raise ValueError(
            "r must have shape [time, batch, heads, head_size], "
            f"got {r.shape}"
        )
    expected_shape = (r.shape[0], *initial_state.shape[:3])
    for name, value in vectors.items():
        if value.ndim != 4 or value.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {value.shape}"
            )
        if value.dtype != r.dtype:
            raise TypeError(
                f"all WKV vectors must use {r.dtype}, got {name}={value.dtype}"
            )
    if r.dtype not in _SUPPORTED_INPUT_DTYPES:
        raise TypeError(
            "WKV vectors must be bfloat16 or float32, "
            f"got {r.dtype}"
        )


def _wkv7_reference_impl(r, w, k, v, neg_kk, kka, initial_state):
    inputs = (r, w, k, v, neg_kk, kka)
    final_state, y = jax.lax.scan(
        lambda carry, values: wkv_step(carry, *values),
        initial_state,
        inputs,
    )
    return y.astype(r.dtype), final_state


def wkv7_reference(r, w, k, v, neg_kk, kka, initial_state):
    """Execute the portable time-major WKV recurrence.

    Vector inputs use shape ``[time, batch, heads, head_size]``. The state uses
    ``[batch, heads, head_size, head_size]`` and remains FP32. The activation
    output has the same dtype as the vector inputs.
    """

    _validate_wkv_inputs(r, w, k, v, neg_kk, kka, initial_state)
    return _wkv7_reference_impl(r, w, k, v, neg_kk, kka, initial_state)


def wkv7_sharded(
    r,
    w,
    k,
    v,
    neg_kk,
    kka,
    initial_state,
    *,
    mesh,
    data_axis: str = "data",
    model_axis: str = "model",
):
    """Run WKV on per-device shards under manual Pallas mesh axes.

    JAX 0.10 requires every mesh axis surrounding ``pallas_call`` to be
    manual. The outer model uses explicit sharding, so this narrow shard-map
    boundary converts only WKV to a device-local view while preserving the
    global input/output partition contract.
    """

    vector_spec = jax.sharding.PartitionSpec(
        None, data_axis, model_axis, None
    )
    state_spec = jax.sharding.PartitionSpec(
        data_axis, model_axis, None, None
    )
    mapped_wkv7 = jax.shard_map(
        wkv7,
        mesh=mesh,
        in_specs=(
            vector_spec,
            vector_spec,
            vector_spec,
            vector_spec,
            vector_spec,
            vector_spec,
            state_spec,
        ),
        out_specs=(vector_spec, state_spec),
        check_vma=False,
    )
    return mapped_wkv7(r, w, k, v, neg_kk, kka, initial_state)


def _wkv7_forward_dispatch(
    r,
    w,
    k,
    v,
    neg_kk,
    kka,
    initial_state,
    *,
    backend: str | None,
    interpret: bool,
    with_aux: bool,
):
    selected = resolve_wkv_backend(backend)
    inputs = (r, w, k, v, neg_kk, kka, initial_state)
    if selected == "reference":
        outputs = _wkv7_reference_impl(*inputs)
        return (outputs, ()) if with_aux else outputs
    if selected in ("pallas_gpu_mosaic", "pallas_gpu_triton"):
        from rwkv7m.kernels.wkv_pallas_gpu import (
            wkv7_pallas_gpu_forward,
            wkv7_pallas_gpu_forward_with_aux,
        )

        lowering = (
            "mosaic" if selected == "pallas_gpu_mosaic" else "triton"
        )
        if with_aux:
            return wkv7_pallas_gpu_forward_with_aux(
                *inputs, lowering=lowering, interpret=interpret
            )
        return wkv7_pallas_gpu_forward(
            *inputs, lowering=lowering, interpret=interpret
        )
    if selected == "pallas_tpu":
        from rwkv7m.kernels.wkv_pallas_tpu import (
            wkv7_pallas_tpu_forward,
            wkv7_pallas_tpu_forward_with_aux,
        )

        if with_aux:
            return wkv7_pallas_tpu_forward_with_aux(
                *inputs, interpret=interpret
            )
        return wkv7_pallas_tpu_forward(*inputs, interpret=interpret)
    ffi_backend = require_wkv_ffi_backend()
    if with_aux:
        return ffi_backend.forward_with_aux(*inputs)
    return ffi_backend.forward(*inputs)


def _wkv7_backward_dispatch(
    inputs,
    cotangents,
    aux,
    *,
    backend: str | None,
    interpret: bool,
):
    selected = resolve_wkv_backend(backend)
    if selected == "reference":
        _, pullback = jax.vjp(_wkv7_reference_impl, *inputs)
        return pullback(cotangents)
    y_cotangent, final_state_cotangent = cotangents
    if selected in ("pallas_gpu_mosaic", "pallas_gpu_triton"):
        from rwkv7m.kernels.wkv_pallas_gpu import wkv7_pallas_gpu_backward

        lowering = (
            "mosaic" if selected == "pallas_gpu_mosaic" else "triton"
        )
        return wkv7_pallas_gpu_backward(
            *inputs[:-1],
            y_cotangent,
            final_state_cotangent,
            *aux,
            lowering=lowering,
            interpret=interpret,
        )
    if selected == "pallas_tpu":
        from rwkv7m.kernels.wkv_pallas_tpu import wkv7_pallas_tpu_backward

        return wkv7_pallas_tpu_backward(
            *inputs[:-1],
            y_cotangent,
            final_state_cotangent,
            *aux,
            interpret=interpret,
        )
    ffi_backend = require_wkv_ffi_backend()
    return ffi_backend.backward(*inputs, *cotangents, *aux)


@partial(jax.custom_vjp, nondiff_argnums=(7, 8))
def wkv7(
    r,
    w,
    k,
    v,
    neg_kk,
    kka,
    initial_state,
    backend: str | None = None,
    interpret: bool = False,
):
    """Execute WKV using Pallas on accelerators and reference code on CPU.

    ``backend`` is normally left as ``None``/``auto``. Explicit backend and
    interpret arguments exist for parity tests, benchmarking, and controlled
    rollback. FFI must be explicitly selected and registered.
    """

    _validate_wkv_inputs(r, w, k, v, neg_kk, kka, initial_state)
    return _wkv7_forward_dispatch(
        r,
        w,
        k,
        v,
        neg_kk,
        kka,
        initial_state,
        backend=backend,
        interpret=interpret,
        with_aux=False,
    )


def _wkv7_fwd(
    r,
    w,
    k,
    v,
    neg_kk,
    kka,
    initial_state,
    backend,
    interpret,
):
    _validate_wkv_inputs(r, w, k, v, neg_kk, kka, initial_state)
    inputs = (r, w, k, v, neg_kk, kka, initial_state)
    outputs, aux = _wkv7_forward_dispatch(
        *inputs,
        backend=backend,
        interpret=interpret,
        with_aux=True,
    )
    return outputs, (inputs, aux)


def _wkv7_bwd(backend, interpret, residuals, cotangents):
    inputs, aux = residuals
    return _wkv7_backward_dispatch(
        inputs,
        cotangents,
        aux,
        backend=backend,
        interpret=interpret,
    )


wkv7.defvjp(_wkv7_fwd, _wkv7_bwd)


__all__ = ["wkv7", "wkv7_reference", "wkv7_sharded"]
