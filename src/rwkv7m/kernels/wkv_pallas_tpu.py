"""Persistent Pallas WKV kernels specialized for TPU VMEM blocks.

One Pallas program owns a complete batch item's head set. This keeps the last
two state dimensions full-sized, satisfying TPU block alignment rules without
sharing the GPU kernel body.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


@dataclass(frozen=True)
class WKVTPUConfig:
    checkpoint_interval: int = 1


def _load_vector(ref, index):
    block = ref[pl.dslice(index, 1), :, :, :]
    return block[0, 0, :, :]


def _store_vector(ref, index, value):
    ref[pl.dslice(index, 1), :, :, :] = value[None, None, :, :]


def _load_state(ref):
    return ref[:][0, :, :, :]


def _store_state(ref, value):
    ref[:] = value[None, :, :, :]


def _load_checkpoint(ref, index):
    block = ref[pl.dslice(index, 1), :, :, :, :]
    return block[0, 0, :, :, :]


def _store_checkpoint(ref, index, value):
    ref[pl.dslice(index, 1), :, :, :, :] = value[
        None, None, :, :, :
    ]


def _validate_config(config: WKVTPUConfig) -> None:
    if config.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")


def _wkv_tpu_forward_kernel(
    r_ref,
    w_ref,
    k_ref,
    v_ref,
    neg_kk_ref,
    kka_ref,
    initial_state_ref,
    y_ref,
    final_state_ref,
    sa_ref,
    checkpoints_ref,
    *,
    time: int,
    checkpoint_interval: int,
    output_dtype,
):
    state = _load_state(initial_state_ref).astype(jnp.float32)
    _store_checkpoint(checkpoints_ref, 0, state)

    @pl.loop(0, time, init_carry=state)
    def time_loop(t, current_state):
        r_t = _load_vector(r_ref, t).astype(jnp.float32)
        w_t = _load_vector(w_ref, t).astype(jnp.float32)
        k_t = _load_vector(k_ref, t).astype(jnp.float32)
        v_t = _load_vector(v_ref, t).astype(jnp.float32)
        neg_kk_t = _load_vector(neg_kk_ref, t).astype(jnp.float32)
        kka_t = _load_vector(kka_ref, t).astype(jnp.float32)

        decay = jnp.exp(-jnp.exp(w_t))
        sa_t = jnp.sum(
            current_state * neg_kk_t[:, None, :], axis=-1
        )
        next_state = (
            current_state * decay[:, None, :]
            + sa_t[:, :, None] * kka_t[:, None, :]
            + v_t[:, :, None] * k_t[:, None, :]
        )
        y_t = jnp.sum(
            next_state * r_t[:, None, :], axis=-1
        )
        _store_vector(y_ref, t, y_t.astype(output_dtype))
        _store_vector(sa_ref, t, sa_t)

        save_checkpoint = (
            ((t + 1) % checkpoint_interval) == 0
        ) | ((t + 1) == time)

        @pl.when(save_checkpoint)
        def store_checkpoint():
            checkpoint = (t + checkpoint_interval) // checkpoint_interval
            _store_checkpoint(checkpoints_ref, checkpoint, next_state)

        return next_state

    _store_state(final_state_ref, time_loop)


def _wkv_tpu_inference_kernel(
    r_ref,
    w_ref,
    k_ref,
    v_ref,
    neg_kk_ref,
    kka_ref,
    initial_state_ref,
    y_ref,
    final_state_ref,
    *,
    time: int,
    output_dtype,
):
    state = _load_state(initial_state_ref).astype(jnp.float32)

    @pl.loop(0, time, init_carry=state)
    def time_loop(t, current_state):
        r_t = _load_vector(r_ref, t).astype(jnp.float32)
        w_t = _load_vector(w_ref, t).astype(jnp.float32)
        k_t = _load_vector(k_ref, t).astype(jnp.float32)
        v_t = _load_vector(v_ref, t).astype(jnp.float32)
        neg_kk_t = _load_vector(neg_kk_ref, t).astype(jnp.float32)
        kka_t = _load_vector(kka_ref, t).astype(jnp.float32)

        decay = jnp.exp(-jnp.exp(w_t))
        sa_t = jnp.sum(
            current_state * neg_kk_t[:, None, :], axis=-1
        )
        next_state = (
            current_state * decay[:, None, :]
            + sa_t[:, :, None] * kka_t[:, None, :]
            + v_t[:, :, None] * k_t[:, None, :]
        )
        _store_vector(
            y_ref,
            t,
            jnp.sum(next_state * r_t[:, None, :], axis=-1).astype(
                output_dtype
            ),
        )
        return next_state

    _store_state(final_state_ref, time_loop)


def _wkv_tpu_backward_kernel(
    r_ref,
    w_ref,
    k_ref,
    v_ref,
    neg_kk_ref,
    kka_ref,
    y_cotangent_ref,
    final_state_cotangent_ref,
    sa_ref,
    checkpoints_ref,
    r_gradient_ref,
    w_gradient_ref,
    k_gradient_ref,
    v_gradient_ref,
    neg_kk_gradient_ref,
    kka_gradient_ref,
    initial_state_gradient_ref,
    *,
    time: int,
    checkpoint_interval: int,
    output_dtype,
):
    checkpoint_count = (
        time + checkpoint_interval - 1
    ) // checkpoint_interval + 1
    state = _load_checkpoint(checkpoints_ref, checkpoint_count - 1)
    state_cotangent = _load_state(final_state_cotangent_ref)

    @pl.loop(0, time, init_carry=(state, state_cotangent))
    def reverse_time_loop(reverse_t, carry):
        current_state, current_state_cotangent = carry
        t = time - 1 - reverse_t
        load_checkpoint = (
            (t == time - 1)
            | (((t + 1) % checkpoint_interval) == 0)
        )
        checkpoint = (t + checkpoint_interval) // checkpoint_interval
        saved_state = _load_checkpoint(checkpoints_ref, checkpoint)
        current_state = jnp.where(
            load_checkpoint, saved_state, current_state
        )

        r_t = _load_vector(r_ref, t).astype(jnp.float32)
        w_t = _load_vector(w_ref, t).astype(jnp.float32)
        k_t = _load_vector(k_ref, t).astype(jnp.float32)
        v_t = _load_vector(v_ref, t).astype(jnp.float32)
        neg_kk_t = _load_vector(neg_kk_ref, t).astype(jnp.float32)
        kka_t = _load_vector(kka_ref, t).astype(jnp.float32)
        y_cotangent_t = _load_vector(y_cotangent_ref, t).astype(
            jnp.float32
        )
        sa_t = _load_vector(sa_ref, t)

        decay = jnp.exp(-jnp.exp(w_t))
        previous_state = _load_checkpoint(checkpoints_ref, t)

        r_gradient = jnp.sum(
            current_state * y_cotangent_t[:, :, None], axis=-2
        )
        state_gradient = (
            current_state_cotangent
            + y_cotangent_t[:, :, None] * r_t[:, None, :]
        )
        decay_gradient = jnp.sum(
            state_gradient * previous_state, axis=-2
        )
        sa_gradient = jnp.sum(
            state_gradient * kka_t[:, None, :], axis=-1
        )
        kka_gradient = jnp.sum(
            state_gradient * sa_t[:, :, None], axis=-2
        )
        v_gradient = jnp.sum(
            state_gradient * k_t[:, None, :], axis=-1
        )
        k_gradient = jnp.sum(
            state_gradient * v_t[:, :, None], axis=-2
        )
        neg_kk_gradient = jnp.sum(
            previous_state * sa_gradient[:, :, None], axis=-2
        )
        previous_state_cotangent = (
            state_gradient * decay[:, None, :]
            + sa_gradient[:, :, None] * neg_kk_t[:, None, :]
        )
        w_gradient = decay_gradient * (-jnp.exp(w_t) * decay)

        _store_vector(
            r_gradient_ref, t, r_gradient.astype(output_dtype)
        )
        _store_vector(
            w_gradient_ref, t, w_gradient.astype(output_dtype)
        )
        _store_vector(
            k_gradient_ref, t, k_gradient.astype(output_dtype)
        )
        _store_vector(
            v_gradient_ref, t, v_gradient.astype(output_dtype)
        )
        _store_vector(
            neg_kk_gradient_ref,
            t,
            neg_kk_gradient.astype(output_dtype),
        )
        _store_vector(
            kka_gradient_ref, t, kka_gradient.astype(output_dtype)
        )
        return previous_state, previous_state_cotangent

    _, initial_state_cotangent = reverse_time_loop
    _store_state(initial_state_gradient_ref, initial_state_cotangent)


def wkv7_pallas_tpu_forward_with_aux(
    r,
    w,
    k,
    v,
    neg_kk,
    kka,
    initial_state,
    *,
    config: WKVTPUConfig = WKVTPUConfig(),
    interpret: bool = False,
):
    """Run the TPU forward kernel and return state reconstruction metadata."""

    _validate_config(config)
    time, batch, heads, head_size = r.shape
    checkpoint_count = (
        time + config.checkpoint_interval - 1
    ) // config.checkpoint_interval + 1
    vector_spec = pl.BlockSpec(
        (time, 1, heads, head_size),
        lambda batch_id: (0, batch_id, 0, 0),
    )
    state_spec = pl.BlockSpec(
        (1, heads, head_size, head_size),
        lambda batch_id: (batch_id, 0, 0, 0),
    )
    checkpoint_spec = pl.BlockSpec(
        (checkpoint_count, 1, heads, head_size, head_size),
        lambda batch_id: (0, batch_id, 0, 0, 0),
    )
    out_shapes = (
        jax.ShapeDtypeStruct(r.shape, r.dtype),
        jax.ShapeDtypeStruct(initial_state.shape, jnp.float32),
        jax.ShapeDtypeStruct(r.shape, jnp.float32),
        jax.ShapeDtypeStruct(
            (checkpoint_count, batch, heads, head_size, head_size),
            jnp.float32,
        ),
    )
    kernel = partial(
        _wkv_tpu_forward_kernel,
        time=time,
        checkpoint_interval=config.checkpoint_interval,
        output_dtype=r.dtype,
    )
    y, final_state, sa, checkpoints = pl.pallas_call(
        kernel,
        out_shape=out_shapes,
        grid=(batch,),
        in_specs=(
            vector_spec,
            vector_spec,
            vector_spec,
            vector_spec,
            vector_spec,
            vector_spec,
            state_spec,
        ),
        out_specs=(
            vector_spec,
            state_spec,
            vector_spec,
            checkpoint_spec,
        ),
        interpret=interpret,
        name="rwkv7_wkv_tpu_forward",
    )(r, w, k, v, neg_kk, kka, initial_state)
    return (y, final_state), (sa, checkpoints)


def wkv7_pallas_tpu_forward(
    r,
    w,
    k,
    v,
    neg_kk,
    kka,
    initial_state,
    *,
    config: WKVTPUConfig = WKVTPUConfig(),
    interpret: bool = False,
):
    """Run the persistent TPU WKV forward kernel."""

    _validate_config(config)
    time, batch, heads, head_size = r.shape
    vector_spec = pl.BlockSpec(
        (time, 1, heads, head_size),
        lambda batch_id: (0, batch_id, 0, 0),
    )
    state_spec = pl.BlockSpec(
        (1, heads, head_size, head_size),
        lambda batch_id: (batch_id, 0, 0, 0),
    )
    kernel = partial(
        _wkv_tpu_inference_kernel,
        time=time,
        output_dtype=r.dtype,
    )
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(r.shape, r.dtype),
            jax.ShapeDtypeStruct(initial_state.shape, jnp.float32),
        ),
        grid=(batch,),
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
        interpret=interpret,
        name="rwkv7_wkv_tpu_inference",
    )(r, w, k, v, neg_kk, kka, initial_state)


def wkv7_pallas_tpu_backward(
    r,
    w,
    k,
    v,
    neg_kk,
    kka,
    y_cotangent,
    final_state_cotangent,
    sa,
    checkpoints,
    *,
    config: WKVTPUConfig = WKVTPUConfig(),
    interpret: bool = False,
):
    """Run the TPU-specialized persistent reverse recurrence."""

    _validate_config(config)
    if config.checkpoint_interval != 1:
        raise ValueError(
            "WKV backward requires checkpoint_interval=1; inverse state "
            "reconstruction is undefined when an FP32 decay rounds to zero"
        )
    time, batch, heads, head_size = r.shape
    checkpoint_count = checkpoints.shape[0]
    expected_checkpoint_count = (
        time + config.checkpoint_interval - 1
    ) // config.checkpoint_interval + 1
    if checkpoint_count != expected_checkpoint_count:
        raise ValueError(
            "checkpoint count does not match time and checkpoint interval"
        )
    vector_spec = pl.BlockSpec(
        (time, 1, heads, head_size),
        lambda batch_id: (0, batch_id, 0, 0),
    )
    state_spec = pl.BlockSpec(
        (1, heads, head_size, head_size),
        lambda batch_id: (batch_id, 0, 0, 0),
    )
    checkpoint_spec = pl.BlockSpec(
        (checkpoint_count, 1, heads, head_size, head_size),
        lambda batch_id: (0, batch_id, 0, 0, 0),
    )
    vector_shape = jax.ShapeDtypeStruct(r.shape, r.dtype)
    out_shapes = (
        vector_shape,
        vector_shape,
        vector_shape,
        vector_shape,
        vector_shape,
        vector_shape,
        jax.ShapeDtypeStruct(final_state_cotangent.shape, jnp.float32),
    )
    kernel = partial(
        _wkv_tpu_backward_kernel,
        time=time,
        checkpoint_interval=config.checkpoint_interval,
        output_dtype=r.dtype,
    )
    return pl.pallas_call(
        kernel,
        out_shape=out_shapes,
        grid=(batch,),
        in_specs=(
            vector_spec,
            vector_spec,
            vector_spec,
            vector_spec,
            vector_spec,
            vector_spec,
            vector_spec,
            state_spec,
            vector_spec,
            checkpoint_spec,
        ),
        out_specs=(
            vector_spec,
            vector_spec,
            vector_spec,
            vector_spec,
            vector_spec,
            vector_spec,
            state_spec,
        ),
        interpret=interpret,
        name="rwkv7_wkv_tpu_backward",
    )(
        r,
        w,
        k,
        v,
        neg_kk,
        kka,
        y_cotangent,
        final_state_cotangent,
        sa,
        checkpoints,
    )


__all__ = [
    "WKVTPUConfig",
    "wkv7_pallas_tpu_backward",
    "wkv7_pallas_tpu_forward",
    "wkv7_pallas_tpu_forward_with_aux",
]
