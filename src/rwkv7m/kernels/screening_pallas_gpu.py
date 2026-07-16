"""Persistent projected-screening recurrence for NVIDIA GPUs.

One Pallas program owns one batch item's slot state. Dense projections are
performed before and after this kernel, leaving only the sequential state
transition and slot reductions in the persistent time loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


GPULowering = Literal["mosaic", "triton"]
_STEP_STAT_COUNT = 8


@dataclass(frozen=True)
class ScreeningGPUConfig:
    num_warps: int = 4
    num_stages: int = 2


def _validate_config(config: ScreeningGPUConfig) -> None:
    if config.num_warps not in (1, 2, 4, 8, 16, 32):
        raise ValueError("num_warps must be a supported power of two")
    if config.num_stages <= 0:
        raise ValueError("num_stages must be positive")


def _load_time_vector(ref, index):
    block = ref[pl.dslice(index, 1), :, :]
    return block[0, 0, :]


def _load_time_matrix(ref, index):
    block = ref[pl.dslice(index, 1), :, :, :]
    return block[0, 0, :, :]


def _load_state_matrix(ref):
    return ref[:][0, :, :]


def _load_state_vector(ref):
    return ref[:][0, :]


def _store_time_vector(ref, index, value):
    ref[pl.dslice(index, 1), :, :] = value[None, None, :]


def _store_time_matrix(ref, index, value):
    ref[pl.dslice(index, 1), :, :, :] = value[None, None, :, :]


def _store_state_matrix(ref, value):
    ref[:] = value[None, :, :]


def _store_state_vector(ref, value):
    ref[:] = value[None, :]


def _unit_norm(value, eps):
    squared_sum = jnp.sum(value * value, axis=-1, keepdims=True)
    return value / jnp.sqrt(squared_sum + eps * eps)


def _trim_square(similarity, tau, eps):
    scaled = (similarity - tau) / (1.0 - tau + eps)
    return jnp.square(jax.nn.relu(scaled))


def _tanh_norm(value, cap, eps):
    squared_sum = jnp.sum(value * value, axis=-1, keepdims=True)
    norm = jnp.sqrt(squared_sum + eps * eps)
    return value * (cap * jnp.tanh(norm / cap) / norm)


def _screening_step(
    carry,
    q_read,
    q_write,
    delta_slots,
    delta_read_keys,
    delta_values,
    delta_write_keys,
    mu,
    tau_read,
    tau_write,
    *,
    write_enabled: bool,
    use_value_unit_norm: bool,
    use_leaky_warmup: bool,
    leaky_alpha: float,
    leaky_gamma: float,
    use_age_mask: bool,
    age_ref: float,
    age_sigma: float,
    write_rel_floor: float,
    usage_ema_decay: float,
    tanh_norm_cap: float,
    eps: float,
):
    slots, read_keys, values, write_keys, ages, usage = carry
    normalized_read_keys = _unit_norm(read_keys, eps)
    normalized_values = (
        _unit_norm(values, eps) if use_value_unit_norm else values
    )
    read_similarity = jnp.sum(
        normalized_read_keys * q_read[None, :], axis=-1
    )
    if use_leaky_warmup:
        hard = _trim_square(read_similarity, tau_read, eps)
        soft = jax.nn.sigmoid(leaky_gamma * (read_similarity - tau_read))
        read_relevance = (1.0 - leaky_alpha) * hard + leaky_alpha * soft
    else:
        read_relevance = _trim_square(read_similarity, tau_read, eps)
    if use_age_mask:
        age_scores = (age_ref - ages) / (age_sigma + eps)
        read_relevance *= jax.nn.sigmoid(age_scores)

    z = jnp.sum(read_relevance[:, None] * normalized_values, axis=0)
    u = _tanh_norm(z, tanh_norm_cap, eps)

    if write_enabled:
        normalized_write_keys = _unit_norm(write_keys, eps)
        write_similarity = jnp.sum(
            normalized_write_keys * q_write[None, :], axis=-1
        )
        write_relevance = _trim_square(write_similarity, tau_write, eps)
        effective_write_relevance = jnp.maximum(
            write_relevance, write_rel_floor
        )
        strength = mu * effective_write_relevance
        next_ages = jnp.where(
            write_relevance > 1e-3, 0.0, ages + 1.0
        )
    else:
        write_relevance = jnp.zeros_like(read_relevance)
        effective_write_relevance = write_relevance
        strength = mu
        next_ages = ages

    strength_vector = strength[:, None]
    next_slots = slots + strength_vector * (delta_slots - slots)
    next_read_keys = read_keys + strength_vector * (
        delta_read_keys - read_keys
    )
    next_values = values + strength_vector * (delta_values - values)
    next_write_keys = write_keys + strength_vector * (
        delta_write_keys - write_keys
    )
    update = next_slots - slots
    update_squared = jnp.sum(update * update, axis=-1)
    activity = jnp.maximum(
        read_relevance,
        write_relevance if write_enabled else read_relevance,
    )
    next_usage = usage_ema_decay * usage + (
        1.0 - usage_ema_decay
    ) * activity
    statistics = jnp.stack(
        (
            jnp.mean(read_relevance),
            jnp.max(read_relevance),
            jnp.sum(read_relevance > 1e-3).astype(jnp.float32),
            jnp.sqrt(jnp.sum(z * z)),
            jnp.sqrt(jnp.sum(u * u)),
            jnp.mean(write_relevance),
            jnp.mean(effective_write_relevance),
            jnp.mean(next_usage),
        )
    )
    next_carry = (
        next_slots,
        next_read_keys,
        next_values,
        next_write_keys,
        next_ages,
        next_usage,
    )
    return u, next_carry, statistics, update_squared


def _screening_gpu_forward_kernel(
    q_read_ref,
    q_write_ref,
    delta_slots_ref,
    delta_read_keys_ref,
    delta_values_ref,
    delta_write_keys_ref,
    initial_slots_ref,
    initial_read_keys_ref,
    initial_values_ref,
    initial_write_keys_ref,
    initial_ages_ref,
    initial_usage_ref,
    mu_ref,
    tau_read_ref,
    tau_write_ref,
    u_ref,
    final_slots_ref,
    final_ages_ref,
    final_usage_ref,
    statistics_ref,
    update_squared_ref,
    *,
    time: int,
    write_enabled: bool,
    use_value_unit_norm: bool,
    use_leaky_warmup: bool,
    leaky_alpha: float,
    leaky_gamma: float,
    use_age_mask: bool,
    age_ref: float,
    age_sigma: float,
    write_rel_floor: float,
    usage_ema_decay: float,
    tanh_norm_cap: float,
    eps: float,
    output_dtype,
):
    slots = _load_state_matrix(initial_slots_ref).astype(jnp.float32)
    read_keys = _load_state_matrix(initial_read_keys_ref).astype(jnp.float32)
    values = _load_state_matrix(initial_values_ref).astype(jnp.float32)
    write_keys = _load_state_matrix(initial_write_keys_ref).astype(jnp.float32)
    ages = _load_state_vector(initial_ages_ref).astype(jnp.float32)
    usage = _load_state_vector(initial_usage_ref).astype(jnp.float32)
    mu = mu_ref[:].astype(jnp.float32)
    tau_read = tau_read_ref[:][0].astype(jnp.float32)
    tau_write = tau_write_ref[:][0].astype(jnp.float32)

    initial_carry = (slots, read_keys, values, write_keys, ages, usage)

    @pl.loop(0, time, init_carry=initial_carry)
    def time_loop(t, carry):
        slots_t, read_keys_t, values_t, write_keys_t, ages_t, usage_t = carry
        q_read_t = _load_time_vector(q_read_ref, t).astype(jnp.float32)
        q_write_t = _load_time_vector(q_write_ref, t).astype(jnp.float32)

        normalized_read_keys = _unit_norm(read_keys_t, eps)
        normalized_values = values_t
        if use_value_unit_norm:
            normalized_values = _unit_norm(values_t, eps)
        read_similarity = jnp.sum(
            normalized_read_keys * q_read_t[None, :], axis=-1
        )
        if use_leaky_warmup:
            hard = _trim_square(read_similarity, tau_read, eps)
            soft = jax.nn.sigmoid(
                leaky_gamma * (read_similarity - tau_read)
            )
            read_relevance = (
                (1.0 - leaky_alpha) * hard + leaky_alpha * soft
            )
        else:
            read_relevance = _trim_square(
                read_similarity, tau_read, eps
            )
        if use_age_mask:
            age_scores = (age_ref - ages_t) / (age_sigma + eps)
            read_relevance *= jax.nn.sigmoid(age_scores)

        z = jnp.sum(
            read_relevance[:, None] * normalized_values, axis=0
        )
        u = _tanh_norm(z, tanh_norm_cap, eps)

        if write_enabled:
            normalized_write_keys = _unit_norm(write_keys_t, eps)
            write_similarity = jnp.sum(
                normalized_write_keys * q_write_t[None, :], axis=-1
            )
            write_relevance = _trim_square(
                write_similarity, tau_write, eps
            )
            effective_write_relevance = jnp.maximum(
                write_relevance, write_rel_floor
            )
            strength = mu * effective_write_relevance
            next_ages = jnp.where(
                write_relevance > 1e-3, 0.0, ages_t + 1.0
            )
        else:
            write_relevance = jnp.zeros_like(read_relevance)
            effective_write_relevance = write_relevance
            strength = mu
            next_ages = ages_t

        strength_vector = strength[:, None]
        next_slots = slots_t + strength_vector * (
            _load_time_matrix(delta_slots_ref, t).astype(jnp.float32)
            - slots_t
        )
        next_read_keys = read_keys_t + strength_vector * (
            _load_time_matrix(delta_read_keys_ref, t).astype(jnp.float32)
            - read_keys_t
        )
        next_values = values_t + strength_vector * (
            _load_time_matrix(delta_values_ref, t).astype(jnp.float32)
            - values_t
        )
        next_write_keys = write_keys_t + strength_vector * (
            _load_time_matrix(delta_write_keys_ref, t).astype(jnp.float32)
            - write_keys_t
        )
        update = next_slots - slots_t
        update_squared = jnp.sum(update * update, axis=-1)

        activity = jnp.maximum(
            read_relevance,
            write_relevance if write_enabled else read_relevance,
        )
        next_usage = usage_ema_decay * usage_t + (
            1.0 - usage_ema_decay
        ) * activity
        statistics = jnp.stack(
            (
                jnp.mean(read_relevance),
                jnp.max(read_relevance),
                jnp.sum(read_relevance > 1e-3).astype(jnp.float32),
                jnp.sqrt(jnp.sum(z * z)),
                jnp.sqrt(jnp.sum(u * u)),
                jnp.mean(write_relevance),
                jnp.mean(effective_write_relevance),
                jnp.mean(next_usage),
            )
        )
        _store_time_vector(u_ref, t, u.astype(output_dtype))
        _store_time_vector(statistics_ref, t, statistics)
        _store_time_vector(update_squared_ref, t, update_squared)
        return (
            next_slots,
            next_read_keys,
            next_values,
            next_write_keys,
            next_ages,
            next_usage,
        )

    final_carry = time_loop
    _store_state_matrix(final_slots_ref, final_carry[0])
    _store_state_vector(final_ages_ref, final_carry[4])
    _store_state_vector(final_usage_ref, final_carry[5])


def _screening_gpu_training_forward_kernel(
    q_read_ref,
    q_write_ref,
    delta_slots_ref,
    delta_read_keys_ref,
    delta_values_ref,
    delta_write_keys_ref,
    initial_slots_ref,
    initial_read_keys_ref,
    initial_values_ref,
    initial_write_keys_ref,
    initial_ages_ref,
    initial_usage_ref,
    mu_ref,
    tau_read_ref,
    tau_write_ref,
    u_ref,
    final_slots_ref,
    final_ages_ref,
    final_usage_ref,
    statistics_ref,
    update_squared_ref,
    tape_slots_ref,
    tape_read_keys_ref,
    tape_values_ref,
    tape_write_keys_ref,
    tape_ages_ref,
    tape_usage_ref,
    **step_config,
):
    time = step_config.pop("time")
    output_dtype = step_config.pop("output_dtype")
    carry = (
        _load_state_matrix(initial_slots_ref).astype(jnp.float32),
        _load_state_matrix(initial_read_keys_ref).astype(jnp.float32),
        _load_state_matrix(initial_values_ref).astype(jnp.float32),
        _load_state_matrix(initial_write_keys_ref).astype(jnp.float32),
        _load_state_vector(initial_ages_ref).astype(jnp.float32),
        _load_state_vector(initial_usage_ref).astype(jnp.float32),
    )
    mu = mu_ref[:].astype(jnp.float32)
    tau_read = tau_read_ref[:][0].astype(jnp.float32)
    tau_write = tau_write_ref[:][0].astype(jnp.float32)

    @pl.loop(0, time, init_carry=carry)
    def time_loop(t, carry_t):
        _store_time_matrix(tape_slots_ref, t, carry_t[0])
        _store_time_matrix(tape_read_keys_ref, t, carry_t[1])
        _store_time_matrix(tape_values_ref, t, carry_t[2])
        _store_time_matrix(tape_write_keys_ref, t, carry_t[3])
        _store_time_vector(tape_ages_ref, t, carry_t[4])
        _store_time_vector(tape_usage_ref, t, carry_t[5])
        u, next_carry, statistics, update_squared = _screening_step(
            carry_t,
            _load_time_vector(q_read_ref, t).astype(jnp.float32),
            _load_time_vector(q_write_ref, t).astype(jnp.float32),
            _load_time_matrix(delta_slots_ref, t).astype(jnp.float32),
            _load_time_matrix(delta_read_keys_ref, t).astype(jnp.float32),
            _load_time_matrix(delta_values_ref, t).astype(jnp.float32),
            _load_time_matrix(delta_write_keys_ref, t).astype(jnp.float32),
            mu,
            tau_read,
            tau_write,
            **step_config,
        )
        _store_time_vector(u_ref, t, u.astype(output_dtype))
        _store_time_vector(statistics_ref, t, statistics)
        _store_time_vector(update_squared_ref, t, update_squared)
        return next_carry

    final_carry = time_loop
    _store_state_matrix(final_slots_ref, final_carry[0])
    _store_state_vector(final_ages_ref, final_carry[4])
    _store_state_vector(final_usage_ref, final_carry[5])


def _screening_gpu_backward_kernel(
    q_read_ref,
    q_write_ref,
    delta_slots_ref,
    delta_read_keys_ref,
    delta_values_ref,
    delta_write_keys_ref,
    initial_slots_ref,
    initial_read_keys_ref,
    initial_values_ref,
    initial_write_keys_ref,
    initial_ages_ref,
    initial_usage_ref,
    mu_ref,
    tau_read_ref,
    tau_write_ref,
    cotangent_u_ref,
    cotangent_final_slots_ref,
    cotangent_final_ages_ref,
    cotangent_final_usage_ref,
    tape_slots_ref,
    tape_read_keys_ref,
    tape_values_ref,
    tape_write_keys_ref,
    tape_ages_ref,
    tape_usage_ref,
    grad_q_read_ref,
    grad_q_write_ref,
    grad_delta_slots_ref,
    grad_delta_read_keys_ref,
    grad_delta_values_ref,
    grad_delta_write_keys_ref,
    grad_initial_slots_ref,
    grad_initial_read_keys_ref,
    grad_initial_values_ref,
    grad_initial_write_keys_ref,
    grad_initial_ages_ref,
    grad_initial_usage_ref,
    grad_mu_ref,
    grad_tau_read_ref,
    grad_tau_write_ref,
    **step_config,
):
    time = step_config.pop("time")
    mu = mu_ref[:].astype(jnp.float32)
    tau_read = tau_read_ref[:][0].astype(jnp.float32)
    tau_write = tau_write_ref[:][0].astype(jnp.float32)
    cotangent_carry = (
        _load_state_matrix(cotangent_final_slots_ref).astype(jnp.float32),
        jnp.zeros_like(_load_state_matrix(initial_read_keys_ref), dtype=jnp.float32),
        jnp.zeros_like(_load_state_matrix(initial_values_ref), dtype=jnp.float32),
        jnp.zeros_like(_load_state_matrix(initial_write_keys_ref), dtype=jnp.float32),
        _load_state_vector(cotangent_final_ages_ref).astype(jnp.float32),
        _load_state_vector(cotangent_final_usage_ref).astype(jnp.float32),
    )
    reverse_carry = (
        *cotangent_carry,
        jnp.zeros_like(mu),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )

    @pl.loop(0, time, init_carry=reverse_carry)
    def reverse_loop(reverse_index, gradient_carry):
        t = time - reverse_index - 1
        carry_t = (
            _load_time_matrix(tape_slots_ref, t).astype(jnp.float32),
            _load_time_matrix(tape_read_keys_ref, t).astype(jnp.float32),
            _load_time_matrix(tape_values_ref, t).astype(jnp.float32),
            _load_time_matrix(tape_write_keys_ref, t).astype(jnp.float32),
            _load_time_vector(tape_ages_ref, t).astype(jnp.float32),
            _load_time_vector(tape_usage_ref, t).astype(jnp.float32),
        )
        q_read_t = _load_time_vector(q_read_ref, t).astype(jnp.float32)
        q_write_t = _load_time_vector(q_write_ref, t).astype(jnp.float32)
        deltas_t = (
            _load_time_matrix(delta_slots_ref, t).astype(jnp.float32),
            _load_time_matrix(delta_read_keys_ref, t).astype(jnp.float32),
            _load_time_matrix(delta_values_ref, t).astype(jnp.float32),
            _load_time_matrix(delta_write_keys_ref, t).astype(jnp.float32),
        )

        def differentiable_step(
            step_carry,
            step_q_read,
            step_q_write,
            step_delta_slots,
            step_delta_read_keys,
            step_delta_values,
            step_delta_write_keys,
            step_mu,
            step_tau_read,
            step_tau_write,
        ):
            u, next_carry, _, _ = _screening_step(
                step_carry,
                step_q_read,
                step_q_write,
                step_delta_slots,
                step_delta_read_keys,
                step_delta_values,
                step_delta_write_keys,
                step_mu,
                step_tau_read,
                step_tau_write,
                **step_config,
            )
            return u, next_carry

        _, pullback = jax.vjp(
            differentiable_step,
            carry_t,
            q_read_t,
            q_write_t,
            *deltas_t,
            mu,
            tau_read,
            tau_write,
        )
        step_gradients = pullback(
            (
                _load_time_vector(cotangent_u_ref, t).astype(jnp.float32),
                gradient_carry[:6],
            )
        )
        previous_carry = step_gradients[0]
        _store_time_vector(grad_q_read_ref, t, step_gradients[1])
        _store_time_vector(grad_q_write_ref, t, step_gradients[2])
        _store_time_matrix(grad_delta_slots_ref, t, step_gradients[3])
        _store_time_matrix(
            grad_delta_read_keys_ref, t, step_gradients[4]
        )
        _store_time_matrix(grad_delta_values_ref, t, step_gradients[5])
        _store_time_matrix(
            grad_delta_write_keys_ref, t, step_gradients[6]
        )
        return (
            *previous_carry,
            gradient_carry[6] + step_gradients[7],
            gradient_carry[7] + step_gradients[8],
            gradient_carry[8] + step_gradients[9],
        )

    final_gradients = reverse_loop
    _store_state_matrix(grad_initial_slots_ref, final_gradients[0])
    _store_state_matrix(grad_initial_read_keys_ref, final_gradients[1])
    _store_state_matrix(grad_initial_values_ref, final_gradients[2])
    _store_state_matrix(grad_initial_write_keys_ref, final_gradients[3])
    _store_state_vector(grad_initial_ages_ref, final_gradients[4])
    _store_state_vector(grad_initial_usage_ref, final_gradients[5])
    grad_mu_ref[:] = final_gradients[6][None, :]
    grad_tau_read_ref[:] = jnp.reshape(final_gradients[7], (1, 1))
    grad_tau_write_ref[:] = jnp.reshape(final_gradients[8], (1, 1))


def _compiler_params(
    lowering: GPULowering,
    config: ScreeningGPUConfig,
    *,
    interpret: bool,
):
    if lowering == "mosaic" and not interpret:
        from jax.experimental.pallas import mosaic_gpu as plgpu

        return plgpu.CompilerParams(
            dimension_semantics=("parallel",),
            reduction_scratch_bytes=6144,
        )
    if lowering == "triton" and not interpret:
        from jax.experimental.pallas import triton as pltriton

        return pltriton.CompilerParams(
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
    return None


def screening_pallas_gpu_forward(
    q_read,
    q_write,
    delta_slots,
    delta_read_keys,
    delta_values,
    delta_write_keys,
    initial_slots,
    initial_read_keys,
    initial_values,
    initial_write_keys,
    initial_ages,
    initial_usage,
    mu,
    tau_read,
    tau_write,
    *,
    config,
    lowering: GPULowering,
    kernel_config: ScreeningGPUConfig = ScreeningGPUConfig(),
    interpret: bool = False,
):
    """Run the persistent projected-screening forward recurrence on GPU."""

    _validate_config(kernel_config)
    time, batch, key_size = q_read.shape
    _, n_slots, slot_size = initial_slots.shape
    value_size = initial_values.shape[-1]

    specs = _screening_gpu_specs(
        time, key_size, n_slots, slot_size, value_size
    )
    kernel = partial(
        _screening_gpu_forward_kernel,
        output_dtype=delta_values.dtype,
        **_screening_step_config(config, time),
    )
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(
                (time, batch, value_size), delta_values.dtype
            ),
            jax.ShapeDtypeStruct(initial_slots.shape, jnp.float32),
            jax.ShapeDtypeStruct(initial_ages.shape, jnp.float32),
            jax.ShapeDtypeStruct(initial_usage.shape, jnp.float32),
            jax.ShapeDtypeStruct(
                (time, batch, _STEP_STAT_COUNT), jnp.float32
            ),
            jax.ShapeDtypeStruct((time, batch, n_slots), jnp.float32),
        ),
        grid=(batch,),
        in_specs=(
            specs["q"], specs["q"],
            specs["time_matrix"](slot_size),
            specs["time_matrix"](key_size),
            specs["time_matrix"](value_size),
            specs["time_matrix"](key_size),
            specs["state_matrix"](slot_size),
            specs["state_matrix"](key_size),
            specs["state_matrix"](value_size),
            specs["state_matrix"](key_size),
            specs["state_vector"], specs["state_vector"],
            specs["shared_vector"], specs["shared_scalar"],
            specs["shared_scalar"],
        ),
        out_specs=(
            specs["u"], specs["state_matrix"](slot_size),
            specs["state_vector"], specs["state_vector"],
            specs["statistics"], specs["update"],
        ),
        interpret=interpret,
        compiler_params=_compiler_params(
            lowering, kernel_config, interpret=interpret
        ),
        name=f"rwkv7_screening_gpu_{lowering}_forward",
    )(
        q_read, q_write, delta_slots, delta_read_keys, delta_values,
        delta_write_keys, initial_slots, initial_read_keys,
        initial_values, initial_write_keys, initial_ages, initial_usage,
        mu, jnp.reshape(tau_read, (1,)), jnp.reshape(tau_write, (1,)),
    )


def _screening_gpu_specs(
    time: int,
    key_size: int,
    n_slots: int,
    slot_size: int,
    value_size: int,
):
    q_spec = pl.BlockSpec(
        (time, 1, key_size), lambda batch_id: (0, batch_id, 0)
    )

    def time_matrix(feature_size):
        return pl.BlockSpec(
            (time, 1, n_slots, feature_size),
            lambda batch_id: (0, batch_id, 0, 0),
        )

    def state_matrix(feature_size):
        return pl.BlockSpec(
            (1, n_slots, feature_size),
            lambda batch_id: (batch_id, 0, 0),
        )

    state_vector = pl.BlockSpec(
        (1, n_slots), lambda batch_id: (batch_id, 0)
    )
    shared_vector = pl.BlockSpec((n_slots,), lambda batch_id: (0,))
    shared_scalar = pl.BlockSpec((1,), lambda batch_id: (0,))
    u = pl.BlockSpec(
        (time, 1, value_size), lambda batch_id: (0, batch_id, 0)
    )
    statistics = pl.BlockSpec(
        (time, 1, _STEP_STAT_COUNT), lambda batch_id: (0, batch_id, 0)
    )
    update = pl.BlockSpec(
        (time, 1, n_slots), lambda batch_id: (0, batch_id, 0)
    )
    per_batch_vector = pl.BlockSpec(
        (1, n_slots), lambda batch_id: (batch_id, 0)
    )
    per_batch_scalar = pl.BlockSpec(
        (1, 1), lambda batch_id: (batch_id, 0)
    )
    return {
        "q": q_spec,
        "time_matrix": time_matrix,
        "state_matrix": state_matrix,
        "state_vector": state_vector,
        "shared_vector": shared_vector,
        "shared_scalar": shared_scalar,
        "u": u,
        "statistics": statistics,
        "update": update,
        "per_batch_vector": per_batch_vector,
        "per_batch_scalar": per_batch_scalar,
    }


def _screening_step_config(config, time: int):
    return {
        "time": time,
        "write_enabled": config.write_enabled,
        "use_value_unit_norm": config.use_value_unit_norm,
        "use_leaky_warmup": config.use_leaky_warmup,
        "leaky_alpha": config.leaky_alpha,
        "leaky_gamma": config.leaky_gamma,
        "use_age_mask": config.use_age_mask,
        "age_ref": config.age_ref,
        "age_sigma": config.age_sigma,
        "write_rel_floor": config.write_rel_floor,
        "usage_ema_decay": config.usage_ema_decay,
        "tanh_norm_cap": config.tanh_norm_cap,
        "eps": config.eps,
    }


def screening_pallas_gpu_forward_with_aux(
    q_read,
    q_write,
    delta_slots,
    delta_read_keys,
    delta_values,
    delta_write_keys,
    initial_slots,
    initial_read_keys,
    initial_values,
    initial_write_keys,
    initial_ages,
    initial_usage,
    mu,
    tau_read,
    tau_write,
    *,
    config,
    lowering: GPULowering,
    kernel_config: ScreeningGPUConfig = ScreeningGPUConfig(),
    interpret: bool = False,
):
    """Run training forward and return the FP32 carry tape for the VJP."""

    _validate_config(kernel_config)
    time, batch, key_size = q_read.shape
    _, n_slots, slot_size = initial_slots.shape
    value_size = initial_values.shape[-1]
    specs = _screening_gpu_specs(
        time, key_size, n_slots, slot_size, value_size
    )
    kernel = partial(
        _screening_gpu_training_forward_kernel,
        output_dtype=delta_values.dtype,
        **_screening_step_config(config, time),
    )
    result = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(
                (time, batch, value_size), delta_values.dtype
            ),
            jax.ShapeDtypeStruct(initial_slots.shape, jnp.float32),
            jax.ShapeDtypeStruct(initial_ages.shape, jnp.float32),
            jax.ShapeDtypeStruct(initial_usage.shape, jnp.float32),
            jax.ShapeDtypeStruct(
                (time, batch, _STEP_STAT_COUNT), jnp.float32
            ),
            jax.ShapeDtypeStruct((time, batch, n_slots), jnp.float32),
            jax.ShapeDtypeStruct(
                (time, batch, n_slots, slot_size), jnp.float32
            ),
            jax.ShapeDtypeStruct(
                (time, batch, n_slots, key_size), jnp.float32
            ),
            jax.ShapeDtypeStruct(
                (time, batch, n_slots, value_size), jnp.float32
            ),
            jax.ShapeDtypeStruct(
                (time, batch, n_slots, key_size), jnp.float32
            ),
            jax.ShapeDtypeStruct((time, batch, n_slots), jnp.float32),
            jax.ShapeDtypeStruct((time, batch, n_slots), jnp.float32),
        ),
        grid=(batch,),
        in_specs=(
            specs["q"],
            specs["q"],
            specs["time_matrix"](slot_size),
            specs["time_matrix"](key_size),
            specs["time_matrix"](value_size),
            specs["time_matrix"](key_size),
            specs["state_matrix"](slot_size),
            specs["state_matrix"](key_size),
            specs["state_matrix"](value_size),
            specs["state_matrix"](key_size),
            specs["state_vector"],
            specs["state_vector"],
            specs["shared_vector"],
            specs["shared_scalar"],
            specs["shared_scalar"],
        ),
        out_specs=(
            specs["u"],
            specs["state_matrix"](slot_size),
            specs["state_vector"],
            specs["state_vector"],
            specs["statistics"],
            specs["update"],
            specs["time_matrix"](slot_size),
            specs["time_matrix"](key_size),
            specs["time_matrix"](value_size),
            specs["time_matrix"](key_size),
            specs["update"],
            specs["update"],
        ),
        interpret=interpret,
        compiler_params=_compiler_params(
            lowering, kernel_config, interpret=interpret
        ),
        name=f"rwkv7_screening_gpu_{lowering}_training_forward",
    )(
        q_read,
        q_write,
        delta_slots,
        delta_read_keys,
        delta_values,
        delta_write_keys,
        initial_slots,
        initial_read_keys,
        initial_values,
        initial_write_keys,
        initial_ages,
        initial_usage,
        mu,
        jnp.reshape(tau_read, (1,)),
        jnp.reshape(tau_write, (1,)),
    )
    return result[:6], result[6:]


def screening_pallas_gpu_backward(
    q_read,
    q_write,
    delta_slots,
    delta_read_keys,
    delta_values,
    delta_write_keys,
    initial_slots,
    initial_read_keys,
    initial_values,
    initial_write_keys,
    initial_ages,
    initial_usage,
    mu,
    tau_read,
    tau_write,
    cotangent_u,
    cotangent_final_slots,
    cotangent_final_ages,
    cotangent_final_usage,
    tape_slots,
    tape_read_keys,
    tape_values,
    tape_write_keys,
    tape_ages,
    tape_usage,
    *,
    config,
    lowering: GPULowering,
    kernel_config: ScreeningGPUConfig = ScreeningGPUConfig(),
    interpret: bool = False,
):
    """Run the dedicated reverse-time projected-screening GPU kernel."""

    _validate_config(kernel_config)
    time, batch, key_size = q_read.shape
    _, n_slots, slot_size = initial_slots.shape
    value_size = initial_values.shape[-1]
    specs = _screening_gpu_specs(
        time, key_size, n_slots, slot_size, value_size
    )
    arrays = (
        q_read,
        q_write,
        delta_slots,
        delta_read_keys,
        delta_values,
        delta_write_keys,
        initial_slots,
        initial_read_keys,
        initial_values,
        initial_write_keys,
        initial_ages,
        initial_usage,
        mu,
        jnp.reshape(tau_read, (1,)),
        jnp.reshape(tau_write, (1,)),
        cotangent_u,
        cotangent_final_slots,
        cotangent_final_ages,
        cotangent_final_usage,
        tape_slots,
        tape_read_keys,
        tape_values,
        tape_write_keys,
        tape_ages,
        tape_usage,
    )
    kernel = partial(
        _screening_gpu_backward_kernel,
        **_screening_step_config(config, time),
    )
    result = pl.pallas_call(
        kernel,
        out_shape=(
            *(jax.ShapeDtypeStruct(value.shape, value.dtype) for value in arrays[:12]),
            jax.ShapeDtypeStruct((batch, n_slots), jnp.float32),
            jax.ShapeDtypeStruct((batch, 1), jnp.float32),
            jax.ShapeDtypeStruct((batch, 1), jnp.float32),
        ),
        grid=(batch,),
        in_specs=(
            specs["q"], specs["q"],
            specs["time_matrix"](slot_size),
            specs["time_matrix"](key_size),
            specs["time_matrix"](value_size),
            specs["time_matrix"](key_size),
            specs["state_matrix"](slot_size),
            specs["state_matrix"](key_size),
            specs["state_matrix"](value_size),
            specs["state_matrix"](key_size),
            specs["state_vector"], specs["state_vector"],
            specs["shared_vector"], specs["shared_scalar"],
            specs["shared_scalar"], specs["u"],
            specs["state_matrix"](slot_size),
            specs["state_vector"], specs["state_vector"],
            specs["time_matrix"](slot_size),
            specs["time_matrix"](key_size),
            specs["time_matrix"](value_size),
            specs["time_matrix"](key_size),
            specs["update"], specs["update"],
        ),
        out_specs=(
            specs["q"], specs["q"],
            specs["time_matrix"](slot_size),
            specs["time_matrix"](key_size),
            specs["time_matrix"](value_size),
            specs["time_matrix"](key_size),
            specs["state_matrix"](slot_size),
            specs["state_matrix"](key_size),
            specs["state_matrix"](value_size),
            specs["state_matrix"](key_size),
            specs["state_vector"], specs["state_vector"],
            specs["per_batch_vector"],
            specs["per_batch_scalar"],
            specs["per_batch_scalar"],
        ),
        interpret=interpret,
        compiler_params=_compiler_params(
            lowering, kernel_config, interpret=interpret
        ),
        name=f"rwkv7_screening_gpu_{lowering}_backward",
    )(*arrays)
    return (
        *result[:12],
        jnp.sum(result[12], axis=0).astype(mu.dtype),
        jnp.sum(result[13], axis=0).reshape(tau_read.shape).astype(tau_read.dtype),
        jnp.sum(result[14], axis=0).reshape(tau_write.shape).astype(tau_write.dtype),
    )

__all__ = [
    "GPULowering",
    "ScreeningGPUConfig",
    "screening_pallas_gpu_backward",
    "screening_pallas_gpu_forward",
    "screening_pallas_gpu_forward_with_aux",
]
