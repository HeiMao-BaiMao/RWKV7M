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

from rwkv7m.model.screening import competitive_write_routing


GPULowering = Literal["mosaic", "triton"]
_STEP_STAT_COUNT = 22


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


def _load_time_scalar(ref, index):
    block = ref[pl.dslice(index, 1), :]
    return block[0, 0]


def _load_time_matrix(ref, index):
    block = ref[pl.dslice(index, 1), :, :, :]
    return block[0, 0, :, :]


def _load_state_matrix(ref):
    return ref[:][0, :, :]


def _load_state_vector(ref):
    return ref[:][0, :]


def _store_time_vector(ref, index, value):
    ref[pl.dslice(index, 1), :, :] = value.astype(ref.dtype)[None, None, :]


def _store_time_scalar(ref, index, value):
    ref[pl.dslice(index, 1), :] = jnp.reshape(value, (1, 1)).astype(ref.dtype)


def _store_time_matrix(ref, index, value):
    ref[pl.dslice(index, 1), :, :, :] = value.astype(ref.dtype)[
        None, None, :, :
    ]


def _store_time_scalars(ref, index, values):
    for value_index, value in enumerate(values):
        ref[
            pl.dslice(index, 1),
            :,
            pl.dslice(value_index, 1),
        ] = jnp.reshape(value, (1, 1, 1)).astype(ref.dtype)


def _store_state_matrix(ref, value):
    ref[:] = value.astype(ref.dtype)[None, :, :]


def _store_state_vector(ref, value):
    ref[:] = value.astype(ref.dtype)[None, :]


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
    admission,
    bank_logits,
    delta_slots,
    delta_read_keys,
    delta_values,
    delta_write_keys,
    mu,
    tau_read,
    tau_write,
    bank_ids,
    *,
    write_enabled: bool,
    write_mode: str | None,
    route_power: float,
    novelty_threshold: float,
    allocation_temperature: float,
    bank_route_temperature: float,
    allocation_age_weight: float,
    allocation_usage_weight: float,
    hard_admission: bool,
    admission_threshold: float,
    n_read_tiles: int,
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
    n_slots, key_size = read_keys.shape
    value_size = values.shape[-1]
    key_tile_size = key_size // n_read_tiles
    value_tile_size = value_size // n_read_tiles
    read_keys_tiled = read_keys.reshape(
        n_slots, n_read_tiles, key_tile_size
    )
    values_tiled = values.reshape(
        n_slots, n_read_tiles, value_tile_size
    )
    q_read_tiled = q_read.reshape(n_read_tiles, key_tile_size)
    normalized_read_keys = _unit_norm(read_keys_tiled, eps)
    normalized_values = values_tiled
    if use_value_unit_norm:
        normalized_values = _unit_norm(values_tiled, eps)
    read_similarity = jnp.sum(
        normalized_read_keys * q_read_tiled[None, :, :], axis=-1
    ).T
    tau_read_tiled = tau_read.reshape(n_read_tiles, 1)
    if use_leaky_warmup:
        hard = _trim_square(read_similarity, tau_read_tiled, eps)
        soft = jax.nn.sigmoid(
            leaky_gamma * (read_similarity - tau_read_tiled)
        )
        read_relevance = (1.0 - leaky_alpha) * hard + leaky_alpha * soft
    else:
        read_relevance = _trim_square(
            read_similarity, tau_read_tiled, eps
        )
    if use_age_mask:
        age_scores = (age_ref - ages) / (age_sigma + eps)
        read_relevance *= jax.nn.sigmoid(age_scores)[None, :]

    z_tiled = jnp.sum(
        read_relevance.T[:, :, None] * normalized_values,
        axis=0,
    )
    u_tiled = _tanh_norm(z_tiled, tanh_norm_cap, eps)
    z = z_tiled.reshape(value_size)
    u = u_tiled.reshape(value_size)
    read_activity = jnp.max(read_relevance, axis=0)

    resolved_mode = write_mode
    if resolved_mode is None:
        resolved_mode = (
            "legacy_threshold" if write_enabled else "legacy_unconditional"
        )
    matched_route = jnp.zeros_like(read_activity)
    novel_route = jnp.zeros_like(read_activity)
    effective_admission = jnp.asarray(0.0, dtype=jnp.float32)
    is_novel = jnp.asarray(False)
    if resolved_mode in ("legacy_threshold", "competitive_novel"):
        normalized_write_keys = _unit_norm(write_keys, eps)
        write_similarity = jnp.sum(
            normalized_write_keys * q_write[None, :], axis=-1
        )
        write_relevance = _trim_square(write_similarity, tau_write, eps)
    else:
        write_relevance = jnp.zeros_like(read_activity)

    if resolved_mode == "legacy_threshold":
        effective_write_relevance = jnp.maximum(
            write_relevance, write_rel_floor
        )
        strength = mu * effective_write_relevance
        next_ages = jnp.where(
            write_relevance > 1e-3, 0.0, ages + 1.0
        )
        matched_route = effective_write_relevance
    elif resolved_mode == "competitive_novel":
        (
            effective_write_relevance,
            raw_matched_route,
            novel_route,
            _,
            is_novel,
            effective_admission,
            _,
        ) = competitive_write_routing(
            write_relevance,
            ages,
            usage,
            admission,
            bank_logits,
            bank_ids,
            route_power=route_power,
            novelty_threshold=novelty_threshold,
            allocation_temperature=allocation_temperature,
            bank_route_temperature=bank_route_temperature,
            allocation_age_weight=allocation_age_weight,
            allocation_usage_weight=allocation_usage_weight,
            hard_admission=hard_admission,
            admission_threshold=admission_threshold,
            eps=eps,
        )
        matched_route = jnp.where(is_novel, 0.0, raw_matched_route)
        strength = mu * effective_write_relevance
        next_ages = jnp.where(
            effective_write_relevance > eps, 0.0, ages + 1.0
        )
    elif resolved_mode == "disabled":
        effective_write_relevance = jnp.zeros_like(read_activity)
        strength = jnp.zeros_like(read_activity)
        next_ages = ages + 1.0
    else:
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
    if resolved_mode == "legacy_threshold":
        activity = jnp.maximum(read_activity, write_relevance)
    else:
        activity = read_activity
    next_usage = usage_ema_decay * usage + (
        1.0 - usage_ema_decay
    ) * activity
    route_mass = jnp.sum(effective_write_relevance)
    route_distribution = effective_write_relevance / (route_mass + eps)
    route_entropy = -jnp.sum(
        route_distribution * jnp.log(route_distribution + eps)
    )
    route_top1 = jnp.max(route_distribution)
    resolved_bank_ids = jnp.asarray(
        bank_ids, dtype=jnp.int32
    )
    bank_write_mass = tuple(
        jnp.sum(
            effective_write_relevance * (resolved_bank_ids == bank_id)
        )
        for bank_id in range(3)
    )
    novel_mass = jnp.sum(novel_route)
    eviction_age = jnp.sum(novel_route * ages) / (novel_mass + eps)
    eviction_usage = jnp.sum(novel_route * usage) / (novel_mass + eps)
    is_competitive = resolved_mode == "competitive_novel"
    statistics = (
        jnp.mean(read_relevance),
        jnp.max(read_relevance),
        jnp.sum(read_activity > 1e-3).astype(jnp.float32),
        jnp.sqrt(jnp.sum(z * z)),
        jnp.sqrt(jnp.sum(u * u)),
        jnp.mean(write_relevance),
        jnp.mean(effective_write_relevance),
        jnp.mean(next_usage),
        jnp.sum(matched_route),
        novel_mass,
        route_entropy,
        route_top1,
        effective_admission,
        is_novel.astype(jnp.float32),
        (route_mass <= eps).astype(jnp.float32),
        *bank_write_mass,
        eviction_age,
        eviction_usage,
        ((effective_admission < 0.05) & is_competitive).astype(jnp.float32),
        ((effective_admission > 0.95) & is_competitive).astype(jnp.float32),
    )
    next_carry = (
        next_slots,
        next_read_keys,
        next_values,
        next_write_keys,
        next_ages,
        next_usage,
    )
    return u, next_carry, statistics, update_squared, strength


def _screening_gpu_forward_kernel(
    q_read_ref,
    q_write_ref,
    admission_ref,
    bank_logits_ref,
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
    bank_ids_ref,
    u_ref,
    final_slots_ref,
    final_ages_ref,
    final_usage_ref,
    statistics_ref,
    update_squared_ref,
    *,
    time: int,
    output_dtype,
    **step_config,
):
    carry = (
        _load_state_matrix(initial_slots_ref).astype(jnp.float32),
        _load_state_matrix(initial_read_keys_ref).astype(jnp.float32),
        _load_state_matrix(initial_values_ref).astype(jnp.float32),
        _load_state_matrix(initial_write_keys_ref).astype(jnp.float32),
        _load_state_vector(initial_ages_ref).astype(jnp.float32),
        _load_state_vector(initial_usage_ref).astype(jnp.float32),
    )
    mu = mu_ref[:].astype(jnp.float32)
    tau_read = tau_read_ref[:].astype(jnp.float32)
    tau_write = tau_write_ref[:][0].astype(jnp.float32)
    bank_ids = bank_ids_ref[:].astype(jnp.int32)

    @pl.loop(0, time, init_carry=carry)
    def time_loop(t, carry_t):
        u, next_carry, statistics, update_squared, _ = _screening_step(
            carry_t,
            _load_time_vector(q_read_ref, t).astype(jnp.float32),
            _load_time_vector(q_write_ref, t).astype(jnp.float32),
            _load_time_scalar(admission_ref, t).astype(jnp.float32),
            _load_time_vector(bank_logits_ref, t).astype(jnp.float32),
            _load_time_matrix(delta_slots_ref, t).astype(jnp.float32),
            _load_time_matrix(delta_read_keys_ref, t).astype(jnp.float32),
            _load_time_matrix(delta_values_ref, t).astype(jnp.float32),
            _load_time_matrix(delta_write_keys_ref, t).astype(jnp.float32),
            mu,
            tau_read,
            tau_write,
            bank_ids,
            **step_config,
        )
        _store_time_vector(u_ref, t, u.astype(output_dtype))
        _store_time_scalars(statistics_ref, t, statistics)
        _store_time_vector(update_squared_ref, t, update_squared)
        return next_carry

    final_carry = time_loop
    _store_state_matrix(final_slots_ref, final_carry[0])
    _store_state_vector(final_ages_ref, final_carry[4])
    _store_state_vector(final_usage_ref, final_carry[5])


def _screening_gpu_training_forward_kernel(
    q_read_ref,
    q_write_ref,
    admission_ref,
    bank_logits_ref,
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
    bank_ids_ref,
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
    tau_read = tau_read_ref[:].astype(jnp.float32)
    tau_write = tau_write_ref[:][0].astype(jnp.float32)
    bank_ids = bank_ids_ref[:].astype(jnp.int32)

    @pl.loop(0, time, init_carry=carry)
    def time_loop(t, carry_t):
        _store_time_matrix(tape_slots_ref, t, carry_t[0])
        _store_time_matrix(tape_read_keys_ref, t, carry_t[1])
        _store_time_matrix(tape_values_ref, t, carry_t[2])
        _store_time_matrix(tape_write_keys_ref, t, carry_t[3])
        _store_time_vector(tape_ages_ref, t, carry_t[4])
        _store_time_vector(tape_usage_ref, t, carry_t[5])
        u, next_carry, statistics, update_squared, _ = _screening_step(
            carry_t,
            _load_time_vector(q_read_ref, t).astype(jnp.float32),
            _load_time_vector(q_write_ref, t).astype(jnp.float32),
            _load_time_scalar(admission_ref, t).astype(jnp.float32),
            _load_time_vector(bank_logits_ref, t).astype(jnp.float32),
            _load_time_matrix(delta_slots_ref, t).astype(jnp.float32),
            _load_time_matrix(delta_read_keys_ref, t).astype(jnp.float32),
            _load_time_matrix(delta_values_ref, t).astype(jnp.float32),
            _load_time_matrix(delta_write_keys_ref, t).astype(jnp.float32),
            mu,
            tau_read,
            tau_write,
            bank_ids,
            **step_config,
        )
        _store_time_vector(u_ref, t, u.astype(output_dtype))
        _store_time_scalars(statistics_ref, t, statistics)
        _store_time_vector(update_squared_ref, t, update_squared)
        return next_carry

    final_carry = time_loop
    _store_state_matrix(final_slots_ref, final_carry[0])
    _store_state_vector(final_ages_ref, final_carry[4])
    _store_state_vector(final_usage_ref, final_carry[5])


def _screening_gpu_checkpoint_forward_kernel(
    q_read_ref,
    q_write_ref,
    admission_ref,
    bank_logits_ref,
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
    bank_ids_ref,
    u_ref,
    final_slots_ref,
    final_ages_ref,
    final_usage_ref,
    statistics_ref,
    update_squared_ref,
    checkpoint_slots_ref,
    checkpoint_read_keys_ref,
    checkpoint_values_ref,
    checkpoint_write_keys_ref,
    tape_ages_ref,
    tape_usage_ref,
    tape_strength_ref,
    *,
    time: int,
    checkpoint_interval: int,
    output_dtype,
    **step_config,
):
    carry = (
        _load_state_matrix(initial_slots_ref).astype(jnp.float32),
        _load_state_matrix(initial_read_keys_ref).astype(jnp.float32),
        _load_state_matrix(initial_values_ref).astype(jnp.float32),
        _load_state_matrix(initial_write_keys_ref).astype(jnp.float32),
        _load_state_vector(initial_ages_ref).astype(jnp.float32),
        _load_state_vector(initial_usage_ref).astype(jnp.float32),
    )
    for ref, value in zip(
        (
            checkpoint_slots_ref,
            checkpoint_read_keys_ref,
            checkpoint_values_ref,
            checkpoint_write_keys_ref,
        ),
        carry[:4],
        strict=True,
    ):
        _store_time_matrix(ref, 0, value)
    mu = mu_ref[:].astype(jnp.float32)
    tau_read = tau_read_ref[:].astype(jnp.float32)
    tau_write = tau_write_ref[:][0].astype(jnp.float32)
    bank_ids = bank_ids_ref[:].astype(jnp.int32)

    @pl.loop(0, time, init_carry=carry)
    def time_loop(t, carry_t):
        _store_time_vector(tape_ages_ref, t, carry_t[4])
        _store_time_vector(tape_usage_ref, t, carry_t[5])
        u, next_carry, statistics, update_squared, strength = _screening_step(
            carry_t,
            _load_time_vector(q_read_ref, t).astype(jnp.float32),
            _load_time_vector(q_write_ref, t).astype(jnp.float32),
            _load_time_scalar(admission_ref, t).astype(jnp.float32),
            _load_time_vector(bank_logits_ref, t).astype(jnp.float32),
            _load_time_matrix(delta_slots_ref, t).astype(jnp.float32),
            _load_time_matrix(delta_read_keys_ref, t).astype(jnp.float32),
            _load_time_matrix(delta_values_ref, t).astype(jnp.float32),
            _load_time_matrix(delta_write_keys_ref, t).astype(jnp.float32),
            mu,
            tau_read,
            tau_write,
            bank_ids,
            **step_config,
        )
        _store_time_vector(u_ref, t, u.astype(output_dtype))
        _store_time_scalars(statistics_ref, t, statistics)
        _store_time_vector(update_squared_ref, t, update_squared)
        _store_time_vector(tape_strength_ref, t, strength)
        checkpoint = (t + checkpoint_interval) // checkpoint_interval
        save_checkpoint = (
            (t + 1) == checkpoint * checkpoint_interval
        ) | ((t + 1) == time)

        @pl.when(save_checkpoint)
        def store_checkpoint():
            _store_time_matrix(checkpoint_slots_ref, checkpoint, next_carry[0])
            _store_time_matrix(
                checkpoint_read_keys_ref, checkpoint, next_carry[1]
            )
            _store_time_matrix(checkpoint_values_ref, checkpoint, next_carry[2])
            _store_time_matrix(
                checkpoint_write_keys_ref, checkpoint, next_carry[3]
            )

        return next_carry

    final_carry = time_loop
    _store_state_matrix(final_slots_ref, final_carry[0])
    _store_state_vector(final_ages_ref, final_carry[4])
    _store_state_vector(final_usage_ref, final_carry[5])


def _screening_gpu_checkpoint_backward_kernel(
    q_read_ref,
    q_write_ref,
    admission_ref,
    bank_logits_ref,
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
    bank_ids_ref,
    cotangent_u_ref,
    cotangent_final_slots_ref,
    cotangent_final_ages_ref,
    cotangent_final_usage_ref,
    checkpoint_slots_ref,
    checkpoint_read_keys_ref,
    checkpoint_values_ref,
    checkpoint_write_keys_ref,
    tape_ages_ref,
    tape_usage_ref,
    tape_strength_ref,
    grad_q_read_ref,
    grad_q_write_ref,
    grad_admission_ref,
    grad_bank_logits_ref,
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
    *,
    time: int,
    checkpoint_interval: int,
    **step_config,
):
    checkpoint_count = (
        time + checkpoint_interval - 1
    ) // checkpoint_interval + 1
    current_content = (
        _load_time_matrix(checkpoint_slots_ref, checkpoint_count - 1),
        _load_time_matrix(checkpoint_read_keys_ref, checkpoint_count - 1),
        _load_time_matrix(checkpoint_values_ref, checkpoint_count - 1),
        _load_time_matrix(checkpoint_write_keys_ref, checkpoint_count - 1),
    )
    mu = mu_ref[:].astype(jnp.float32)
    tau_read = tau_read_ref[:].astype(jnp.float32)
    tau_write = tau_write_ref[:][0].astype(jnp.float32)
    bank_ids = bank_ids_ref[:].astype(jnp.int32)
    cotangent_carry = (
        _load_state_matrix(cotangent_final_slots_ref).astype(jnp.float32),
        jnp.zeros_like(_load_state_matrix(initial_read_keys_ref), dtype=jnp.float32),
        jnp.zeros_like(_load_state_matrix(initial_values_ref), dtype=jnp.float32),
        jnp.zeros_like(_load_state_matrix(initial_write_keys_ref), dtype=jnp.float32),
        _load_state_vector(cotangent_final_ages_ref).astype(jnp.float32),
        _load_state_vector(cotangent_final_usage_ref).astype(jnp.float32),
    )
    reverse_carry = (
        *current_content,
        *cotangent_carry,
        jnp.zeros_like(mu),
        jnp.zeros_like(tau_read),
        jnp.asarray(0.0, dtype=jnp.float32),
    )

    @pl.loop(0, time, init_carry=reverse_carry)
    def reverse_loop(reverse_index, reverse_state):
        t = time - reverse_index - 1
        checkpoint = (t + checkpoint_interval) // checkpoint_interval
        load_checkpoint = (
            (t == time - 1)
            | ((t + 1) == checkpoint * checkpoint_interval)
        )
        saved_content = (
            _load_time_matrix(checkpoint_slots_ref, checkpoint),
            _load_time_matrix(checkpoint_read_keys_ref, checkpoint),
            _load_time_matrix(checkpoint_values_ref, checkpoint),
            _load_time_matrix(checkpoint_write_keys_ref, checkpoint),
        )
        current = tuple(
            jnp.where(load_checkpoint, saved, value)
            for saved, value in zip(
                saved_content, reverse_state[:4], strict=True
            )
        )
        strength = _load_time_vector(tape_strength_ref, t).astype(jnp.float32)
        deltas_t = (
            _load_time_matrix(delta_slots_ref, t).astype(jnp.float32),
            _load_time_matrix(delta_read_keys_ref, t).astype(jnp.float32),
            _load_time_matrix(delta_values_ref, t).astype(jnp.float32),
            _load_time_matrix(delta_write_keys_ref, t).astype(jnp.float32),
        )
        denominator = 1.0 - strength[:, None]
        previous_content = tuple(
            (value - strength[:, None] * delta) / denominator
            for value, delta in zip(current, deltas_t, strict=True)
        )
        carry_t = (
            *previous_content,
            _load_time_vector(tape_ages_ref, t).astype(jnp.float32),
            _load_time_vector(tape_usage_ref, t).astype(jnp.float32),
        )
        q_read_t = _load_time_vector(q_read_ref, t).astype(jnp.float32)
        q_write_t = _load_time_vector(q_write_ref, t).astype(jnp.float32)
        admission_t = _load_time_scalar(admission_ref, t).astype(jnp.float32)
        bank_logits_t = _load_time_vector(bank_logits_ref, t).astype(jnp.float32)

        def differentiable_step(
            step_carry,
            step_q_read,
            step_q_write,
            step_admission,
            step_bank_logits,
            step_delta_slots,
            step_delta_read_keys,
            step_delta_values,
            step_delta_write_keys,
            step_mu,
            step_tau_read,
            step_tau_write,
        ):
            u, next_carry, _, _, _ = _screening_step(
                step_carry,
                step_q_read,
                step_q_write,
                step_admission,
                step_bank_logits,
                step_delta_slots,
                step_delta_read_keys,
                step_delta_values,
                step_delta_write_keys,
                step_mu,
                step_tau_read,
                step_tau_write,
                bank_ids,
                **step_config,
            )
            return u, next_carry

        _, pullback = jax.vjp(
            differentiable_step,
            carry_t,
            q_read_t,
            q_write_t,
            admission_t,
            bank_logits_t,
            *deltas_t,
            mu,
            tau_read,
            tau_write,
        )
        gradients = pullback(
            (
                _load_time_vector(cotangent_u_ref, t).astype(jnp.float32),
                reverse_state[4:10],
            )
        )
        _store_time_vector(grad_q_read_ref, t, gradients[1])
        _store_time_vector(grad_q_write_ref, t, gradients[2])
        _store_time_scalar(grad_admission_ref, t, gradients[3])
        _store_time_vector(grad_bank_logits_ref, t, gradients[4])
        _store_time_matrix(grad_delta_slots_ref, t, gradients[5])
        _store_time_matrix(grad_delta_read_keys_ref, t, gradients[6])
        _store_time_matrix(grad_delta_values_ref, t, gradients[7])
        _store_time_matrix(grad_delta_write_keys_ref, t, gradients[8])
        return (
            *previous_content,
            *gradients[0],
            reverse_state[10] + gradients[9],
            reverse_state[11] + gradients[10],
            reverse_state[12] + gradients[11],
        )

    final = reverse_loop
    _store_state_matrix(grad_initial_slots_ref, final[4])
    _store_state_matrix(grad_initial_read_keys_ref, final[5])
    _store_state_matrix(grad_initial_values_ref, final[6])
    _store_state_matrix(grad_initial_write_keys_ref, final[7])
    _store_state_vector(grad_initial_ages_ref, final[8])
    _store_state_vector(grad_initial_usage_ref, final[9])
    grad_mu_ref[:] = final[10][None, :]
    grad_tau_read_ref[:] = final[11][None, :]
    grad_tau_write_ref[:] = jnp.reshape(final[12], (1, 1))


def _screening_gpu_backward_kernel(
    q_read_ref,
    q_write_ref,
    admission_ref,
    bank_logits_ref,
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
    bank_ids_ref,
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
    grad_admission_ref,
    grad_bank_logits_ref,
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
    tau_read = tau_read_ref[:].astype(jnp.float32)
    tau_write = tau_write_ref[:][0].astype(jnp.float32)
    bank_ids = bank_ids_ref[:].astype(jnp.int32)
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
        jnp.zeros_like(tau_read),
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
        admission_t = _load_time_scalar(admission_ref, t).astype(jnp.float32)
        bank_logits_t = _load_time_vector(bank_logits_ref, t).astype(jnp.float32)
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
            step_admission,
            step_bank_logits,
            step_delta_slots,
            step_delta_read_keys,
            step_delta_values,
            step_delta_write_keys,
            step_mu,
            step_tau_read,
            step_tau_write,
        ):
            u, next_carry, _, _, _ = _screening_step(
                step_carry,
                step_q_read,
                step_q_write,
                step_admission,
                step_bank_logits,
                step_delta_slots,
                step_delta_read_keys,
                step_delta_values,
                step_delta_write_keys,
                step_mu,
                step_tau_read,
                step_tau_write,
                bank_ids,
                **step_config,
            )
            return u, next_carry

        _, pullback = jax.vjp(
            differentiable_step,
            carry_t,
            q_read_t,
            q_write_t,
            admission_t,
            bank_logits_t,
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
        _store_time_scalar(grad_admission_ref, t, step_gradients[3])
        _store_time_vector(grad_bank_logits_ref, t, step_gradients[4])
        _store_time_matrix(grad_delta_slots_ref, t, step_gradients[5])
        _store_time_matrix(
            grad_delta_read_keys_ref, t, step_gradients[6]
        )
        _store_time_matrix(grad_delta_values_ref, t, step_gradients[7])
        _store_time_matrix(
            grad_delta_write_keys_ref, t, step_gradients[8]
        )
        return (
            *previous_carry,
            gradient_carry[6] + step_gradients[9],
            gradient_carry[7] + step_gradients[10],
            gradient_carry[8] + step_gradients[11],
        )

    final_gradients = reverse_loop
    _store_state_matrix(grad_initial_slots_ref, final_gradients[0])
    _store_state_matrix(grad_initial_read_keys_ref, final_gradients[1])
    _store_state_matrix(grad_initial_values_ref, final_gradients[2])
    _store_state_matrix(grad_initial_write_keys_ref, final_gradients[3])
    _store_state_vector(grad_initial_ages_ref, final_gradients[4])
    _store_state_vector(grad_initial_usage_ref, final_gradients[5])
    grad_mu_ref[:] = final_gradients[6][None, :]
    grad_tau_read_ref[:] = final_gradients[7][None, :]
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
    admission,
    bank_logits,
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
            specs["q"], specs["q"], specs["time_scalar"],
            specs["bank_logits"],
            specs["time_matrix"](slot_size),
            specs["time_matrix"](key_size),
            specs["time_matrix"](value_size),
            specs["time_matrix"](key_size),
            specs["state_matrix"](slot_size),
            specs["state_matrix"](key_size),
            specs["state_matrix"](value_size),
            specs["state_matrix"](key_size),
            specs["state_vector"], specs["state_vector"],
            specs["shared_vector"],
            specs["shared_feature"](config.n_read_tiles),
            specs["shared_scalar"],
            specs["shared_vector"],
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
        q_read, q_write, admission, bank_logits,
        delta_slots, delta_read_keys, delta_values,
        delta_write_keys, initial_slots, initial_read_keys,
        initial_values, initial_write_keys, initial_ages, initial_usage,
        mu,
        jnp.reshape(tau_read, (config.n_read_tiles,)),
        jnp.reshape(tau_write, (1,)),
        jnp.asarray(config.bank_ids or (0,) * n_slots, dtype=jnp.int32),
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
    time_scalar = pl.BlockSpec(
        (time, 1), lambda batch_id: (0, batch_id)
    )
    bank_logits = pl.BlockSpec(
        (time, 1, 3), lambda batch_id: (0, batch_id, 0)
    )

    def time_matrix(feature_size):
        return pl.BlockSpec(
            (time, 1, n_slots, feature_size),
            lambda batch_id: (0, batch_id, 0, 0),
        )

    def sequence_matrix(length, feature_size):
        return pl.BlockSpec(
            (length, 1, n_slots, feature_size),
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
        "time_scalar": time_scalar,
        "bank_logits": bank_logits,
        "time_matrix": time_matrix,
        "sequence_matrix": sequence_matrix,
        "state_matrix": state_matrix,
        "state_vector": state_vector,
        "shared_vector": shared_vector,
        "shared_scalar": shared_scalar,
        "shared_feature": lambda size: pl.BlockSpec(
            (size,), lambda batch_id: (0,)
        ),
        "u": u,
        "statistics": statistics,
        "update": update,
        "per_batch_vector": per_batch_vector,
        "per_batch_scalar": per_batch_scalar,
        "per_batch_feature": lambda size: pl.BlockSpec(
            (1, size), lambda batch_id: (batch_id, 0)
        ),
    }


def _screening_step_config(config, time: int):
    return {
        "time": time,
        "write_enabled": config.write_enabled,
        "write_mode": config.write_mode,
        "route_power": config.route_power,
        "novelty_threshold": config.novelty_threshold,
        "allocation_temperature": config.allocation_temperature,
        "bank_route_temperature": config.bank_route_temperature,
        "allocation_age_weight": config.allocation_age_weight,
        "allocation_usage_weight": config.allocation_usage_weight,
        "hard_admission": config.hard_admission,
        "admission_threshold": config.admission_threshold,
        "n_read_tiles": config.n_read_tiles,
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


def _screening_pallas_gpu_forward_with_checkpoints(
    q_read,
    q_write,
    admission,
    bank_logits,
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
    lowering,
    kernel_config,
    interpret,
):
    time, batch, key_size = q_read.shape
    _, n_slots, slot_size = initial_slots.shape
    value_size = initial_values.shape[-1]
    checkpoint_interval = config.checkpoint_interval
    checkpoint_count = (
        time + checkpoint_interval - 1
    ) // checkpoint_interval + 1
    specs = _screening_gpu_specs(
        time, key_size, n_slots, slot_size, value_size
    )
    result = pl.pallas_call(
        partial(
            _screening_gpu_checkpoint_forward_kernel,
            checkpoint_interval=checkpoint_interval,
            output_dtype=delta_values.dtype,
            **_screening_step_config(config, time),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((time, batch, value_size), delta_values.dtype),
            jax.ShapeDtypeStruct(initial_slots.shape, jnp.float32),
            jax.ShapeDtypeStruct(initial_ages.shape, jnp.float32),
            jax.ShapeDtypeStruct(initial_usage.shape, jnp.float32),
            jax.ShapeDtypeStruct(
                (time, batch, _STEP_STAT_COUNT), jnp.float32
            ),
            jax.ShapeDtypeStruct((time, batch, n_slots), jnp.float32),
            jax.ShapeDtypeStruct(
                (checkpoint_count, batch, n_slots, slot_size), jnp.float32
            ),
            jax.ShapeDtypeStruct(
                (checkpoint_count, batch, n_slots, key_size), jnp.float32
            ),
            jax.ShapeDtypeStruct(
                (checkpoint_count, batch, n_slots, value_size), jnp.float32
            ),
            jax.ShapeDtypeStruct(
                (checkpoint_count, batch, n_slots, key_size), jnp.float32
            ),
            jax.ShapeDtypeStruct((time, batch, n_slots), jnp.float32),
            jax.ShapeDtypeStruct((time, batch, n_slots), jnp.float32),
            jax.ShapeDtypeStruct((time, batch, n_slots), jnp.float32),
        ),
        grid=(batch,),
        in_specs=(
            specs["q"], specs["q"], specs["time_scalar"],
            specs["bank_logits"],
            specs["time_matrix"](slot_size),
            specs["time_matrix"](key_size),
            specs["time_matrix"](value_size),
            specs["time_matrix"](key_size),
            specs["state_matrix"](slot_size),
            specs["state_matrix"](key_size),
            specs["state_matrix"](value_size),
            specs["state_matrix"](key_size),
            specs["state_vector"], specs["state_vector"],
            specs["shared_vector"],
            specs["shared_feature"](config.n_read_tiles),
            specs["shared_scalar"],
            specs["shared_vector"],
        ),
        out_specs=(
            specs["u"], specs["state_matrix"](slot_size),
            specs["state_vector"], specs["state_vector"],
            specs["statistics"], specs["update"],
            specs["sequence_matrix"](checkpoint_count, slot_size),
            specs["sequence_matrix"](checkpoint_count, key_size),
            specs["sequence_matrix"](checkpoint_count, value_size),
            specs["sequence_matrix"](checkpoint_count, key_size),
            specs["update"], specs["update"], specs["update"],
        ),
        interpret=interpret,
        compiler_params=_compiler_params(
            lowering, kernel_config, interpret=interpret
        ),
        name=f"rwkv7_screening_gpu_{lowering}_checkpoint_forward",
    )(
        q_read,
        q_write,
        admission,
        bank_logits,
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
        jnp.reshape(tau_read, (config.n_read_tiles,)),
        jnp.reshape(tau_write, (1,)),
        jnp.asarray(config.bank_ids or (0,) * n_slots, dtype=jnp.int32),
    )
    return result[:6], result[6:]


def screening_pallas_gpu_forward_with_aux(
    q_read,
    q_write,
    admission,
    bank_logits,
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
    if config.checkpoint_interval is not None:
        return _screening_pallas_gpu_forward_with_checkpoints(
            q_read,
            q_write,
            admission,
            bank_logits,
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
            config=config,
            lowering=lowering,
            kernel_config=kernel_config,
            interpret=interpret,
        )
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
            specs["time_scalar"],
            specs["bank_logits"],
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
            specs["shared_feature"](config.n_read_tiles),
            specs["shared_scalar"],
            specs["shared_vector"],
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
        admission,
        bank_logits,
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
        jnp.reshape(tau_read, (config.n_read_tiles,)),
        jnp.reshape(tau_write, (1,)),
        jnp.asarray(config.bank_ids or (0,) * n_slots, dtype=jnp.int32),
    )
    return result[:6], result[6:]


def _screening_pallas_gpu_backward_with_checkpoints(
    q_read,
    q_write,
    admission,
    bank_logits,
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
    checkpoint_slots,
    checkpoint_read_keys,
    checkpoint_values,
    checkpoint_write_keys,
    tape_ages,
    tape_usage,
    tape_strength,
    *,
    config,
    lowering,
    kernel_config,
    interpret,
):
    time, batch, key_size = q_read.shape
    _, n_slots, slot_size = initial_slots.shape
    value_size = initial_values.shape[-1]
    checkpoint_count = checkpoint_slots.shape[0]
    specs = _screening_gpu_specs(
        time, key_size, n_slots, slot_size, value_size
    )
    arrays = (
        q_read,
        q_write,
        admission,
        bank_logits,
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
        jnp.reshape(tau_read, (config.n_read_tiles,)),
        jnp.reshape(tau_write, (1,)),
        jnp.asarray(config.bank_ids or (0,) * n_slots, dtype=jnp.int32),
        cotangent_u,
        cotangent_final_slots,
        cotangent_final_ages,
        cotangent_final_usage,
        checkpoint_slots,
        checkpoint_read_keys,
        checkpoint_values,
        checkpoint_write_keys,
        tape_ages,
        tape_usage,
        tape_strength,
    )
    result = pl.pallas_call(
        partial(
            _screening_gpu_checkpoint_backward_kernel,
            checkpoint_interval=config.checkpoint_interval,
            **_screening_step_config(config, time),
        ),
        out_shape=(
            *(jax.ShapeDtypeStruct(value.shape, value.dtype) for value in arrays[:14]),
            jax.ShapeDtypeStruct((batch, n_slots), jnp.float32),
            jax.ShapeDtypeStruct(
                (batch, config.n_read_tiles), jnp.float32
            ),
            jax.ShapeDtypeStruct((batch, 1), jnp.float32),
        ),
        grid=(batch,),
        in_specs=(
            specs["q"], specs["q"], specs["time_scalar"],
            specs["bank_logits"],
            specs["time_matrix"](slot_size),
            specs["time_matrix"](key_size),
            specs["time_matrix"](value_size),
            specs["time_matrix"](key_size),
            specs["state_matrix"](slot_size),
            specs["state_matrix"](key_size),
            specs["state_matrix"](value_size),
            specs["state_matrix"](key_size),
            specs["state_vector"], specs["state_vector"],
            specs["shared_vector"],
            specs["shared_feature"](config.n_read_tiles),
            specs["shared_scalar"], specs["shared_vector"], specs["u"],
            specs["state_matrix"](slot_size),
            specs["state_vector"], specs["state_vector"],
            specs["sequence_matrix"](checkpoint_count, slot_size),
            specs["sequence_matrix"](checkpoint_count, key_size),
            specs["sequence_matrix"](checkpoint_count, value_size),
            specs["sequence_matrix"](checkpoint_count, key_size),
            specs["update"], specs["update"], specs["update"],
        ),
        out_specs=(
            specs["q"], specs["q"], specs["time_scalar"],
            specs["bank_logits"],
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
            specs["per_batch_feature"](config.n_read_tiles),
            specs["per_batch_scalar"],
        ),
        interpret=interpret,
        compiler_params=_compiler_params(
            lowering, kernel_config, interpret=interpret
        ),
        name=f"rwkv7_screening_gpu_{lowering}_checkpoint_backward",
    )(*arrays)
    return (
        *result[:14],
        jnp.sum(result[14], axis=0).astype(mu.dtype),
        jnp.sum(result[15], axis=0).reshape(tau_read.shape).astype(tau_read.dtype),
        jnp.sum(result[16], axis=0).reshape(tau_write.shape).astype(tau_write.dtype),
    )


def screening_pallas_gpu_backward(
    q_read,
    q_write,
    admission,
    bank_logits,
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
    *aux,
    config,
    lowering: GPULowering,
    kernel_config: ScreeningGPUConfig = ScreeningGPUConfig(),
    interpret: bool = False,
):
    """Run the dedicated reverse-time projected-screening GPU kernel."""

    _validate_config(kernel_config)
    if config.checkpoint_interval is not None:
        return _screening_pallas_gpu_backward_with_checkpoints(
            q_read,
            q_write,
            admission,
            bank_logits,
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
            *aux,
            config=config,
            lowering=lowering,
            kernel_config=kernel_config,
            interpret=interpret,
        )
    if len(aux) != 6:
        raise ValueError("full-tape screening backward expects six auxiliaries")
    (
        tape_slots,
        tape_read_keys,
        tape_values,
        tape_write_keys,
        tape_ages,
        tape_usage,
    ) = aux
    time, batch, key_size = q_read.shape
    _, n_slots, slot_size = initial_slots.shape
    value_size = initial_values.shape[-1]
    specs = _screening_gpu_specs(
        time, key_size, n_slots, slot_size, value_size
    )
    arrays = (
        q_read,
        q_write,
        admission,
        bank_logits,
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
        jnp.reshape(tau_read, (config.n_read_tiles,)),
        jnp.reshape(tau_write, (1,)),
        jnp.asarray(config.bank_ids or (0,) * n_slots, dtype=jnp.int32),
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
            *(jax.ShapeDtypeStruct(value.shape, value.dtype) for value in arrays[:14]),
            jax.ShapeDtypeStruct((batch, n_slots), jnp.float32),
            jax.ShapeDtypeStruct(
                (batch, config.n_read_tiles), jnp.float32
            ),
            jax.ShapeDtypeStruct((batch, 1), jnp.float32),
        ),
        grid=(batch,),
        in_specs=(
            specs["q"], specs["q"], specs["time_scalar"],
            specs["bank_logits"],
            specs["time_matrix"](slot_size),
            specs["time_matrix"](key_size),
            specs["time_matrix"](value_size),
            specs["time_matrix"](key_size),
            specs["state_matrix"](slot_size),
            specs["state_matrix"](key_size),
            specs["state_matrix"](value_size),
            specs["state_matrix"](key_size),
            specs["state_vector"], specs["state_vector"],
            specs["shared_vector"],
            specs["shared_feature"](config.n_read_tiles),
            specs["shared_scalar"], specs["shared_vector"], specs["u"],
            specs["state_matrix"](slot_size),
            specs["state_vector"], specs["state_vector"],
            specs["time_matrix"](slot_size),
            specs["time_matrix"](key_size),
            specs["time_matrix"](value_size),
            specs["time_matrix"](key_size),
            specs["update"], specs["update"],
        ),
        out_specs=(
            specs["q"], specs["q"], specs["time_scalar"],
            specs["bank_logits"],
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
            specs["per_batch_feature"](config.n_read_tiles),
            specs["per_batch_scalar"],
        ),
        interpret=interpret,
        compiler_params=_compiler_params(
            lowering, kernel_config, interpret=interpret
        ),
        name=f"rwkv7_screening_gpu_{lowering}_backward",
    )(*arrays)
    return (
        *result[:14],
        jnp.sum(result[14], axis=0).astype(mu.dtype),
        jnp.sum(result[15], axis=0).reshape(tau_read.shape).astype(tau_read.dtype),
        jnp.sum(result[16], axis=0).reshape(tau_write.shape).astype(tau_write.dtype),
    )

__all__ = [
    "GPULowering",
    "ScreeningGPUConfig",
    "screening_pallas_gpu_backward",
    "screening_pallas_gpu_forward",
    "screening_pallas_gpu_forward_with_aux",
]
