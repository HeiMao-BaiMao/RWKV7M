"""Persistent projected-screening recurrence specialized for TPU VMEM."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from rwkv7m.model.screening import competitive_write_routing


_STEP_STAT_COUNT = 22


def _load_time_vector(ref, index):
    block = ref[pl.dslice(index, 1), :, :, :]
    return block[0, 0, :, :]


def _load_time_matrix(ref, index):
    block = ref[pl.dslice(index, 1), :, :, :]
    return block[0, 0, :, :]


def _load_state_matrix(ref):
    return ref[:][0, :, :]


def _load_state_vector(ref):
    return ref[:][0, :, :]


def _store_time_vector(ref, index, value):
    ref[pl.dslice(index, 1), :, :, :] = value[None, None, :, :]


def _store_time_matrix(ref, index, value):
    ref[pl.dslice(index, 1), :, :, :] = value[None, None, :, :]


def _store_state_matrix(ref, value):
    ref[:] = value[None, :, :]


def _store_state_vector(ref, value):
    ref[:] = value[None, :, :]


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
    ages_1d = ages[:, 0]
    usage_1d = usage[:, 0]
    mu_1d = mu[:, 0]
    q_read_1d = q_read[0]
    q_write_1d = q_write[0]
    admission_scalar = admission[0, 0]
    bank_logits_1d = bank_logits[0]
    n_slots, key_size = read_keys.shape
    value_size = values.shape[-1]
    key_tile_size = key_size // n_read_tiles
    value_tile_size = value_size // n_read_tiles
    # Mosaic TPU cannot lower the otherwise natural
    # ``[1, feature] -> [tiles, feature / tiles]`` shape cast while preserving
    # its vector layout. Keep every tile rank-two and use static slices instead.
    age_gate = jnp.ones_like(ages_1d)
    if use_age_mask:
        age_scores = (age_ref - ages_1d) / (age_sigma + eps)
        age_gate = jax.nn.sigmoid(age_scores)
    read_activity = jnp.zeros_like(ages_1d)
    read_relevance_sum = jnp.asarray(0.0, dtype=jnp.float32)
    read_relevance_max = jnp.asarray(0.0, dtype=jnp.float32)
    z_tiles = []
    u_tiles = []
    for tile in range(n_read_tiles):
        key_start = tile * key_tile_size
        key_limit = key_start + key_tile_size
        value_start = tile * value_tile_size
        value_limit = value_start + value_tile_size
        read_keys_tile = jax.lax.slice_in_dim(
            read_keys, key_start, key_limit, axis=1
        )
        values_tile = jax.lax.slice_in_dim(
            values, value_start, value_limit, axis=1
        )
        q_read_tile = jax.lax.slice_in_dim(
            q_read_1d, key_start, key_limit, axis=0
        )
        normalized_read_keys = _unit_norm(read_keys_tile, eps)
        normalized_values = values_tile
        if use_value_unit_norm:
            normalized_values = _unit_norm(values_tile, eps)
        read_similarity = jnp.sum(
            normalized_read_keys * q_read_tile[None, :], axis=-1
        )
        tau_read_tile = tau_read[tile]
        if use_leaky_warmup:
            hard = _trim_square(read_similarity, tau_read_tile, eps)
            soft = jax.nn.sigmoid(
                leaky_gamma * (read_similarity - tau_read_tile)
            )
            read_relevance_tile = (
                (1.0 - leaky_alpha) * hard + leaky_alpha * soft
            )
        else:
            read_relevance_tile = _trim_square(
                read_similarity, tau_read_tile, eps
            )
        read_relevance_tile *= age_gate
        z_tile = jnp.sum(
            read_relevance_tile[:, None] * normalized_values, axis=0
        )
        u_tile = _tanh_norm(z_tile, tanh_norm_cap, eps)
        z_tiles.append(z_tile)
        u_tiles.append(u_tile)
        read_activity = jnp.maximum(read_activity, read_relevance_tile)
        # Reducing a short sliced vector can require an unsupported Mosaic
        # layout-offset change. Slot count is static, so aggregate these two
        # diagnostic scalars explicitly instead of introducing a reduction.
        for slot in range(n_slots):
            relevance_scalar = read_relevance_tile[slot]
            read_relevance_sum += relevance_scalar
            read_relevance_max = jnp.maximum(
                read_relevance_max, relevance_scalar
            )
    z = jnp.concatenate(tuple(z_tiles), axis=0)[None, :]
    u = jnp.concatenate(tuple(u_tiles), axis=0)[None, :]

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
            normalized_write_keys * q_write_1d[None, :], axis=-1
        )
        write_relevance = _trim_square(write_similarity, tau_write, eps)
    else:
        write_relevance = jnp.zeros_like(read_activity)

    if resolved_mode == "legacy_threshold":
        effective_write_relevance = jnp.maximum(
            write_relevance, write_rel_floor
        )
        strength = mu_1d * effective_write_relevance
        next_ages = jnp.where(
            write_relevance > 1e-3, 0.0, ages_1d + 1.0
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
            ages_1d,
            usage_1d,
            admission_scalar,
            bank_logits_1d,
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
        strength = mu_1d * effective_write_relevance
        next_ages = jnp.where(
            effective_write_relevance > eps, 0.0, ages_1d + 1.0
        )
    elif resolved_mode == "disabled":
        effective_write_relevance = jnp.zeros_like(read_activity)
        strength = jnp.zeros_like(read_activity)
        next_ages = ages_1d + 1.0
    else:
        effective_write_relevance = write_relevance
        strength = mu_1d
        next_ages = ages_1d

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
    update_squared = jnp.sum(update * update, axis=-1, keepdims=True)
    if resolved_mode == "legacy_threshold":
        activity = jnp.maximum(read_activity, write_relevance)
    else:
        activity = read_activity
    next_usage = usage_ema_decay * usage_1d + (
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
    eviction_age = jnp.sum(novel_route * ages_1d) / (novel_mass + eps)
    eviction_usage = jnp.sum(novel_route * usage_1d) / (novel_mass + eps)
    is_competitive = resolved_mode == "competitive_novel"
    statistic_scalars = (
        read_relevance_sum / (n_read_tiles * n_slots),
        read_relevance_max,
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
    statistics = jnp.concatenate(
        tuple(jnp.reshape(value, (1, 1)) for value in statistic_scalars),
        axis=-1,
    )
    return (
        u,
        (
            next_slots,
            next_read_keys,
            next_values,
            next_write_keys,
            next_ages[:, None],
            next_usage[:, None],
        ),
        statistics,
        update_squared,
        strength[:, None],
    )


def _screening_tpu_forward_kernel(
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
    slots = _load_state_matrix(initial_slots_ref).astype(jnp.float32)
    read_keys = _load_state_matrix(initial_read_keys_ref).astype(jnp.float32)
    values = _load_state_matrix(initial_values_ref).astype(jnp.float32)
    write_keys = _load_state_matrix(initial_write_keys_ref).astype(jnp.float32)
    ages = _load_state_vector(initial_ages_ref).astype(jnp.float32)
    usage = _load_state_vector(initial_usage_ref).astype(jnp.float32)
    mu = mu_ref[:][0, :, :].astype(jnp.float32)
    tau_read = tau_read_ref[:][0, :].astype(jnp.float32)
    tau_write = tau_write_ref[:][0, 0].astype(jnp.float32)
    bank_ids = bank_ids_ref[:][0, :, 0].astype(jnp.int32)

    initial_carry = (slots, read_keys, values, write_keys, ages, usage)

    @pl.loop(0, time, init_carry=initial_carry)
    def time_loop(t, carry):
        slots_t, read_keys_t, values_t, write_keys_t, ages_t, usage_t = carry
        q_read_t = _load_time_vector(q_read_ref, t).astype(jnp.float32)
        q_write_t = _load_time_vector(q_write_ref, t).astype(jnp.float32)

        u, next_carry, statistics, update_squared, _ = _screening_step(
            carry,
            q_read_t,
            q_write_t,
            _load_time_vector(admission_ref, t).astype(jnp.float32),
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
        _store_time_vector(statistics_ref, t, statistics)
        _store_time_vector(update_squared_ref, t, update_squared)
        return next_carry

    final_carry = time_loop
    _store_state_matrix(final_slots_ref, final_carry[0])
    _store_state_vector(final_ages_ref, final_carry[4])
    _store_state_vector(final_usage_ref, final_carry[5])


def _screening_tpu_training_forward_kernel(
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
    mu = mu_ref[:][0, :, :].astype(jnp.float32)
    tau_read = tau_read_ref[:][0, :].astype(jnp.float32)
    tau_write = tau_write_ref[:][0, 0].astype(jnp.float32)
    bank_ids = bank_ids_ref[:][0, :, 0].astype(jnp.int32)

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
            _load_time_vector(admission_ref, t).astype(jnp.float32),
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
        _store_time_vector(statistics_ref, t, statistics)
        _store_time_vector(update_squared_ref, t, update_squared)
        return next_carry

    final_carry = time_loop
    _store_state_matrix(final_slots_ref, final_carry[0])
    _store_state_vector(final_ages_ref, final_carry[4])
    _store_state_vector(final_usage_ref, final_carry[5])


def _screening_tpu_checkpoint_forward_kernel(
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
    mu = mu_ref[:][0, :, :].astype(jnp.float32)
    tau_read = tau_read_ref[:][0, :].astype(jnp.float32)
    tau_write = tau_write_ref[:][0, 0].astype(jnp.float32)
    bank_ids = bank_ids_ref[:][0, :, 0].astype(jnp.int32)

    @pl.loop(0, time, init_carry=carry)
    def time_loop(t, carry_t):
        _store_time_vector(tape_ages_ref, t, carry_t[4])
        _store_time_vector(tape_usage_ref, t, carry_t[5])
        u, next_carry, statistics, update_squared, strength = _screening_step(
            carry_t,
            _load_time_vector(q_read_ref, t).astype(jnp.float32),
            _load_time_vector(q_write_ref, t).astype(jnp.float32),
            _load_time_vector(admission_ref, t).astype(jnp.float32),
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
        _store_time_vector(statistics_ref, t, statistics)
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


def _screening_tpu_checkpoint_backward_kernel(
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
    gradient_dtypes,
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
    mu = mu_ref[:][0, :, :].astype(jnp.float32)
    tau_read = tau_read_ref[:][0, :].astype(jnp.float32)
    tau_write = tau_write_ref[:][0, 0].astype(jnp.float32)
    bank_ids = bank_ids_ref[:][0, :, 0].astype(jnp.int32)
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
        denominator = 1.0 - strength
        previous_content = tuple(
            (value - strength * delta) / denominator
            for value, delta in zip(current, deltas_t, strict=True)
        )
        carry_t = (
            *previous_content,
            _load_time_vector(tape_ages_ref, t).astype(jnp.float32),
            _load_time_vector(tape_usage_ref, t).astype(jnp.float32),
        )
        q_read_t = _load_time_vector(q_read_ref, t).astype(jnp.float32)
        q_write_t = _load_time_vector(q_write_ref, t).astype(jnp.float32)
        admission_t = _load_time_vector(admission_ref, t).astype(jnp.float32)
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
        _store_time_vector(
            grad_q_read_ref, t, gradients[1].astype(gradient_dtypes[0])
        )
        _store_time_vector(
            grad_q_write_ref, t, gradients[2].astype(gradient_dtypes[1])
        )
        _store_time_vector(
            grad_admission_ref, t, gradients[3].astype(gradient_dtypes[2])
        )
        _store_time_vector(
            grad_bank_logits_ref, t, gradients[4].astype(gradient_dtypes[3])
        )
        _store_time_matrix(
            grad_delta_slots_ref, t, gradients[5].astype(gradient_dtypes[4])
        )
        _store_time_matrix(
            grad_delta_read_keys_ref, t, gradients[6].astype(gradient_dtypes[5])
        )
        _store_time_matrix(
            grad_delta_values_ref, t, gradients[7].astype(gradient_dtypes[6])
        )
        _store_time_matrix(
            grad_delta_write_keys_ref,
            t,
            gradients[8].astype(gradient_dtypes[7]),
        )
        return (
            *previous_content,
            *gradients[0],
            reverse_state[10] + gradients[9],
            reverse_state[11] + gradients[10],
            reverse_state[12] + gradients[11],
        )

    final = reverse_loop
    _store_state_matrix(
        grad_initial_slots_ref, final[4].astype(gradient_dtypes[8])
    )
    _store_state_matrix(
        grad_initial_read_keys_ref, final[5].astype(gradient_dtypes[9])
    )
    _store_state_matrix(
        grad_initial_values_ref, final[6].astype(gradient_dtypes[10])
    )
    _store_state_matrix(
        grad_initial_write_keys_ref, final[7].astype(gradient_dtypes[11])
    )
    _store_state_vector(
        grad_initial_ages_ref, final[8].astype(gradient_dtypes[12])
    )
    _store_state_vector(
        grad_initial_usage_ref, final[9].astype(gradient_dtypes[13])
    )
    grad_mu_ref[:] = final[10][None, :, :]
    grad_tau_read_ref[:] = final[11][None, :]
    grad_tau_write_ref[:] = jnp.reshape(final[12], (1, 1))


def _screening_tpu_backward_kernel(
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
    gradient_dtypes = step_config.pop("gradient_dtypes")
    mu = mu_ref[:][0, :, :].astype(jnp.float32)
    tau_read = tau_read_ref[:][0, :].astype(jnp.float32)
    tau_write = tau_write_ref[:][0, 0].astype(jnp.float32)
    bank_ids = bank_ids_ref[:][0, :, 0].astype(jnp.int32)
    reverse_carry = (
        _load_state_matrix(cotangent_final_slots_ref).astype(jnp.float32),
        jnp.zeros_like(_load_state_matrix(initial_read_keys_ref), dtype=jnp.float32),
        jnp.zeros_like(_load_state_matrix(initial_values_ref), dtype=jnp.float32),
        jnp.zeros_like(_load_state_matrix(initial_write_keys_ref), dtype=jnp.float32),
        _load_state_vector(cotangent_final_ages_ref).astype(jnp.float32),
        _load_state_vector(cotangent_final_usage_ref).astype(jnp.float32),
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
        admission_t = _load_time_vector(admission_ref, t).astype(jnp.float32)
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
        gradients = pullback(
            (
                _load_time_vector(cotangent_u_ref, t).astype(jnp.float32),
                gradient_carry[:6],
            )
        )
        _store_time_vector(
            grad_q_read_ref, t, gradients[1].astype(gradient_dtypes[0])
        )
        _store_time_vector(
            grad_q_write_ref, t, gradients[2].astype(gradient_dtypes[1])
        )
        _store_time_vector(
            grad_admission_ref, t, gradients[3].astype(gradient_dtypes[2])
        )
        _store_time_vector(
            grad_bank_logits_ref, t, gradients[4].astype(gradient_dtypes[3])
        )
        _store_time_matrix(
            grad_delta_slots_ref, t, gradients[5].astype(gradient_dtypes[4])
        )
        _store_time_matrix(
            grad_delta_read_keys_ref,
            t,
            gradients[6].astype(gradient_dtypes[5]),
        )
        _store_time_matrix(
            grad_delta_values_ref,
            t,
            gradients[7].astype(gradient_dtypes[6]),
        )
        _store_time_matrix(
            grad_delta_write_keys_ref,
            t,
            gradients[8].astype(gradient_dtypes[7]),
        )
        return (
            *gradients[0],
            gradient_carry[6] + gradients[9],
            gradient_carry[7] + gradients[10],
            gradient_carry[8] + gradients[11],
        )

    final_gradients = reverse_loop
    _store_state_matrix(
        grad_initial_slots_ref, final_gradients[0].astype(gradient_dtypes[8])
    )
    _store_state_matrix(
        grad_initial_read_keys_ref,
        final_gradients[1].astype(gradient_dtypes[9]),
    )
    _store_state_matrix(
        grad_initial_values_ref,
        final_gradients[2].astype(gradient_dtypes[10]),
    )
    _store_state_matrix(
        grad_initial_write_keys_ref,
        final_gradients[3].astype(gradient_dtypes[11]),
    )
    _store_state_vector(
        grad_initial_ages_ref,
        final_gradients[4].astype(gradient_dtypes[12]),
    )
    _store_state_vector(
        grad_initial_usage_ref,
        final_gradients[5].astype(gradient_dtypes[13]),
    )
    grad_mu_ref[:] = final_gradients[6][None, :, :]
    grad_tau_read_ref[:] = final_gradients[7][None, :]
    grad_tau_write_ref[:] = jnp.reshape(final_gradients[8], (1, 1))


def screening_pallas_tpu_forward(
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
    interpret: bool = False,
):
    """Run the TPU-specialized projected-screening forward recurrence."""

    time, batch, key_size = q_read.shape
    _, n_slots, slot_size = initial_slots.shape
    value_size = initial_values.shape[-1]

    specs = _screening_tpu_specs(
        time, key_size, n_slots, slot_size, value_size
    )

    kernel = partial(
        _screening_tpu_forward_kernel,
        output_dtype=delta_values.dtype,
        **_screening_step_config(config, time),
    )
    result = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(
                (time, batch, 1, value_size), delta_values.dtype
            ),
            jax.ShapeDtypeStruct(initial_slots.shape, jnp.float32),
            jax.ShapeDtypeStruct((*initial_ages.shape, 1), jnp.float32),
            jax.ShapeDtypeStruct((*initial_usage.shape, 1), jnp.float32),
            jax.ShapeDtypeStruct(
                (time, batch, 1, _STEP_STAT_COUNT), jnp.float32
            ),
            jax.ShapeDtypeStruct((time, batch, n_slots, 1), jnp.float32),
        ),
        grid=(batch,),
        in_specs=(
            specs["q"], specs["q"], specs["time_feature"](1),
            specs["time_feature"](3),
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
        name="rwkv7_screening_tpu_forward",
    )(
        q_read[:, :, None, :],
        q_write[:, :, None, :],
        admission[:, :, None, None],
        bank_logits[:, :, None, :],
        delta_slots,
        delta_read_keys,
        delta_values,
        delta_write_keys,
        initial_slots,
        initial_read_keys,
        initial_values,
        initial_write_keys,
        initial_ages[:, :, None],
        initial_usage[:, :, None],
        mu[None, :, None],
        jnp.reshape(tau_read, (1, config.n_read_tiles)),
        jnp.reshape(tau_write, (1, 1)),
        jnp.asarray(
            config.bank_ids or (0,) * n_slots, dtype=jnp.int32
        )[None, :, None],
    )
    return (
        result[0][:, :, 0, :],
        result[1],
        result[2][:, :, 0],
        result[3][:, :, 0],
        result[4][:, :, 0, :],
        result[5][:, :, :, 0],
    )


def _screening_tpu_specs(
    time: int,
    key_size: int,
    n_slots: int,
    slot_size: int,
    value_size: int,
):
    q = pl.BlockSpec(
        (time, 1, 1, key_size), lambda batch_id: (0, batch_id, 0, 0)
    )

    def time_feature(feature_size):
        return pl.BlockSpec(
            (time, 1, 1, feature_size),
            lambda batch_id: (0, batch_id, 0, 0),
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
        (1, n_slots, 1), lambda batch_id: (batch_id, 0, 0)
    )
    return {
        "q": q,
        "time_feature": time_feature,
        "time_matrix": time_matrix,
        "sequence_matrix": sequence_matrix,
        "state_matrix": state_matrix,
        "state_vector": state_vector,
        "shared_vector": pl.BlockSpec(
            (1, n_slots, 1), lambda batch_id: (0, 0, 0)
        ),
        "shared_scalar": pl.BlockSpec((1, 1), lambda batch_id: (0, 0)),
        "shared_feature": lambda size: pl.BlockSpec(
            (1, size), lambda batch_id: (0, 0)
        ),
        "u": pl.BlockSpec(
            (time, 1, 1, value_size),
            lambda batch_id: (0, batch_id, 0, 0),
        ),
        "statistics": pl.BlockSpec(
            (time, 1, 1, _STEP_STAT_COUNT),
            lambda batch_id: (0, batch_id, 0, 0),
        ),
        "update": pl.BlockSpec(
            (time, 1, n_slots, 1),
            lambda batch_id: (0, batch_id, 0, 0),
        ),
        "per_batch_vector": pl.BlockSpec(
            (1, n_slots, 1), lambda batch_id: (batch_id, 0, 0)
        ),
        "per_batch_scalar": pl.BlockSpec(
            (1, 1), lambda batch_id: (batch_id, 0)
        ),
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


def _screening_pallas_tpu_forward_with_checkpoints(
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
    interpret,
):
    time, batch, key_size = q_read.shape
    _, n_slots, slot_size = initial_slots.shape
    value_size = initial_values.shape[-1]
    checkpoint_interval = config.checkpoint_interval
    checkpoint_count = (
        time + checkpoint_interval - 1
    ) // checkpoint_interval + 1
    specs = _screening_tpu_specs(
        time, key_size, n_slots, slot_size, value_size
    )
    result = pl.pallas_call(
        partial(
            _screening_tpu_checkpoint_forward_kernel,
            checkpoint_interval=checkpoint_interval,
            output_dtype=delta_values.dtype,
            **_screening_step_config(config, time),
        ),
        out_shape=(
            jax.ShapeDtypeStruct(
                (time, batch, 1, value_size), delta_values.dtype
            ),
            jax.ShapeDtypeStruct(initial_slots.shape, jnp.float32),
            jax.ShapeDtypeStruct((*initial_ages.shape, 1), jnp.float32),
            jax.ShapeDtypeStruct((*initial_usage.shape, 1), jnp.float32),
            jax.ShapeDtypeStruct(
                (time, batch, 1, _STEP_STAT_COUNT), jnp.float32
            ),
            jax.ShapeDtypeStruct((time, batch, n_slots, 1), jnp.float32),
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
            jax.ShapeDtypeStruct((time, batch, n_slots, 1), jnp.float32),
            jax.ShapeDtypeStruct((time, batch, n_slots, 1), jnp.float32),
            jax.ShapeDtypeStruct((time, batch, n_slots, 1), jnp.float32),
        ),
        grid=(batch,),
        in_specs=(
            specs["q"], specs["q"], specs["time_feature"](1),
            specs["time_feature"](3),
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
        name="rwkv7_screening_tpu_checkpoint_forward",
    )(
        q_read[:, :, None, :],
        q_write[:, :, None, :],
        admission[:, :, None, None],
        bank_logits[:, :, None, :],
        delta_slots,
        delta_read_keys,
        delta_values,
        delta_write_keys,
        initial_slots,
        initial_read_keys,
        initial_values,
        initial_write_keys,
        initial_ages[:, :, None],
        initial_usage[:, :, None],
        mu[None, :, None],
        jnp.reshape(tau_read, (1, config.n_read_tiles)),
        jnp.reshape(tau_write, (1, 1)),
        jnp.asarray(
            config.bank_ids or (0,) * n_slots, dtype=jnp.int32
        )[None, :, None],
    )
    outputs = (
        result[0][:, :, 0, :],
        result[1],
        result[2][:, :, 0],
        result[3][:, :, 0],
        result[4][:, :, 0, :],
        result[5][:, :, :, 0],
    )
    aux = (
        *result[6:10],
        result[10][:, :, :, 0],
        result[11][:, :, :, 0],
        result[12][:, :, :, 0],
    )
    return outputs, aux


def screening_pallas_tpu_forward_with_aux(
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
    interpret: bool = False,
):
    """Run training forward and return the FP32 carry tape for the VJP."""

    if config.checkpoint_interval is not None:
        return _screening_pallas_tpu_forward_with_checkpoints(
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
            interpret=interpret,
        )
    time, batch, key_size = q_read.shape
    _, n_slots, slot_size = initial_slots.shape
    value_size = initial_values.shape[-1]
    specs = _screening_tpu_specs(
        time, key_size, n_slots, slot_size, value_size
    )
    result = pl.pallas_call(
        partial(
            _screening_tpu_training_forward_kernel,
            output_dtype=delta_values.dtype,
            **_screening_step_config(config, time),
        ),
        out_shape=(
            jax.ShapeDtypeStruct(
                (time, batch, 1, value_size), delta_values.dtype
            ),
            jax.ShapeDtypeStruct(initial_slots.shape, jnp.float32),
            jax.ShapeDtypeStruct((*initial_ages.shape, 1), jnp.float32),
            jax.ShapeDtypeStruct((*initial_usage.shape, 1), jnp.float32),
            jax.ShapeDtypeStruct(
                (time, batch, 1, _STEP_STAT_COUNT), jnp.float32
            ),
            jax.ShapeDtypeStruct((time, batch, n_slots, 1), jnp.float32),
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
            jax.ShapeDtypeStruct((time, batch, n_slots, 1), jnp.float32),
            jax.ShapeDtypeStruct((time, batch, n_slots, 1), jnp.float32),
        ),
        grid=(batch,),
        in_specs=(
            specs["q"], specs["q"], specs["time_feature"](1),
            specs["time_feature"](3),
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
            specs["time_matrix"](slot_size),
            specs["time_matrix"](key_size),
            specs["time_matrix"](value_size),
            specs["time_matrix"](key_size),
            specs["update"], specs["update"],
        ),
        interpret=interpret,
        name="rwkv7_screening_tpu_training_forward",
    )(
        q_read[:, :, None, :], q_write[:, :, None, :],
        admission[:, :, None, None], bank_logits[:, :, None, :],
        delta_slots, delta_read_keys, delta_values,
        delta_write_keys, initial_slots, initial_read_keys,
        initial_values, initial_write_keys,
        initial_ages[:, :, None], initial_usage[:, :, None],
        mu[None, :, None],
        jnp.reshape(tau_read, (1, config.n_read_tiles)),
        jnp.reshape(tau_write, (1, 1)),
        jnp.asarray(
            config.bank_ids or (0,) * n_slots, dtype=jnp.int32
        )[None, :, None],
    )
    outputs = (
        result[0][:, :, 0, :],
        result[1],
        result[2][:, :, 0],
        result[3][:, :, 0],
        result[4][:, :, 0, :],
        result[5][:, :, :, 0],
    )
    aux = (
        *result[6:10],
        result[10][:, :, :, 0],
        result[11][:, :, :, 0],
    )
    return outputs, aux


def _screening_pallas_tpu_backward_with_checkpoints(
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
    interpret,
):
    time, batch, key_size = q_read.shape
    _, n_slots, slot_size = initial_slots.shape
    value_size = initial_values.shape[-1]
    checkpoint_count = checkpoint_slots.shape[0]
    specs = _screening_tpu_specs(
        time, key_size, n_slots, slot_size, value_size
    )
    arrays = (
        q_read[:, :, None, :],
        q_write[:, :, None, :],
        admission[:, :, None, None],
        bank_logits[:, :, None, :],
        delta_slots,
        delta_read_keys,
        delta_values,
        delta_write_keys,
        initial_slots,
        initial_read_keys,
        initial_values,
        initial_write_keys,
        initial_ages[:, :, None],
        initial_usage[:, :, None],
        mu[None, :, None],
        jnp.reshape(tau_read, (1, config.n_read_tiles)),
        jnp.reshape(tau_write, (1, 1)),
        jnp.asarray(
            config.bank_ids or (0,) * n_slots, dtype=jnp.int32
        )[None, :, None],
        cotangent_u[:, :, None, :],
        cotangent_final_slots,
        cotangent_final_ages[:, :, None],
        cotangent_final_usage[:, :, None],
        checkpoint_slots,
        checkpoint_read_keys,
        checkpoint_values,
        checkpoint_write_keys,
        tape_ages[:, :, :, None],
        tape_usage[:, :, :, None],
        tape_strength[:, :, :, None],
    )
    result = pl.pallas_call(
        partial(
            _screening_tpu_checkpoint_backward_kernel,
            checkpoint_interval=config.checkpoint_interval,
            gradient_dtypes=tuple(value.dtype for value in arrays[:14]),
            **_screening_step_config(config, time),
        ),
        out_shape=(
            *(jax.ShapeDtypeStruct(value.shape, value.dtype) for value in arrays[:14]),
            jax.ShapeDtypeStruct((batch, n_slots, 1), jnp.float32),
            jax.ShapeDtypeStruct(
                (batch, config.n_read_tiles), jnp.float32
            ),
            jax.ShapeDtypeStruct((batch, 1), jnp.float32),
        ),
        grid=(batch,),
        in_specs=(
            specs["q"], specs["q"], specs["time_feature"](1),
            specs["time_feature"](3),
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
            specs["q"], specs["q"], specs["time_feature"](1),
            specs["time_feature"](3),
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
        name="rwkv7_screening_tpu_checkpoint_backward",
    )(*arrays)
    return (
        result[0][:, :, 0, :],
        result[1][:, :, 0, :],
        result[2][:, :, 0, 0],
        result[3][:, :, 0, :],
        *result[4:12],
        result[12][:, :, 0],
        result[13][:, :, 0],
        jnp.sum(result[14], axis=0)[:, 0].astype(mu.dtype),
        jnp.sum(result[15], axis=0).reshape(tau_read.shape).astype(tau_read.dtype),
        jnp.sum(result[16], axis=0).reshape(tau_write.shape).astype(tau_write.dtype),
    )


def screening_pallas_tpu_backward(
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
    interpret: bool = False,
):
    """Run the dedicated reverse-time projected-screening TPU kernel."""

    if config.checkpoint_interval is not None:
        return _screening_pallas_tpu_backward_with_checkpoints(
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
    specs = _screening_tpu_specs(
        time, key_size, n_slots, slot_size, value_size
    )
    arrays = (
        q_read[:, :, None, :], q_write[:, :, None, :],
        admission[:, :, None, None], bank_logits[:, :, None, :],
        delta_slots, delta_read_keys, delta_values,
        delta_write_keys, initial_slots, initial_read_keys,
        initial_values, initial_write_keys,
        initial_ages[:, :, None], initial_usage[:, :, None],
        mu[None, :, None],
        jnp.reshape(tau_read, (1, config.n_read_tiles)),
        jnp.reshape(tau_write, (1, 1)),
        jnp.asarray(
            config.bank_ids or (0,) * n_slots, dtype=jnp.int32
        )[None, :, None],
        cotangent_u[:, :, None, :], cotangent_final_slots,
        cotangent_final_ages[:, :, None],
        cotangent_final_usage[:, :, None],
        tape_slots, tape_read_keys, tape_values, tape_write_keys,
        tape_ages[:, :, :, None], tape_usage[:, :, :, None],
    )
    result = pl.pallas_call(
        partial(
            _screening_tpu_backward_kernel,
            gradient_dtypes=tuple(value.dtype for value in arrays[:14]),
            **_screening_step_config(config, time),
        ),
        out_shape=(
            *(jax.ShapeDtypeStruct(value.shape, value.dtype) for value in arrays[:14]),
            jax.ShapeDtypeStruct((batch, n_slots, 1), jnp.float32),
            jax.ShapeDtypeStruct(
                (batch, config.n_read_tiles), jnp.float32
            ),
            jax.ShapeDtypeStruct((batch, 1), jnp.float32),
        ),
        grid=(batch,),
        in_specs=(
            specs["q"], specs["q"], specs["time_feature"](1),
            specs["time_feature"](3),
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
            specs["q"], specs["q"], specs["time_feature"](1),
            specs["time_feature"](3),
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
        name="rwkv7_screening_tpu_backward",
    )(*arrays)
    return (
        result[0][:, :, 0, :],
        result[1][:, :, 0, :],
        result[2][:, :, 0, 0],
        result[3][:, :, 0, :],
        *result[4:12],
        result[12][:, :, 0],
        result[13][:, :, 0],
        jnp.sum(result[14], axis=0)[:, 0].astype(mu.dtype),
        jnp.sum(result[15], axis=0).reshape(tau_read.shape).astype(tau_read.dtype),
        jnp.sum(result[16], axis=0).reshape(tau_write.shape).astype(tau_write.dtype),
    )


__all__ = [
    "screening_pallas_tpu_backward",
    "screening_pallas_tpu_forward",
    "screening_pallas_tpu_forward_with_aux",
]
