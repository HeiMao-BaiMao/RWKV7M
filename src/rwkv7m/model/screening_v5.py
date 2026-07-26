"""Portable reference recurrence for the versioned Screening v5 core.

The paper requires this semantic profile to pass a portable-reference gate
before accelerator kernels or reverse-reconstruction checkpointing are
implemented.  This module therefore owns the v5 equations and intentionally
does not reuse the v4 Pallas bodies.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .screening import (
    bounded_non_amplifying_aggregate,
    capacity_calibrated_similarity_threshold,
    smooth_trim_square,
    tanh_norm,
    trim_square,
    unit_norm,
)


READ_MEAN = 0
READ_MAX = 1
ACTIVE_SLOTS = 2
Z_NORM = 3
U_NORM = 4
WRITE_MEAN = 5
WRITE_EFFECTIVE_MEAN = 6
USAGE_MEAN = 7
MATCHED_ROUTE_MASS = 8
NOVEL_ROUTE_MASS = 9
ROUTE_ENTROPY = 10
ROUTE_TOP1 = 11
ADMISSION_MEAN = 12
NOVEL_RATE = 13
REJECTED_RATE = 14
BANK_SHORT_WRITE_MASS = 15
BANK_MID_WRITE_MASS = 16
BANK_LONG_WRITE_MASS = 17
EVICTION_AGE_MEAN = 18
EVICTION_USAGE_MEAN = 19
ADMISSION_LOW_RATE = 20
ADMISSION_HIGH_RATE = 21
OCCUPANCY_MEAN = 22
MATCHED_ERASE_MASS = 23
MATCHED_WRITE_MASS = 24
NOVEL_ERASE_MASS = 25
NOVEL_WRITE_MASS = 26
ACCEPTED_NOVEL_RATE = 27
EMPTY_ALLOCATION_RATE = 28
OCCUPIED_EVICTION_RATE = 29
READ_ENERGY_MEAN = 30
WRITE_SATURATION_RATE = 31
TAU_READ_MEAN = 32
TAU_WRITE_MEAN = 33
TAU_NOVEL_MEAN = 34
WRITE_BUDGET_RATE = 35
SELF_WRITE_SIMILARITY = 36
SELF_READ_SIMILARITY = 37
SELF_INDEX_LOSS = 38
SCREENING_V5_STEP_STAT_COUNT = 39

_SUPPORTED_VECTOR_DTYPES = (
    jnp.dtype(jnp.bfloat16),
    jnp.dtype(jnp.float32),
)


@dataclass(frozen=True)
class ScreeningV5RecurrenceConfig:
    use_value_unit_norm: bool
    usage_ema_decay: float
    tanh_norm_cap: float
    eps: float
    bank_ids: tuple[int, ...]
    route_power: float
    novelty_temperature: float
    admission_threshold: float
    allocation_temperature: float
    bank_route_temperature: float
    allocation_age_weight: float
    allocation_usage_weight: float
    allocation_redundancy_weight: float
    n_read_tiles: int
    tau_min: float
    tau_max: float
    target_false_read_rate: float
    target_false_write_rate: float
    target_false_match_rate: float
    threshold_warmup_by_load: bool
    threshold_warmup_tau: float
    eta_ambiguity: float
    edit_mode: str
    write_accounting_floor: float
    read_soft_warmup_alpha: float | jax.Array = 0.0
    read_soft_warmup_temperature: float = 0.1
    self_index_margin: float = 0.0
    training_activation: float | jax.Array = 1.0
    norm_eps: float = 1e-6


def _masked_softmax(logits, mask, *, temperature, eps):
    mask = jnp.broadcast_to(mask, logits.shape)
    scaled = logits / temperature
    masked = jnp.where(mask, scaled, jnp.asarray(-1e30, scaled.dtype))
    maximum = jnp.max(masked, axis=-1, keepdims=True)
    exponent = jnp.where(mask, jnp.exp(masked - maximum), 0.0)
    return exponent / (jnp.sum(exponent, axis=-1, keepdims=True) + eps)


def deterministic_lowest_argmax(logits, mask):
    """One-hot argmax with a backend-independent lowest-index tie break."""

    masked = jnp.where(mask, logits, jnp.asarray(-1e30, logits.dtype))
    maximum = jnp.max(
        masked,
        axis=-1,
        keepdims=True,
    )
    tied = (masked == maximum) & mask
    hard = tied & (jnp.cumsum(tied, axis=-1) == 1)
    any_valid = jnp.any(mask, axis=-1, keepdims=True)
    return (hard & any_valid).astype(jnp.float32)


def _straight_through_route(logits, mask, *, temperature, eps):
    soft = _masked_softmax(
        logits,
        mask,
        temperature=temperature,
        eps=eps,
    )
    hard = deterministic_lowest_argmax(logits, mask)
    route = soft + jax.lax.stop_gradient(hard - soft)
    return route, hard, soft


def _masked_minmax_normalize(values, mask, eps):
    minimum = jnp.min(
        jnp.where(mask, values, jnp.asarray(1e30, values.dtype)),
        axis=-1,
        keepdims=True,
    )
    maximum = jnp.max(
        jnp.where(mask, values, jnp.asarray(-1e30, values.dtype)),
        axis=-1,
        keepdims=True,
    )
    span = maximum - minimum
    # Equal (or numerically indistinguishable) allocation statistics carry
    # no ranking information. Dividing their zero numerator by ``eps`` keeps
    # the forward value at zero but creates an arbitrary O(1 / eps) reverse
    # signal through tie-breaking reductions. Recurrent usage statistics can
    # encounter that tie for hundreds of tokens, so suppress the undefined
    # preference instead of amplifying it.
    has_rank_signal = jax.lax.stop_gradient(span > eps)
    denominator = jnp.where(has_rank_signal, span + eps, 1.0)
    normalized = (values - minimum) / denominator
    return jnp.where(mask & has_rank_signal, normalized, 0.0)


def _capacity_tau(
    occupied_count,
    *,
    tests_per_slot,
    key_dimension,
    false_positive_rate,
    learned_offset,
    slot_count,
    warmup_by_load,
    config,
):
    test_count = jnp.maximum(occupied_count * tests_per_slot, 1.0)
    base = capacity_calibrated_similarity_threshold(
        test_count,
        key_dimension,
        false_positive_rate,
        tau_min=config.tau_min,
        tau_max=config.tau_max,
    )
    tau = jnp.clip(
        base[..., None] + learned_offset,
        config.tau_min,
        config.tau_max,
    )
    if warmup_by_load:
        load = jnp.clip(occupied_count / float(slot_count), 0.0, 1.0)
        tau = (
            config.threshold_warmup_tau
            + load[..., None] * (tau - config.threshold_warmup_tau)
        )
    return jnp.clip(tau, config.tau_min, config.tau_max)


def _slot_redundancy(write_keys, occupancy, bank_ids, eps):
    normalized = unit_norm(write_keys.astype(jnp.float32), eps=eps)
    similarity = jnp.einsum("bmk,bnk->bmn", normalized, normalized)
    slot_count = write_keys.shape[1]
    same_bank = bank_ids[:, None] == bank_ids[None, :]
    non_diagonal = ~jnp.eye(slot_count, dtype=jnp.bool_)
    occupied_pair = (
        occupancy[:, :, None] > 0.5
    ) & (occupancy[:, None, :] > 0.5)
    mask = occupied_pair & same_bank[None, :, :] & non_diagonal[None, :, :]
    maximum = jnp.max(
        jnp.where(mask, similarity, -1.0),
        axis=-1,
    )
    return jnp.maximum(maximum, 0.0)


def ambiguity_aware_matched_routing(
    eligibility,
    *,
    route_power,
    eta_ambiguity,
):
    """Return address distribution and non-amplified matched confidence."""

    powered = jnp.power(eligibility.astype(jnp.float32), route_power)
    powered_sum = jnp.sum(powered, axis=-1, keepdims=True)
    distribution = powered / jnp.where(
        powered_sum > 0.0,
        powered_sum,
        1.0,
    )
    absolute_confidence = jnp.max(eligibility, axis=-1)
    concentration = jnp.sum(distribution**2, axis=-1)
    if eta_ambiguity == 0.0:
        matched_confidence = absolute_confidence
    else:
        # A positive eligibility can still underflow to zero after the
        # routing power. In that case the address distribution has no
        # representable mass, so differentiating ``concentration ** eta`` at
        # zero would produce an infinite derivative for 0 < eta < 1. Base
        # the guard on the powered distribution that is actually routed.
        has_distribution = powered_sum[..., 0] > 0.0
        # ``concentration ** eta`` has an infinite derivative at zero when
        # 0 < eta < 1.  JAX evaluates both sides of ``where`` during tracing,
        # so the semantically inactive branch must also be numerically safe.
        safe_concentration = jnp.where(
            has_distribution,
            concentration,
            1.0,
        )
        matched_confidence = jnp.where(
            has_distribution,
            absolute_confidence * safe_concentration**eta_ambiguity,
            0.0,
        )
    return (
        distribution,
        absolute_confidence,
        concentration,
        matched_confidence,
    )


def _validate_inputs(
    q_read,
    q_write,
    admission_context_logits,
    admission_feature_weights,
    bank_logits,
    matched_erase_logits,
    matched_write_logits,
    novel_erase_logits,
    novel_write_logits,
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
    initial_occupancy,
    mu,
    tau_read_offset,
    tau_write_offset,
    *,
    config,
):
    if q_read.ndim != 3:
        raise ValueError("q_read must have shape [time, batch, key]")
    time, batch, key_size = q_read.shape
    if q_write.shape != q_read.shape:
        raise ValueError("q_write must match q_read")
    if q_read.dtype != jnp.float32 or q_write.dtype != jnp.float32:
        raise TypeError("v5 normalized queries must be float32")
    if admission_context_logits.shape != (time, batch):
        raise ValueError("admission_context_logits must have shape [time, batch]")
    if admission_feature_weights.shape != (5,):
        raise ValueError("admission_feature_weights must have shape [5]")
    if bank_logits.shape != (time, batch, 3):
        raise ValueError("bank_logits must have shape [time, batch, 3]")
    if initial_slots.ndim != 3:
        raise ValueError("initial_slots must have shape [batch, slots, slot]")
    state_batch, slot_count, slot_size = initial_slots.shape
    if state_batch != batch:
        raise ValueError("query and state batch sizes must match")
    if not config.bank_ids or len(config.bank_ids) != slot_count:
        raise ValueError("v5 bank_ids must contain one entry per slot")
    if any(bank not in (0, 1, 2) for bank in config.bank_ids):
        raise ValueError("v5 bank_ids must contain only 0, 1, or 2")
    if set(config.bank_ids) != {0, 1, 2}:
        raise ValueError("v5 requires at least one slot in each bank")
    scalar_time_slot = (time, batch, slot_count)
    for name, value in (
        ("matched_erase_logits", matched_erase_logits),
        ("matched_write_logits", matched_write_logits),
    ):
        if value.shape != scalar_time_slot:
            raise ValueError(f"{name} must have shape {scalar_time_slot}")
    for name, value in (
        ("novel_erase_logits", novel_erase_logits),
        ("novel_write_logits", novel_write_logits),
    ):
        if value.shape != (time, batch):
            raise ValueError(f"{name} must have shape {(time, batch)}")
    if delta_slots.shape != (time, batch, slot_count, slot_size):
        raise ValueError("delta_slots has an invalid shape")
    if initial_read_keys.shape != (batch, slot_count, key_size):
        raise ValueError("initial_read_keys has an invalid shape")
    if initial_write_keys.shape != initial_read_keys.shape:
        raise ValueError("initial_write_keys must match initial_read_keys")
    if delta_read_keys.shape != (time, *initial_read_keys.shape):
        raise ValueError("delta_read_keys has an invalid shape")
    if delta_write_keys.shape != delta_read_keys.shape:
        raise ValueError("delta_write_keys must match delta_read_keys")
    if initial_values.ndim != 3 or initial_values.shape[:2] != (
        batch,
        slot_count,
    ):
        raise ValueError("initial_values has an invalid shape")
    value_size = initial_values.shape[-1]
    if delta_values.shape != (time, batch, slot_count, value_size):
        raise ValueError("delta_values has an invalid shape")
    if key_size % config.n_read_tiles:
        raise ValueError("key size must divide evenly into read tiles")
    if value_size % config.n_read_tiles:
        raise ValueError("value size must divide evenly into read tiles")
    scalar_state_shape = (batch, slot_count)
    for name, value in (
        ("initial_ages", initial_ages),
        ("initial_usage", initial_usage),
        ("initial_occupancy", initial_occupancy),
    ):
        if value.shape != scalar_state_shape or value.dtype != jnp.float32:
            raise TypeError(f"{name} must be float32 {scalar_state_shape}")
    for name, value in (
        ("delta_slots", delta_slots),
        ("delta_read_keys", delta_read_keys),
        ("delta_values", delta_values),
        ("delta_write_keys", delta_write_keys),
        ("initial_slots", initial_slots),
        ("initial_read_keys", initial_read_keys),
        ("initial_values", initial_values),
        ("initial_write_keys", initial_write_keys),
    ):
        if value.dtype not in _SUPPORTED_VECTOR_DTYPES:
            raise TypeError(f"{name} must be bfloat16 or float32")
    if mu.shape != (slot_count,) or mu.dtype != jnp.float32:
        raise TypeError(f"mu must be float32 {(slot_count,)}")
    expected_read_offset = (
        ()
        if config.n_read_tiles == 1
        else (config.n_read_tiles,)
    )
    if tau_read_offset.shape != expected_read_offset:
        raise ValueError(
            f"tau_read_offset must have shape {expected_read_offset}"
        )
    if tau_write_offset.shape != ():
        raise ValueError("tau_write_offset must be scalar")


def screening_v5_recurrence_reference(
    q_read,
    q_write,
    admission_context_logits,
    admission_feature_weights,
    bank_logits,
    matched_erase_logits,
    matched_write_logits,
    novel_erase_logits,
    novel_write_logits,
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
    initial_occupancy,
    mu,
    tau_read_offset,
    tau_write_offset,
    config: ScreeningV5RecurrenceConfig,
    *,
    admission_hard_mask=None,
):
    """Execute the complete v5-core recurrence with portable JAX."""

    if config.read_soft_warmup_temperature <= 0.0:
        raise ValueError("read_soft_warmup_temperature must be positive")
    if config.norm_eps <= 0.0:
        raise ValueError("norm_eps must be positive")
    if not 0.0 <= config.self_index_margin < 1.0:
        raise ValueError("self_index_margin must be in [0, 1)")

    inputs = (
        q_read,
        q_write,
        admission_context_logits,
        admission_feature_weights,
        bank_logits,
        matched_erase_logits,
        matched_write_logits,
        novel_erase_logits,
        novel_write_logits,
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
        initial_occupancy,
        mu,
        tau_read_offset,
        tau_write_offset,
    )
    _validate_inputs(*inputs, config=config)
    has_admission_hard_mask = admission_hard_mask is not None
    if admission_hard_mask is None:
        admission_hard_mask = jnp.zeros_like(
            admission_context_logits,
            dtype=jnp.float32,
        )
    elif admission_hard_mask.shape != admission_context_logits.shape:
        raise ValueError(
            "admission_hard_mask must match admission_context_logits shape"
        )
    admission_hard_mask = jax.lax.stop_gradient(
        admission_hard_mask.astype(jnp.float32)
    )
    bank_ids = jnp.asarray(config.bank_ids, dtype=jnp.int32)
    slot_count = initial_slots.shape[1]
    key_size = q_read.shape[-1]
    read_tiles = config.n_read_tiles
    key_tile_size = key_size // read_tiles
    occupancy_levels = jnp.arange(slot_count + 1, dtype=jnp.float32)
    # Thresholds depend only on discrete occupancy and learned offsets. Build
    # the complete table once per recurrence instead of evaluating the null
    # quantile three times in every token step.
    read_tau_lookup = _capacity_tau(
        occupancy_levels,
        tests_per_slot=float(read_tiles),
        key_dimension=key_tile_size,
        false_positive_rate=config.target_false_read_rate,
        learned_offset=jnp.reshape(tau_read_offset, (read_tiles,)),
        slot_count=slot_count,
        warmup_by_load=config.threshold_warmup_by_load,
        config=config,
    )
    # The low-load threshold warm-up exists to keep the read path trainable
    # before enough slots are occupied.  Applying the same negative warm-up
    # threshold to write matching makes the first occupied slot match nearly
    # every token, which prevents empty-first allocation from filling the
    # remaining capacity.  Writes therefore always use the calibrated null
    # thresholds, even while reads are being warmed up.
    write_tau_lookup = _capacity_tau(
        occupancy_levels,
        tests_per_slot=1.0,
        key_dimension=key_size,
        false_positive_rate=config.target_false_write_rate,
        learned_offset=jnp.reshape(tau_write_offset, (1,)),
        slot_count=slot_count,
        warmup_by_load=False,
        config=config,
    )[..., 0]
    novel_tau_lookup = _capacity_tau(
        occupancy_levels,
        tests_per_slot=1.0,
        key_dimension=key_size,
        false_positive_rate=config.target_false_match_rate,
        learned_offset=jnp.reshape(tau_write_offset, (1,)),
        slot_count=slot_count,
        warmup_by_load=False,
        config=config,
    )[..., 0]

    def step(carry, token_inputs):
        (
            slots,
            read_keys,
            values,
            write_keys,
            ages,
            usage,
            occupancy,
        ) = carry
        (
            q_read_t,
            q_write_t,
            admission_context_t,
            bank_logits_t,
            matched_erase_logits_t,
            matched_write_logits_t,
            novel_erase_logit_t,
            novel_write_logit_t,
            delta_slots_t,
            delta_read_keys_t,
            delta_values_t,
            delta_write_keys_t,
            admission_hard_t,
        ) = token_inputs
        training_activation = jnp.clip(
            jnp.asarray(config.training_activation, dtype=jnp.float32),
            0.0,
            1.0,
        )
        training_enabled = jax.lax.stop_gradient(
            (training_activation > 0.0).astype(jnp.float32)
        )

        batch, slot_count, key_size = read_keys.shape
        value_size = values.shape[-1]
        value_tile_size = value_size // read_tiles
        occupied = jax.lax.stop_gradient((occupancy > 0.5).astype(jnp.float32))
        occupied_count = jnp.sum(occupied, axis=-1)
        occupied_index = jnp.clip(
            occupied_count.astype(jnp.int32),
            0,
            slot_count,
        )
        occupancy_selector = jax.nn.one_hot(
            occupied_index,
            slot_count + 1,
            dtype=jnp.float32,
        )
        # A dynamic gather from a replicated lookup to a data-sharded batch
        # has ambiguous output sharding under explicit meshes. The equivalent
        # one-hot selection has an unambiguous data-sharded result and keeps
        # the recurrence collective-free.
        read_tau = jnp.sum(
            occupancy_selector[..., None] * read_tau_lookup[None, ...],
            axis=1,
        )
        write_tau = jnp.sum(
            occupancy_selector * write_tau_lookup[None, :],
            axis=1,
        )
        novel_similarity_tau = jnp.sum(
            occupancy_selector * novel_tau_lookup[None, :],
            axis=1,
        )

        read_keys_tiled = read_keys.astype(jnp.float32).reshape(
            batch, slot_count, read_tiles, key_tile_size
        )
        values_tiled = values.astype(jnp.float32).reshape(
            batch, slot_count, read_tiles, value_tile_size
        )
        q_read_tiled = q_read_t.reshape(batch, read_tiles, key_tile_size)
        read_similarity = jnp.einsum(
            "brk,bmrk->brm",
            q_read_tiled,
            unit_norm(read_keys_tiled, eps=config.norm_eps),
        )
        hard_read_relevance = trim_square(
            read_similarity,
            read_tau[..., None],
            eps=config.eps,
        )
        soft_read_relevance = smooth_trim_square(
            read_similarity,
            read_tau[..., None],
            config.read_soft_warmup_temperature,
            eps=config.eps,
        )
        read_alpha = jnp.clip(
            jnp.asarray(config.read_soft_warmup_alpha, dtype=jnp.float32),
            0.0,
            1.0,
        )
        read_relevance = hard_read_relevance + read_alpha * (
            soft_read_relevance - hard_read_relevance
        )
        read_relevance *= occupied[:, None, :] * training_enabled
        normalized_values = (
            unit_norm(values_tiled, eps=config.norm_eps)
            if config.use_value_unit_norm
            else values_tiled
        )
        z_sum_tiled = jnp.einsum(
            "brm,bmrv->brv",
            read_relevance,
            normalized_values,
        )
        z_tiled, read_energy = bounded_non_amplifying_aggregate(z_sum_tiled)
        u_tiled = tanh_norm(
            z_tiled,
            cap=config.tanh_norm_cap,
            eps=config.eps,
        )
        z = z_tiled.reshape(batch, value_size)
        u = u_tiled.reshape(batch, value_size)
        read_activity = jnp.max(read_relevance, axis=1)

        write_similarity = jnp.einsum(
            "bk,bmk->bm",
            q_write_t,
            unit_norm(
                write_keys.astype(jnp.float32), eps=config.norm_eps
            ),
        )
        eligibility = trim_square(
            write_similarity,
            write_tau[:, None],
            eps=config.eps,
        )
        eligibility *= occupied
        (
            address_distribution,
            _,
            _,
            matched_confidence,
        ) = ambiguity_aware_matched_routing(
            eligibility,
            route_power=config.route_power,
            eta_ambiguity=config.eta_ambiguity,
        )

        novelty_threshold = trim_square(
            novel_similarity_tau,
            write_tau,
            eps=config.eps,
        )
        novel_hard = (
            (occupied_count == 0)
            | (jnp.max(eligibility, axis=-1) <= 0.0)
            | (matched_confidence < novelty_threshold)
        ).astype(jnp.float32)
        novel_soft = jax.nn.sigmoid(
            (novelty_threshold - matched_confidence)
            / config.novelty_temperature
        )
        novel_st = novel_soft + jax.lax.stop_gradient(
            novel_hard - novel_soft
        )

        bank_membership = (
            bank_ids[:, None] == jnp.arange(3, dtype=jnp.int32)[None, :]
        )
        bank_capacity = jnp.sum(
            bank_membership.astype(jnp.float32),
            axis=0,
        )
        bank_load = jnp.einsum(
            "bm,mr->br",
            occupied,
            bank_membership.astype(jnp.float32),
        ) / bank_capacity[None, :]
        memory_load = occupied_count / float(slot_count)
        admission_features = jnp.concatenate(
            [
                matched_confidence[:, None],
                memory_load[:, None],
                bank_load,
            ],
            axis=-1,
        )
        admission_logit = admission_context_t + jnp.einsum(
            "bf,f->b",
            admission_features,
            admission_feature_weights,
        )
        admission_soft = jax.nn.sigmoid(admission_logit)
        admission_hard = (
            admission_hard_t
            if has_admission_hard_mask
            else (
                admission_soft >= config.admission_threshold
            ).astype(jnp.float32)
        )
        admission_st = admission_soft + jax.lax.stop_gradient(
            admission_hard - admission_soft
        )

        available_banks = bank_capacity > 0
        bank_route, bank_hard, _ = _straight_through_route(
            bank_logits_t,
            available_banks[None, :],
            temperature=config.bank_route_temperature,
            eps=config.eps,
        )
        redundancy = (
            _slot_redundancy(
                write_keys,
                occupied,
                bank_ids,
                config.norm_eps,
            )
            if config.allocation_redundancy_weight > 0.0
            else jnp.zeros_like(occupied)
        )
        allocation_st = jnp.zeros_like(eligibility)
        allocation_hard = jnp.zeros_like(eligibility)
        empty_selected = jnp.zeros((batch,), dtype=jnp.float32)
        for bank in range(3):
            member = bank_ids == bank
            member_mask = jnp.broadcast_to(member[None, :], occupied.shape)
            empty_mask = member_mask & (occupied <= 0.5)
            has_empty = jnp.any(empty_mask, axis=-1)
            empty_hard = deterministic_lowest_argmax(
                -jnp.broadcast_to(
                    jnp.arange(slot_count, dtype=jnp.float32)[None, :],
                    occupied.shape,
                ),
                empty_mask,
            )

            victim_mask = member_mask & (occupied > 0.5)
            age_normalized = _masked_minmax_normalize(
                ages,
                victim_mask,
                config.eps,
            )
            usage_normalized = _masked_minmax_normalize(
                usage,
                victim_mask,
                config.eps,
            )
            victim_score = (
                config.allocation_age_weight * age_normalized
                - config.allocation_usage_weight * usage_normalized
                + config.allocation_redundancy_weight * redundancy
            )
            victim_route, victim_hard, _ = _straight_through_route(
                victim_score,
                victim_mask,
                temperature=config.allocation_temperature,
                eps=config.eps,
            )
            slot_route = jnp.where(
                has_empty[:, None],
                jax.lax.stop_gradient(empty_hard),
                victim_route,
            )
            slot_hard = jnp.where(
                has_empty[:, None],
                empty_hard,
                victim_hard,
            )
            allocation_st += bank_route[:, bank, None] * slot_route
            allocation_hard += bank_hard[:, bank, None] * slot_hard
            empty_selected += bank_hard[:, bank] * has_empty.astype(jnp.float32)

        matched_address = (
            (1.0 - novel_st[:, None])
            * matched_confidence[:, None]
            * address_distribution
            * training_enabled
        )
        matched_base = mu[None, :] * matched_address
        matched_erase_gate = jax.nn.sigmoid(matched_erase_logits_t)
        matched_write_gate = jax.nn.sigmoid(matched_write_logits_t)
        if config.edit_mode == "tied":
            matched_erase = matched_base
            matched_write = matched_base
        elif config.edit_mode == "capacity_conserving":
            matched_erase = matched_base * matched_erase_gate
            matched_write = matched_erase * matched_write_gate
        else:
            matched_erase = matched_base * matched_erase_gate
            matched_write = matched_base * matched_write_gate

        novel_base = (
            novel_st[:, None]
            * admission_st[:, None]
            * allocation_st
            * training_enabled
        )
        accepted_novel_hard = jax.lax.stop_gradient(
            novel_hard[:, None]
            * admission_hard[:, None]
            * allocation_hard
            * training_enabled
        )
        novel_erase_gate = jax.nn.sigmoid(novel_erase_logit_t)[:, None]
        novel_write_gate = jax.nn.sigmoid(novel_write_logit_t)[:, None]
        if config.edit_mode == "tied":
            novel_erase = occupied * novel_base
            novel_write = novel_base
        elif config.edit_mode == "capacity_conserving":
            novel_erase = occupied * novel_base * novel_erase_gate
            novel_write = (
                novel_base
                * novel_write_gate
                * ((1.0 - occupied) + occupied * novel_erase_gate)
            )
        else:
            novel_erase = occupied * novel_base * novel_erase_gate
            novel_write = novel_base * novel_write_gate

        erase_mass = matched_erase + novel_erase
        write_mass = matched_write + novel_write
        erase_vector = erase_mass[..., None]
        write_vector = write_mass[..., None]
        next_slots = (
            (1.0 - erase_vector) * slots
            + write_vector * delta_slots_t.astype(jnp.float32)
        )
        next_read_keys = (
            (1.0 - erase_vector) * read_keys
            + write_vector * delta_read_keys_t.astype(jnp.float32)
        )
        next_values = (
            (1.0 - erase_vector) * values
            + write_vector * delta_values_t.astype(jnp.float32)
        )
        next_write_keys = (
            (1.0 - erase_vector) * write_keys
            + write_vector * delta_write_keys_t.astype(jnp.float32)
        )

        applied_mass = jnp.maximum(erase_mass, write_mass)
        accounted_write = jax.lax.stop_gradient(
            (applied_mass >= config.write_accounting_floor).astype(jnp.float32)
        )
        next_occupancy = jax.lax.stop_gradient(
            jnp.maximum(occupied, accepted_novel_hard)
        )
        next_ages = jnp.where(
            accepted_novel_hard > 0.5,
            0.0,
            jnp.where(
                occupied <= 0.5,
                0.0,
                jnp.where(accounted_write > 0.5, 0.0, ages + 1.0),
            ),
        )
        next_usage = (
            config.usage_ema_decay * usage
            + (1.0 - config.usage_ema_decay) * read_activity
        )

        update = next_slots - slots
        update_squared = jnp.sum(update * update, axis=-1)
        address_mass = matched_address + novel_base
        route_mass = jnp.sum(address_mass, axis=-1)
        route_distribution = address_mass / (
            route_mass[:, None] + config.eps
        )
        route_entropy = -jnp.sum(
            route_distribution
            * jnp.log(route_distribution + config.eps),
            axis=-1,
        )
        route_top1 = jnp.max(route_distribution, axis=-1)
        bank_write_mass = tuple(
            jnp.sum(
                write_mass * (bank_ids == bank)[None, :],
                axis=-1,
            )
            for bank in range(3)
        )
        occupied_eviction = accepted_novel_hard * occupied
        eviction_count = jnp.sum(occupied_eviction, axis=-1)
        eviction_age = jnp.sum(occupied_eviction * ages, axis=-1) / (
            eviction_count + config.eps
        )
        eviction_usage = jnp.sum(occupied_eviction * usage, axis=-1) / (
            eviction_count + config.eps
        )
        accepted_novel = jnp.max(accepted_novel_hard, axis=-1)
        empty_allocation = accepted_novel * empty_selected
        candidate_write_similarity = jnp.einsum(
            "bk,bmk->bm",
            q_write_t,
            unit_norm(
                delta_write_keys_t.astype(jnp.float32),
                eps=config.norm_eps,
            ),
        )
        candidate_read_keys = delta_read_keys_t.astype(jnp.float32).reshape(
            batch, slot_count, read_tiles, key_tile_size
        )
        candidate_read_similarity = jnp.einsum(
            "brk,bmrk->brm",
            q_read_tiled,
            unit_norm(candidate_read_keys, eps=config.norm_eps),
        )
        selected_write_similarity = jnp.sum(
            allocation_hard * candidate_write_similarity,
            axis=-1,
        )
        selected_read_similarity = jnp.einsum(
            "bm,brm->br",
            allocation_hard,
            candidate_read_similarity,
        )
        write_self_target = jax.lax.stop_gradient(
            jnp.clip(
                novel_similarity_tau + config.self_index_margin,
                config.tau_min,
                config.tau_max,
            )
        )
        read_self_target = jax.lax.stop_gradient(
            jnp.clip(
                read_tau + config.self_index_margin,
                config.tau_min,
                config.tau_max,
            )
        )
        self_index_loss = accepted_novel * (
            jnp.square(
                jax.nn.relu(
                    write_self_target - selected_write_similarity
                )
            )
            + jnp.mean(
                jnp.square(
                    jax.nn.relu(
                        read_self_target - selected_read_similarity
                    )
                ),
                axis=-1,
            )
        )
        saturated = jnp.max(
            jnp.maximum(erase_mass, write_mass),
            axis=-1,
        ) >= 0.95
        step_statistics = jnp.stack(
            (
                jnp.mean(read_relevance, axis=(-2, -1)),
                jnp.max(read_relevance, axis=(-2, -1)),
                jnp.sum(read_activity > 1e-3, axis=-1).astype(jnp.float32),
                jax.lax.stop_gradient(jnp.linalg.norm(z, axis=-1)),
                jax.lax.stop_gradient(jnp.linalg.norm(u, axis=-1)),
                jnp.mean(eligibility, axis=-1),
                jnp.mean(applied_mass, axis=-1),
                jnp.mean(next_usage, axis=-1),
                jnp.sum(matched_address, axis=-1),
                jnp.sum(novel_base, axis=-1),
                route_entropy,
                route_top1,
                admission_soft,
                novel_hard,
                (jnp.sum(accounted_write, axis=-1) == 0).astype(jnp.float32),
                *bank_write_mass,
                eviction_age,
                eviction_usage,
                (admission_soft < 0.05).astype(jnp.float32),
                (admission_soft > 0.95).astype(jnp.float32),
                jnp.mean(next_occupancy, axis=-1),
                jnp.sum(matched_erase, axis=-1),
                jnp.sum(matched_write, axis=-1),
                jnp.sum(novel_erase, axis=-1),
                jnp.sum(novel_write, axis=-1),
                accepted_novel,
                empty_allocation,
                (eviction_count > 0).astype(jnp.float32),
                jnp.mean(read_energy, axis=-1),
                saturated.astype(jnp.float32),
                jnp.mean(read_tau, axis=-1),
                write_tau,
                novelty_threshold,
                training_enabled * admission_soft * novel_soft,
                accepted_novel * selected_write_similarity,
                accepted_novel * jnp.mean(
                    selected_read_similarity,
                    axis=-1,
                ),
                self_index_loss,
            ),
            axis=-1,
        )
        next_carry = (
            next_slots,
            next_read_keys,
            next_values,
            next_write_keys,
            next_ages,
            next_usage,
            next_occupancy,
        )
        return next_carry, (u, step_statistics, update_squared)

    initial_carry = (
        initial_slots.astype(jnp.float32),
        initial_read_keys.astype(jnp.float32),
        initial_values.astype(jnp.float32),
        initial_write_keys.astype(jnp.float32),
        initial_ages,
        initial_usage,
        jax.lax.stop_gradient(initial_occupancy),
    )
    time_inputs = (
        q_read,
        q_write,
        admission_context_logits,
        bank_logits,
        matched_erase_logits,
        matched_write_logits,
        novel_erase_logits,
        novel_write_logits,
        delta_slots,
        delta_read_keys,
        delta_values,
        delta_write_keys,
        admission_hard_mask,
    )
    final_carry, (u, statistics, update_squared) = jax.lax.scan(
        step,
        initial_carry,
        time_inputs,
    )
    final_slots, _, _, _, final_ages, final_usage, final_occupancy = (
        final_carry
    )
    return (
        u.astype(delta_values.dtype),
        final_slots,
        final_ages,
        final_usage,
        final_occupancy,
        statistics,
        jax.lax.stop_gradient(update_squared),
    )


__all__ = [
    "ACCEPTED_NOVEL_RATE",
    "ACTIVE_SLOTS",
    "ADMISSION_HIGH_RATE",
    "ADMISSION_LOW_RATE",
    "ADMISSION_MEAN",
    "BANK_LONG_WRITE_MASS",
    "BANK_MID_WRITE_MASS",
    "BANK_SHORT_WRITE_MASS",
    "EMPTY_ALLOCATION_RATE",
    "EVICTION_AGE_MEAN",
    "EVICTION_USAGE_MEAN",
    "MATCHED_ERASE_MASS",
    "MATCHED_ROUTE_MASS",
    "MATCHED_WRITE_MASS",
    "NOVEL_ERASE_MASS",
    "NOVEL_RATE",
    "NOVEL_ROUTE_MASS",
    "NOVEL_WRITE_MASS",
    "OCCUPANCY_MEAN",
    "OCCUPIED_EVICTION_RATE",
    "READ_ENERGY_MEAN",
    "READ_MAX",
    "READ_MEAN",
    "SELF_INDEX_LOSS",
    "SELF_READ_SIMILARITY",
    "SELF_WRITE_SIMILARITY",
    "REJECTED_RATE",
    "ROUTE_ENTROPY",
    "ROUTE_TOP1",
    "SCREENING_V5_STEP_STAT_COUNT",
    "ScreeningV5RecurrenceConfig",
    "TAU_NOVEL_MEAN",
    "TAU_READ_MEAN",
    "TAU_WRITE_MEAN",
    "U_NORM",
    "USAGE_MEAN",
    "WRITE_EFFECTIVE_MEAN",
    "WRITE_BUDGET_RATE",
    "WRITE_MEAN",
    "WRITE_SATURATION_RATE",
    "Z_NORM",
    "ambiguity_aware_matched_routing",
    "deterministic_lowest_argmax",
    "screening_v5_recurrence_reference",
]
