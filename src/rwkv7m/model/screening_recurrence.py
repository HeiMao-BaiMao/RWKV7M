"""Pallas-first state-level-screening recurrence.

Dense projections are intentionally outside this boundary. The recurrence
owns only time-dependent state, normalization, relevance, read aggregation,
and state updates. This keeps large matrix multiplication on the backend's
normal GEMM path and makes the persistent kernel independent of model width.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from rwkv7m.kernels.screening_backend import resolve_screening_backend

from .screening import (
    competitive_write_routing,
    normalize_write_mode,
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
SCREENING_STEP_STAT_COUNT = 22

_SUPPORTED_VECTOR_DTYPES = (
    jnp.dtype(jnp.bfloat16),
    jnp.dtype(jnp.float32),
)


@dataclass(frozen=True)
class ScreeningRecurrenceConfig:
    """Static numerical policy owned by the recurrent screening kernel."""

    write_enabled: bool
    use_value_unit_norm: bool
    use_leaky_warmup: bool
    leaky_alpha: float
    leaky_gamma: float
    use_age_mask: bool
    age_ref: float
    age_sigma: float
    write_rel_floor: float
    usage_ema_decay: float
    tanh_norm_cap: float
    eps: float
    write_mode: str | None = None
    bank_ids: tuple[int, ...] = ()
    route_power: float = 1.0
    novelty_threshold: float = 0.1
    allocation_temperature: float = 1.0
    bank_route_temperature: float = 1.0
    allocation_age_weight: float = 1.0
    allocation_usage_weight: float = 1.0
    hard_admission: bool = False
    admission_threshold: float = 0.5
    n_read_tiles: int = 1
    checkpoint_interval: int | None = None


def _resolved_write_mode(config: ScreeningRecurrenceConfig) -> str:
    if config.write_mode is not None:
        return normalize_write_mode(config.write_mode)
    return "legacy_threshold" if config.write_enabled else "legacy_unconditional"


def _validate_screening_recurrence_inputs(
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
    config: ScreeningRecurrenceConfig,
):
    _resolved_write_mode(config)
    if config.n_read_tiles <= 0:
        raise ValueError("n_read_tiles must be positive")
    if (
        config.checkpoint_interval is not None
        and config.checkpoint_interval <= 0
    ):
        raise ValueError("checkpoint_interval must be positive when set")
    if q_read.ndim != 3:
        raise ValueError("q_read must have shape [time, batch, key]")
    time, batch, key_size = q_read.shape
    if time <= 0 or batch <= 0 or key_size <= 0:
        raise ValueError(
            "screening time, batch, and key dimensions must be positive"
        )
    if q_write.shape != q_read.shape:
        raise ValueError(
            f"q_write must have shape {q_read.shape}, got {q_write.shape}"
        )
    if q_read.dtype != jnp.float32 or q_write.dtype != jnp.float32:
        raise TypeError("normalized screening queries must be float32")
    if admission.shape != (time, batch) or admission.dtype != jnp.float32:
        raise TypeError(
            f"admission must be float32 with shape {(time, batch)}"
        )
    if bank_logits.shape != (time, batch, 3) or bank_logits.dtype != jnp.float32:
        raise TypeError(
            f"bank_logits must be float32 with shape {(time, batch, 3)}"
        )
    if initial_slots.ndim != 3:
        raise ValueError(
            "initial_slots must have shape [batch, slots, slot_size]"
        )
    state_batch, n_slots, slot_size = initial_slots.shape
    if n_slots <= 0 or slot_size <= 0:
        raise ValueError("screening slot dimensions must be positive")
    if state_batch != batch:
        raise ValueError("query and screening-state batch sizes must match")
    if initial_slots.dtype != jnp.float32:
        raise TypeError("initial_slots must be float32")
    expected_slot_delta = (time, batch, n_slots, slot_size)
    if delta_slots.shape != expected_slot_delta:
        raise ValueError(
            "delta_slots must have shape "
            f"{expected_slot_delta}, got {delta_slots.shape}"
        )
    expected_key_state = (batch, n_slots, key_size)
    expected_key_delta = (time, *expected_key_state)
    for name, value in (
        ("delta_read_keys", delta_read_keys),
        ("delta_write_keys", delta_write_keys),
    ):
        if value.shape != expected_key_delta:
            raise ValueError(
                f"{name} must have shape {expected_key_delta}, got {value.shape}"
            )
    for name, value in (
        ("initial_read_keys", initial_read_keys),
        ("initial_write_keys", initial_write_keys),
    ):
        if value.shape != expected_key_state:
            raise ValueError(
                f"{name} must have shape {expected_key_state}, got {value.shape}"
            )
    if initial_values.ndim != 3 or initial_values.shape[:2] != (batch, n_slots):
        raise ValueError(
            "initial_values must have shape [batch, slots, value_size]"
        )
    value_size = initial_values.shape[-1]
    if value_size <= 0:
        raise ValueError("screening value dimension must be positive")
    if key_size % config.n_read_tiles != 0:
        raise ValueError("screening key size must divide evenly into read tiles")
    if value_size % config.n_read_tiles != 0:
        raise ValueError("screening value size must divide evenly into read tiles")
    if config.bank_ids and len(config.bank_ids) != n_slots:
        raise ValueError("screening recurrence bank_ids must match n_slots")
    if any(bank_id not in (0, 1, 2) for bank_id in config.bank_ids):
        raise ValueError("screening recurrence bank_ids must contain only 0, 1, or 2")
    expected_value_delta = (time, batch, n_slots, value_size)
    if delta_values.shape != expected_value_delta:
        raise ValueError(
            "delta_values must have shape "
            f"{expected_value_delta}, got {delta_values.shape}"
        )
    for name, value in (
        ("delta_slots", delta_slots),
        ("delta_read_keys", delta_read_keys),
        ("delta_values", delta_values),
        ("delta_write_keys", delta_write_keys),
        ("initial_read_keys", initial_read_keys),
        ("initial_values", initial_values),
        ("initial_write_keys", initial_write_keys),
    ):
        if value.dtype not in _SUPPORTED_VECTOR_DTYPES:
            raise TypeError(
                f"{name} must be bfloat16 or float32, got {value.dtype}"
            )
    expected_scalar_state = (batch, n_slots)
    for name, value in (
        ("initial_ages", initial_ages),
        ("initial_usage", initial_usage),
    ):
        if value.shape != expected_scalar_state or value.dtype != jnp.float32:
            raise TypeError(
                f"{name} must be float32 with shape {expected_scalar_state}"
            )
    if mu.shape != (n_slots,) or mu.dtype != jnp.float32:
        raise TypeError(f"mu must be float32 with shape {(n_slots,)}")
    expected_tau_read_shape = () if config.n_read_tiles == 1 else (
        config.n_read_tiles,
    )
    if tau_read.shape != expected_tau_read_shape or tau_read.dtype != jnp.float32:
        raise TypeError(
            "tau_read must be float32 with shape "
            f"{expected_tau_read_shape}"
        )
    if tau_write.shape != () or tau_write.dtype != jnp.float32:
        raise TypeError("tau_write must be a float32 scalar")


def _screening_recurrence_reference_impl(
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
    config: ScreeningRecurrenceConfig,
):
    def step(carry, inputs):
        slots, read_keys, values, write_keys, ages, usage = carry
        (
            q_read_t,
            q_write_t,
            admission_t,
            bank_logits_t,
            delta_slots_t,
            delta_read_keys_t,
            delta_values_t,
            delta_write_keys_t,
        ) = inputs

        batch, n_slots, key_size = read_keys.shape
        read_tiles = config.n_read_tiles
        key_tile_size = key_size // read_tiles
        value_size = values.shape[-1]
        value_tile_size = value_size // read_tiles
        read_keys_tiled = read_keys.astype(jnp.float32).reshape(
            batch, n_slots, read_tiles, key_tile_size
        )
        q_read_tiled = q_read_t.reshape(batch, read_tiles, key_tile_size)
        read_keys_normalized = unit_norm(read_keys_tiled, eps=config.eps)
        q_read_normalized = q_read_tiled
        values_tiled = values.astype(jnp.float32).reshape(
            batch, n_slots, read_tiles, value_tile_size
        )
        values_normalized = values_tiled
        if config.use_value_unit_norm:
            values_normalized = unit_norm(
                values_normalized, eps=config.eps
            )
        read_similarity = jnp.einsum(
            "brk,bmrk->brm", q_read_normalized, read_keys_normalized
        )
        tau_read_tiled = jnp.reshape(tau_read, (read_tiles, 1))
        if config.use_leaky_warmup:
            hard = trim_square(
                read_similarity, tau_read_tiled, eps=config.eps
            )
            soft = jax.nn.sigmoid(
                config.leaky_gamma * (read_similarity - tau_read_tiled)
            )
            read_relevance = (
                (1.0 - config.leaky_alpha) * hard
                + config.leaky_alpha * soft
            )
        else:
            read_relevance = trim_square(
                read_similarity, tau_read_tiled, eps=config.eps
            )
        if config.use_age_mask:
            age_scores = (
                (config.age_ref - ages) / (config.age_sigma + config.eps)
            )
            read_relevance *= jax.nn.sigmoid(age_scores)[:, None, :]

        z_tiled = jnp.einsum(
            "brm,bmrv->brv", read_relevance, values_normalized
        )
        u_tiled = tanh_norm(
            z_tiled, cap=config.tanh_norm_cap, eps=config.eps
        )
        z = z_tiled.reshape(batch, value_size)
        u = u_tiled.reshape(batch, value_size)
        read_activity = jnp.max(read_relevance, axis=1)

        write_mode = _resolved_write_mode(config)
        matched_route = jnp.zeros_like(read_activity)
        novel_route = jnp.zeros_like(read_activity)
        effective_admission = jnp.zeros_like(admission_t)
        is_novel = jnp.zeros_like(admission_t, dtype=jnp.bool_)
        if write_mode in ("legacy_threshold", "competitive_novel"):
            write_keys_normalized = unit_norm(
                write_keys.astype(jnp.float32), eps=config.eps
            )
            write_similarity = jnp.einsum(
                "bk,bmk->bm", q_write_t, write_keys_normalized
            )
            write_relevance = trim_square(
                write_similarity, tau_write, eps=config.eps
            )
        else:
            write_relevance = jnp.zeros_like(read_activity)

        if write_mode == "legacy_threshold":
            effective_write_relevance = jnp.maximum(
                write_relevance, config.write_rel_floor
            )
            strength = mu[None, :] * effective_write_relevance
            next_ages = jnp.where(
                write_relevance > 1e-3, 0.0, ages + 1.0
            )
            matched_route = effective_write_relevance
        elif write_mode == "competitive_novel":
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
                admission_t,
                bank_logits_t,
                config.bank_ids or (0,) * n_slots,
                route_power=config.route_power,
                novelty_threshold=config.novelty_threshold,
                allocation_temperature=config.allocation_temperature,
                bank_route_temperature=config.bank_route_temperature,
                allocation_age_weight=config.allocation_age_weight,
                allocation_usage_weight=config.allocation_usage_weight,
                hard_admission=config.hard_admission,
                admission_threshold=config.admission_threshold,
                eps=config.eps,
            )
            matched_route = jnp.where(
                is_novel[:, None], 0.0, raw_matched_route
            )
            strength = mu[None, :] * effective_write_relevance
            next_ages = jnp.where(
                effective_write_relevance > config.eps,
                0.0,
                ages + 1.0,
            )
        elif write_mode == "disabled":
            effective_write_relevance = jnp.zeros_like(read_activity)
            strength = jnp.zeros_like(read_activity)
            next_ages = ages + 1.0
        else:
            effective_write_relevance = write_relevance
            strength = jnp.broadcast_to(mu[None, :], read_activity.shape)
            next_ages = ages

        strength_vector = strength[:, :, None]
        next_slots = slots + strength_vector * (
            delta_slots_t.astype(jnp.float32) - slots
        )
        next_read_keys = read_keys + strength_vector * (
            delta_read_keys_t.astype(jnp.float32) - read_keys
        )
        next_values = values + strength_vector * (
            delta_values_t.astype(jnp.float32) - values
        )
        next_write_keys = write_keys + strength_vector * (
            delta_write_keys_t.astype(jnp.float32) - write_keys
        )
        update = next_slots - slots
        update_squared = jnp.sum(update * update, axis=-1)

        if write_mode == "legacy_threshold":
            activity = jnp.maximum(read_activity, write_relevance)
        else:
            activity = read_activity
        next_usage = config.usage_ema_decay * usage + (
            1.0 - config.usage_ema_decay
        ) * activity
        route_mass = jnp.sum(effective_write_relevance, axis=-1)
        route_distribution = effective_write_relevance / (
            route_mass[:, None] + config.eps
        )
        route_entropy = -jnp.sum(
            route_distribution * jnp.log(route_distribution + config.eps),
            axis=-1,
        )
        route_top1 = jnp.max(route_distribution, axis=-1)
        bank_ids = jnp.asarray(
            config.bank_ids or (0,) * n_slots,
            dtype=jnp.int32,
        )
        bank_write_mass = tuple(
            jnp.sum(
                effective_write_relevance
                * (bank_ids == bank_id)[None, :],
                axis=-1,
            )
            for bank_id in range(3)
        )
        novel_mass = jnp.sum(novel_route, axis=-1)
        eviction_age = jnp.sum(novel_route * ages, axis=-1) / (
            novel_mass + config.eps
        )
        eviction_usage = jnp.sum(novel_route * usage, axis=-1) / (
            novel_mass + config.eps
        )
        step_statistics = jnp.stack(
            (
                jnp.mean(read_relevance, axis=(-2, -1)),
                jnp.max(read_relevance, axis=(-2, -1)),
                jnp.sum(read_activity > 1e-3, axis=-1).astype(jnp.float32),
                jnp.linalg.norm(z, axis=-1),
                jnp.linalg.norm(u, axis=-1),
                jnp.mean(write_relevance, axis=-1),
                jnp.mean(effective_write_relevance, axis=-1),
                jnp.mean(next_usage, axis=-1),
                jnp.sum(matched_route, axis=-1),
                novel_mass,
                route_entropy,
                route_top1,
                effective_admission,
                is_novel.astype(jnp.float32),
                (route_mass <= config.eps).astype(jnp.float32),
                *bank_write_mass,
                eviction_age,
                eviction_usage,
                (
                    (effective_admission < 0.05)
                    & (write_mode == "competitive_novel")
                ).astype(jnp.float32),
                (
                    (effective_admission > 0.95)
                    & (write_mode == "competitive_novel")
                ).astype(jnp.float32),
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
        )
        return next_carry, (u, step_statistics, update_squared)

    initial_carry = (
        initial_slots,
        initial_read_keys.astype(jnp.float32),
        initial_values.astype(jnp.float32),
        initial_write_keys.astype(jnp.float32),
        initial_ages,
        initial_usage,
    )
    inputs = (
        q_read,
        q_write,
        admission,
        bank_logits,
        delta_slots,
        delta_read_keys,
        delta_values,
        delta_write_keys,
    )
    final_carry, (u, statistics, update_squared) = jax.lax.scan(
        step, initial_carry, inputs
    )
    final_slots, _, _, _, final_ages, final_usage = final_carry
    return (
        u.astype(delta_values.dtype),
        final_slots,
        final_ages,
        final_usage,
        jax.lax.stop_gradient(statistics),
        jax.lax.stop_gradient(update_squared),
    )


def screening_recurrence_reference(
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
    config: ScreeningRecurrenceConfig,
):
    """Execute the projected screening recurrence with portable JAX."""

    inputs = (
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
    )
    _validate_screening_recurrence_inputs(*inputs, config=config)
    return _screening_recurrence_reference_impl(*inputs, config=config)


def _screening_forward_dispatch(
    *inputs,
    config: ScreeningRecurrenceConfig,
    backend: str | None,
    interpret: bool,
    with_aux: bool,
):
    selected = resolve_screening_backend(backend)
    if selected == "reference":
        outputs = _screening_recurrence_reference_impl(
            *inputs, config=config
        )
        return (outputs, ()) if with_aux else outputs
    if selected in ("pallas_gpu_mosaic", "pallas_gpu_triton"):
        from rwkv7m.kernels.screening_pallas_gpu import (
            screening_pallas_gpu_forward_with_aux,
            screening_pallas_gpu_forward,
        )

        lowering = (
            "mosaic" if selected == "pallas_gpu_mosaic" else "triton"
        )
        function = (
            screening_pallas_gpu_forward_with_aux
            if with_aux
            else screening_pallas_gpu_forward
        )
        return function(
            *inputs,
            config=config,
            lowering=lowering,
            interpret=interpret,
        )
    from rwkv7m.kernels.screening_pallas_tpu import (
        screening_pallas_tpu_forward_with_aux,
        screening_pallas_tpu_forward,
    )

    function = (
        screening_pallas_tpu_forward_with_aux
        if with_aux
        else screening_pallas_tpu_forward
    )
    return function(
        *inputs, config=config, interpret=interpret
    )


def _screening_backward_dispatch(
    inputs,
    cotangents,
    aux,
    *,
    config: ScreeningRecurrenceConfig,
    backend: str | None,
    interpret: bool,
):
    selected = resolve_screening_backend(backend)
    if selected == "reference":
        _, pullback = jax.vjp(
            lambda *values: _screening_recurrence_reference_impl(
                *values, config=config
            ),
            *inputs,
        )
        return pullback(cotangents)
    u_cotangent, slots_cotangent, ages_cotangent, usage_cotangent, _, _ = (
        cotangents
    )
    if selected in ("pallas_gpu_mosaic", "pallas_gpu_triton"):
        from rwkv7m.kernels.screening_pallas_gpu import (
            screening_pallas_gpu_backward,
        )

        lowering = (
            "mosaic" if selected == "pallas_gpu_mosaic" else "triton"
        )
        return screening_pallas_gpu_backward(
            *inputs,
            u_cotangent,
            slots_cotangent,
            ages_cotangent,
            usage_cotangent,
            *aux,
            config=config,
            lowering=lowering,
            interpret=interpret,
        )
    from rwkv7m.kernels.screening_pallas_tpu import (
        screening_pallas_tpu_backward,
    )

    return screening_pallas_tpu_backward(
        *inputs,
        u_cotangent,
        slots_cotangent,
        ages_cotangent,
        usage_cotangent,
        *aux,
        config=config,
        interpret=interpret,
    )


@partial(jax.custom_vjp, nondiff_argnums=(17, 18, 19))
def screening_recurrence(
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
    config: ScreeningRecurrenceConfig,
    backend: str | None = None,
    interpret: bool = False,
):
    """Run the persistent screening recurrence.

    Accelerator execution uses backend-specific persistent Pallas forward and
    reverse-time kernels. The portable reference remains the CPU and explicit
    fallback implementation.
    """

    inputs = (
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
    )
    _validate_screening_recurrence_inputs(*inputs, config=config)
    return _screening_forward_dispatch(
        *inputs,
        config=config,
        backend=backend,
        interpret=interpret,
        with_aux=False,
    )


def _screening_recurrence_fwd(
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
    config,
    backend,
    interpret,
):
    inputs = (
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
    )
    _validate_screening_recurrence_inputs(*inputs, config=config)
    outputs, aux = _screening_forward_dispatch(
        *inputs,
        config=config,
        backend=backend,
        interpret=interpret,
        with_aux=True,
    )
    return outputs, (inputs, aux)


def _screening_recurrence_bwd(
    config,
    backend,
    interpret,
    residuals,
    cotangents,
):
    inputs, aux = residuals
    return _screening_backward_dispatch(
        inputs,
        cotangents,
        aux,
        config=config,
        backend=backend,
        interpret=interpret,
    )


screening_recurrence.defvjp(
    _screening_recurrence_fwd,
    _screening_recurrence_bwd,
)


def screening_recurrence_sharded(
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
    config: ScreeningRecurrenceConfig,
    *,
    mesh,
    data_axis: str = "data",
    model_axis: str = "model",
):
    """Run data/model-sharded Pallas screening with explicit collectives.

    The slot feature shard is gathered once at the recurrence boundary. Each
    model device then runs the same persistent recurrence, replicated outputs
    are averaged to give their transpose one logical contribution, and the
    final slot state is sliced back to its owning model shard. This avoids
    collectives in the sequential time loop and gives the custom VJP an
    explicit, differentiable model-axis contract.
    """

    time_data = jax.sharding.PartitionSpec(None, data_axis, None)
    time_batch = jax.sharding.PartitionSpec(None, data_axis)
    time_data_model = jax.sharding.PartitionSpec(
        None, data_axis, None, model_axis
    )
    time_data_replicated = jax.sharding.PartitionSpec(
        None, data_axis, None, None
    )
    data_slots = jax.sharding.PartitionSpec(data_axis, None, model_axis)
    data_vectors = jax.sharding.PartitionSpec(data_axis, None, None)
    data_scalars = jax.sharding.PartitionSpec(data_axis, None)
    replicated_vector = jax.sharding.PartitionSpec(None)
    replicated_scalar = jax.sharding.PartitionSpec()

    def mapped(*local_inputs):
        local_delta_slots = local_inputs[4]
        local_initial_slots = local_inputs[8]
        full_delta_slots = jax.lax.all_gather(
            local_delta_slots, model_axis, axis=3, tiled=True
        )
        full_initial_slots = jax.lax.all_gather(
            local_initial_slots, model_axis, axis=2, tiled=True
        )
        recurrence_inputs = (
            *local_inputs[:4],
            full_delta_slots,
            *local_inputs[5:8],
            full_initial_slots,
            *local_inputs[9:],
        )
        outputs = screening_recurrence(
            *recurrence_inputs,
            config,
        )
        model_size = mesh.shape[model_axis]
        slot_feature_size = local_initial_slots.shape[-1]
        slot_feature_start = (
            jax.lax.axis_index(model_axis) * slot_feature_size
        )
        final_slots = jax.lax.dynamic_slice_in_dim(
            outputs[1],
            slot_feature_start,
            slot_feature_size,
            axis=2,
        )

        def replicated(value):
            return jax.lax.psum(value, model_axis) / model_size

        return (
            replicated(outputs[0]),
            final_slots,
            replicated(outputs[2]),
            replicated(outputs[3]),
            replicated(outputs[4]),
            replicated(outputs[5]),
        )

    input_specs = (
        time_data,
        time_data,
        time_batch,
        time_data,
        time_data_model,
        time_data_replicated,
        time_data_replicated,
        time_data_replicated,
        data_slots,
        data_vectors,
        data_vectors,
        data_vectors,
        data_scalars,
        data_scalars,
        replicated_vector,
        replicated_scalar if config.n_read_tiles == 1 else replicated_vector,
        replicated_scalar,
    )
    mapped_recurrence = jax.shard_map(
        mapped,
        mesh=mesh,
        in_specs=input_specs,
        out_specs=(
            time_data,
            data_slots,
            data_scalars,
            data_scalars,
            time_data,
            jax.sharding.PartitionSpec(None, data_axis, None),
        ),
        check_vma=False,
    )
    inputs = (
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
    )
    placed_inputs = tuple(
        jax.reshard(value, jax.NamedSharding(mesh, spec))
        for value, spec in zip(inputs, input_specs, strict=True)
    )
    return mapped_recurrence(*placed_inputs)


__all__ = [
    "ACTIVE_SLOTS",
    "ADMISSION_HIGH_RATE",
    "ADMISSION_LOW_RATE",
    "ADMISSION_MEAN",
    "BANK_LONG_WRITE_MASS",
    "BANK_MID_WRITE_MASS",
    "BANK_SHORT_WRITE_MASS",
    "EVICTION_AGE_MEAN",
    "EVICTION_USAGE_MEAN",
    "MATCHED_ROUTE_MASS",
    "NOVEL_RATE",
    "NOVEL_ROUTE_MASS",
    "READ_MAX",
    "READ_MEAN",
    "REJECTED_RATE",
    "ROUTE_ENTROPY",
    "ROUTE_TOP1",
    "SCREENING_STEP_STAT_COUNT",
    "ScreeningRecurrenceConfig",
    "U_NORM",
    "USAGE_MEAN",
    "WRITE_EFFECTIVE_MEAN",
    "WRITE_MEAN",
    "Z_NORM",
    "screening_recurrence",
    "screening_recurrence_reference",
    "screening_recurrence_sharded",
]
