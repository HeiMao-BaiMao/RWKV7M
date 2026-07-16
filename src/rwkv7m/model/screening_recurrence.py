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

from .screening import tanh_norm, trim_square, unit_norm


READ_MEAN = 0
READ_MAX = 1
ACTIVE_SLOTS = 2
Z_NORM = 3
U_NORM = 4
WRITE_MEAN = 5
WRITE_EFFECTIVE_MEAN = 6
USAGE_MEAN = 7
SCREENING_STEP_STAT_COUNT = 8

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


def _validate_screening_recurrence_inputs(
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
):
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
    for name, value in (("tau_read", tau_read), ("tau_write", tau_write)):
        if value.shape != () or value.dtype != jnp.float32:
            raise TypeError(f"{name} must be a float32 scalar")


def _screening_recurrence_reference_impl(
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
    config: ScreeningRecurrenceConfig,
):
    def step(carry, inputs):
        slots, read_keys, values, write_keys, ages, usage = carry
        (
            q_read_t,
            q_write_t,
            delta_slots_t,
            delta_read_keys_t,
            delta_values_t,
            delta_write_keys_t,
        ) = inputs

        read_keys_normalized = unit_norm(
            read_keys.astype(jnp.float32), eps=config.eps
        )
        values_normalized = values.astype(jnp.float32)
        if config.use_value_unit_norm:
            values_normalized = unit_norm(
                values_normalized, eps=config.eps
            )
        read_similarity = jnp.einsum(
            "bk,bmk->bm", q_read_t, read_keys_normalized
        )
        if config.use_leaky_warmup:
            hard = trim_square(read_similarity, tau_read, eps=config.eps)
            soft = jax.nn.sigmoid(
                config.leaky_gamma * (read_similarity - tau_read)
            )
            read_relevance = (
                (1.0 - config.leaky_alpha) * hard
                + config.leaky_alpha * soft
            )
        else:
            read_relevance = trim_square(
                read_similarity, tau_read, eps=config.eps
            )
        if config.use_age_mask:
            age_scores = (
                (config.age_ref - ages) / (config.age_sigma + config.eps)
            )
            read_relevance *= jax.nn.sigmoid(age_scores)

        z = jnp.einsum("bm,bmv->bv", read_relevance, values_normalized)
        u = tanh_norm(z, cap=config.tanh_norm_cap, eps=config.eps)

        if config.write_enabled:
            write_keys_normalized = unit_norm(
                write_keys.astype(jnp.float32), eps=config.eps
            )
            write_similarity = jnp.einsum(
                "bk,bmk->bm", q_write_t, write_keys_normalized
            )
            write_relevance = trim_square(
                write_similarity, tau_write, eps=config.eps
            )
            effective_write_relevance = jnp.maximum(
                write_relevance, config.write_rel_floor
            )
            strength = mu[None, :] * effective_write_relevance
            next_ages = jnp.where(
                write_relevance > 1e-3, 0.0, ages + 1.0
            )
        else:
            write_relevance = jnp.zeros_like(read_relevance)
            effective_write_relevance = write_relevance
            strength = jnp.broadcast_to(mu[None, :], read_relevance.shape)
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

        activity = jnp.maximum(
            read_relevance,
            write_relevance if config.write_enabled else read_relevance,
        )
        next_usage = config.usage_ema_decay * usage + (
            1.0 - config.usage_ema_decay
        ) * activity
        step_statistics = jnp.stack(
            (
                jnp.mean(read_relevance, axis=-1),
                jnp.max(read_relevance, axis=-1),
                jnp.sum(read_relevance > 1e-3, axis=-1).astype(jnp.float32),
                jnp.linalg.norm(z, axis=-1),
                jnp.linalg.norm(u, axis=-1),
                jnp.mean(write_relevance, axis=-1),
                jnp.mean(effective_write_relevance, axis=-1),
                jnp.mean(next_usage, axis=-1),
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
    _validate_screening_recurrence_inputs(*inputs)
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


@partial(jax.custom_vjp, nondiff_argnums=(15, 16, 17))
def screening_recurrence(
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
    _validate_screening_recurrence_inputs(*inputs)
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
    _validate_screening_recurrence_inputs(*inputs)
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
        local_delta_slots = local_inputs[2]
        local_initial_slots = local_inputs[6]
        full_delta_slots = jax.lax.all_gather(
            local_delta_slots, model_axis, axis=3, tiled=True
        )
        full_initial_slots = jax.lax.all_gather(
            local_initial_slots, model_axis, axis=2, tiled=True
        )
        recurrence_inputs = (
            *local_inputs[:2],
            full_delta_slots,
            *local_inputs[3:6],
            full_initial_slots,
            *local_inputs[7:],
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
        replicated_scalar,
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
    "READ_MAX",
    "READ_MEAN",
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
