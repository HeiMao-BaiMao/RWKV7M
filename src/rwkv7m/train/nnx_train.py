"""NNX-native optimizer lifecycle and training step."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from ..model.nnx_model import (
    NNXScreenedRWKVModel,
    NNXShardingConfig,
    initialize_nnx_model,
)
from ..model.nnx_conversion import nnx_params_to_linen
from .train_state import create_optimizer


def _optimizer_config(config):
    return {
        "lr_init": getattr(config, "lr_init", 1e-3),
        "lr_final": getattr(config, "lr_final", 1e-5),
        "warmup_steps": getattr(config, "warmup_steps", 10),
        "lr_schedule": getattr(config, "lr_schedule", "optax_cosine"),
        "max_grad_norm": getattr(config, "max_grad_norm", 1.0),
        "weight_decay": getattr(config, "weight_decay", 0.001),
        "adam_beta1": getattr(config, "adam_beta1", 0.9),
        "adam_beta2": getattr(config, "adam_beta2", 0.999),
        "adam_eps": getattr(config, "adam_eps", 1e-8),
        "optimizer_state_dtype": getattr(
            config, "optimizer_state_dtype", "float32"
        ),
        "optimizer_backend": getattr(config, "optimizer_backend", "optax"),
        "screening_lr_multiplier": getattr(
            getattr(config, "screening", None),
            "optimizer_lr_multiplier",
            1.0,
        ),
        "screening_activation_step": getattr(
            getattr(config, "screening", None),
            "activation_step",
            0,
        ),
        "screening_activation_warmup_steps": getattr(
            getattr(config, "screening", None),
            "activation_warmup_steps",
            0,
        ),
    }


@dataclass
class NNXTrainState:
    """Thin compatibility handle around an NNX model and NNX optimizer."""

    model: NNXScreenedRWKVModel
    optimizer: nnx.Optimizer

    @property
    def step(self):
        return self.optimizer.step[...]

    @property
    def params(self):
        # Preserve the established portable/checkpoint boundary while the
        # optimizer itself remains fully NNX-native.
        return nnx_params_to_linen(self.model)

    @property
    def nnx_params(self):
        return nnx.state(self.model, nnx.Param)

    @property
    def opt_state(self):
        return self.optimizer.opt_state

    @property
    def tx(self):
        return self.optimizer.tx

    def ready_state(self):
        """Return every updated array leaf for a measurement-window barrier."""

        return nnx.state((self.model, self.optimizer))


def create_nnx_train_state(
    model: NNXScreenedRWKVModel,
    config,
    *,
    total_steps: int = 10000,
) -> NNXTrainState:
    tx = create_optimizer(_optimizer_config(config), total_steps=total_steps)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    return NNXTrainState(model=model, optimizer=optimizer)


def initialize_nnx_train_state(
    rng_key,
    config,
    *,
    total_steps: int = 10000,
    sharding: NNXShardingConfig | None = None,
) -> NNXTrainState:
    """Create model and Optax state without an unsharded materialization."""

    if sharding is None:
        model = initialize_nnx_model(rng_key, config)
        return create_nnx_train_state(model, config, total_steps=total_steps)
    tx = create_optimizer(_optimizer_config(config), total_steps=total_steps)

    @jax.jit
    def initialize(key):
        model = NNXScreenedRWKVModel(
            config,
            rngs=nnx.Rngs(params=key),
            sharding=sharding,
        )
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        return model, optimizer

    with jax.set_mesh(sharding.mesh):
        model, optimizer = initialize(rng_key)
    return NNXTrainState(model=model, optimizer=optimizer)


def _dtype_from_name(name):
    if name == "float32":
        return jnp.float32
    if name == "bfloat16":
        return jnp.bfloat16
    raise ValueError(f"unsupported training dtype: {name!r}")


def _slice_batch_tree(tree, start, stop):
    return jax.tree.map(lambda value: value[start:stop], tree)


def _concat_batch_trees(trees):
    if len(trees) == 1:
        return trees[0]
    return jax.tree.map(lambda *values: jnp.concatenate(values, axis=0), *trees)


def _cast_gradient_tree(grads, dtype):
    return jax.tree.map(
        lambda value: value.astype(dtype)
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.inexact)
        else value,
        grads,
    )


def _split_sequence_chunks(value, chunk_count, chunk_size):
    batch_size = int(value.shape[0])
    return jnp.swapaxes(
        value.reshape(
            (batch_size, chunk_count, chunk_size, *value.shape[2:])
        ),
        0,
        1,
    )


def _merge_hidden_chunks(model, chunked_hidden):
    """Restore [chunks, batch, tokens, hidden] to the activation contract."""
    chunk_count, batch_size, chunk_size, hidden_size = chunked_hidden.shape
    transposed = jnp.swapaxes(chunked_hidden, 0, 1)
    shape = (batch_size, chunk_count * chunk_size, hidden_size)
    sharding = model.sharding
    if sharding is not None and sharding.uses_explicit_axes:
        return jax.lax.reshape(
            transposed,
            shape,
            out_sharding=sharding.activation(3, model_sharded=True),
        )
    return jnp.reshape(transposed, shape)


def compute_v5_admission_floor_loss(
    write_rate,
    slot_utilization,
    screening_config,
    training_step,
):
    """Return the temporary empty-capacity admission floor and its target."""

    if (
        training_step is None
        or getattr(screening_config, "semantics_version", None)
        not in {"screening-v5-core", "screening-v5-retention"}
        or screening_config.admission_floor_weight <= 0.0
        or screening_config.admission_floor_target_initial <= 0.0
        or screening_config.admission_floor_steps <= 0
    ):
        zero = jnp.zeros((), dtype=jnp.float32)
        return zero, zero
    curriculum_step = jnp.maximum(
        jnp.asarray(training_step, dtype=jnp.float32)
        - float(screening_config.activation_step),
        0.0,
    )
    anneal = jnp.clip(
        1.0
        - curriculum_step / float(screening_config.admission_floor_steps),
        0.0,
        1.0,
    )
    empty_fraction = jax.lax.stop_gradient(
        1.0 - jnp.asarray(slot_utilization, dtype=jnp.float32)
    )
    target = (
        screening_config.admission_floor_target_initial
        * anneal
        * jnp.clip(empty_fraction, 0.0, 1.0)
    )
    loss = screening_config.admission_floor_weight * jnp.square(
        jax.nn.relu(target - jnp.asarray(write_rate, dtype=jnp.float32))
    )
    return loss, target


def _tree_finite_flag(tree):
    leaves = [
        value
        for value in jax.tree.leaves(tree)
        if hasattr(value, "dtype")
        and jnp.issubdtype(value.dtype, jnp.inexact)
    ]
    if not leaves:
        return jnp.ones((), dtype=jnp.float32)
    return jnp.stack(
        [jnp.all(jnp.isfinite(value)) for value in leaves]
    ).all().astype(jnp.float32)


def _tree_gradient_statistics(tree):
    leaves = [
        value.astype(jnp.float32)
        for value in jax.tree.leaves(tree)
        if hasattr(value, "dtype")
        and jnp.issubdtype(value.dtype, jnp.inexact)
    ]
    if not leaves:
        zero = jnp.zeros((), dtype=jnp.float32)
        return jnp.ones((), dtype=jnp.float32), zero, zero
    all_finite = jnp.stack(
        [jnp.all(jnp.isfinite(value)) for value in leaves]
    ).all()
    squared_norm = sum(
        (jnp.sum(jnp.square(value)) for value in leaves),
        start=jnp.zeros((), dtype=jnp.float32),
    )
    max_abs = jnp.max(
        jnp.stack([jnp.max(jnp.abs(value)) for value in leaves])
    )
    return (
        all_finite.astype(jnp.float32),
        jnp.sqrt(squared_norm),
        max_abs,
    )


def compute_v5_write_budget_loss(
    write_rate,
    screening_config,
    slot_utilization=None,
):
    """Return the v5 upper write-budget penalty and configured ceiling."""

    if (
        getattr(screening_config, "semantics_version", None)
        not in {"screening-v5-core", "screening-v5-retention"}
        or screening_config.write_budget_weight <= 0.0
    ):
        zero = jnp.zeros((), dtype=jnp.float32)
        return zero, zero
    target = jnp.asarray(
        screening_config.write_budget_target_max,
        dtype=jnp.float32,
    )
    excess = jax.nn.relu(
        jnp.asarray(write_rate, dtype=jnp.float32) - target
    )
    enabled = jnp.ones((), dtype=jnp.float32)
    minimum_utilization = float(
        screening_config.write_budget_min_slot_utilization
    )
    if slot_utilization is not None and minimum_utilization > 0.0:
        enabled = jax.lax.stop_gradient(
            (
                jnp.asarray(slot_utilization, dtype=jnp.float32)
                >= minimum_utilization
            ).astype(jnp.float32)
        )
    return (
        screening_config.write_budget_weight
        * enabled
        * jnp.square(excess),
        target,
    )


def compute_v5_self_index_loss(
    raw_loss,
    screening_config,
    training_step,
):
    """Apply the temporary v5 self-index curriculum to its raw route loss."""

    if (
        training_step is None
        or getattr(screening_config, "semantics_version", None)
        not in {"screening-v5-core", "screening-v5-retention"}
        or screening_config.self_index_loss_weight <= 0.0
        or screening_config.self_index_loss_steps <= 0
    ):
        zero = jnp.zeros((), dtype=jnp.float32)
        return zero, zero
    curriculum_step = jnp.maximum(
        jnp.asarray(training_step, dtype=jnp.float32)
        - float(screening_config.activation_step),
        0.0,
    )
    anneal = jnp.clip(
        1.0
        - curriculum_step / float(screening_config.self_index_loss_steps),
        0.0,
        1.0,
    )
    coefficient = screening_config.self_index_loss_weight * anneal
    return coefficient * jnp.asarray(raw_loss, dtype=jnp.float32), coefficient


def nnx_model_loss(
    active_model,
    batch,
    rwkv_state,
    screen_state,
    *,
    phase,
    deterministic,
    include_l2wrap,
    ce_loss_scale=1.0,
    l2_loss_scale=1.0,
    aux_loss_scale=1.0,
    training_step=None,
):
    """Run recurrent and LM-head chunks independently without truncated BPTT."""

    token_count = int(batch["input_ids"].shape[1])
    recurrent_chunk_size = (
        active_model.config.sequence_chunk_size or token_count
    )
    recurrent_chunk_size = min(int(recurrent_chunk_size), token_count)
    head_chunk_size = (
        active_model.config.head_chunk_size
        or active_model.config.sequence_chunk_size
        or token_count
    )
    head_chunk_size = min(int(head_chunk_size), token_count)
    ce_total = jnp.zeros((), dtype=jnp.float32)
    ce_count = jnp.zeros((), dtype=jnp.float32)
    l2_total = jnp.zeros((), dtype=jnp.float32)
    l2_count = jnp.zeros((), dtype=jnp.float32)
    stats_total = None
    current_rwkv = rwkv_state
    current_screen = screen_state
    target_sharding = None
    if active_model.config.vocab_parallel and active_model.sharding is not None:
        target_sharding = active_model.sharding.activation(
            2,
            model_sharded=False,
        )

    def apply_recurrent_chunk(model, carry, chunk_input_ids):
        chunk_rwkv, chunk_screen = carry
        hidden, new_rwkv, new_screen, stats = model.compute_recurrent_hidden(
            chunk_input_ids,
            chunk_rwkv,
            chunk_screen,
            phase=phase,
            deterministic=deterministic,
            training_step=training_step,
        )
        return (new_rwkv, new_screen), {"hidden": hidden, "stats": stats}

    recurrent_chunk_count, recurrent_remainder = divmod(
        token_count, recurrent_chunk_size
    )
    if recurrent_chunk_count > 1 and recurrent_remainder == 0:
        chunked_input_ids = _split_sequence_chunks(
            batch["input_ids"],
            recurrent_chunk_count,
            recurrent_chunk_size,
        )

        @nnx.scan(
            in_axes=(nnx.Carry, 0, None),
            out_axes=(nnx.Carry, 0),
        )
        def scan_recurrent_chunks(carry, chunk_input_ids, model):
            return apply_recurrent_chunk(model, carry, chunk_input_ids)

        (current_rwkv, current_screen), recurrent_outputs = (
            scan_recurrent_chunks(
                (current_rwkv, current_screen),
                chunked_input_ids,
                active_model,
            )
        )
        hidden = _merge_hidden_chunks(
            active_model,
            recurrent_outputs["hidden"],
        )
        stats_total = jax.tree.map(
            lambda value: jnp.sum(value, axis=0) * recurrent_chunk_size,
            recurrent_outputs["stats"],
        )
    else:
        hidden_chunks = []
        # Avoid padding a final short chunk because it would mutate the
        # returned recurrent state and change exact chunked-BPTT semantics.
        for start in range(0, token_count, recurrent_chunk_size):
            stop = min(start + recurrent_chunk_size, token_count)
            (current_rwkv, current_screen), recurrent_outputs = (
                apply_recurrent_chunk(
                    active_model,
                    (current_rwkv, current_screen),
                    batch["input_ids"][:, start:stop],
                )
            )
            hidden_chunks.append(recurrent_outputs["hidden"])
            chunk_tokens = jnp.asarray(stop - start, dtype=jnp.float32)
            stats = recurrent_outputs["stats"]
            if stats_total is None:
                stats_total = {
                    key: value * chunk_tokens for key, value in stats.items()
                }
            else:
                stats_total = {
                    key: stats_total[key] + value * chunk_tokens
                    for key, value in stats.items()
                }
        hidden = (
            hidden_chunks[0]
            if len(hidden_chunks) == 1
            else jnp.concatenate(hidden_chunks, axis=1)
        )

    def apply_head_chunk(model, chunk_hidden, chunk_batch):
        return model.compute_training_loss(
            chunk_hidden,
            chunk_batch["target_ids"],
            chunk_batch.get("mask"),
            target_sharding=target_sharding,
        )

    loss_batch = {"target_ids": batch["target_ids"]}
    if "mask" in batch:
        loss_batch["mask"] = batch["mask"]
    head_chunk_count, head_remainder = divmod(token_count, head_chunk_size)
    if head_chunk_count > 1 and head_remainder == 0:
        chunked_hidden = _split_sequence_chunks(
            hidden,
            head_chunk_count,
            head_chunk_size,
        )
        chunked_loss_batch = jax.tree.map(
            lambda value: _split_sequence_chunks(
                value,
                head_chunk_count,
                head_chunk_size,
            ),
            loss_batch,
        )

        @nnx.scan(in_axes=(0, 0, None), out_axes=0)
        def scan_head_chunks(chunk_hidden, chunk_batch, model):
            return apply_head_chunk(model, chunk_hidden, chunk_batch)

        head_outputs = scan_head_chunks(
            chunked_hidden,
            chunked_loss_batch,
            active_model,
        )
        ce_total = jnp.sum(head_outputs["ce_total"], axis=0)
        ce_count = jnp.sum(head_outputs["ce_count"], axis=0)
        l2_total = jnp.sum(head_outputs["l2_total"], axis=0)
        l2_count = jnp.sum(head_outputs["l2_count"], axis=0)
    else:
        for start in range(0, token_count, head_chunk_size):
            stop = min(start + head_chunk_size, token_count)
            chunk_batch = {
                key: value[:, start:stop] for key, value in loss_batch.items()
            }
            head_outputs = apply_head_chunk(
                active_model,
                hidden[:, start:stop],
                chunk_batch,
            )
            ce_total += head_outputs["ce_total"]
            ce_count += head_outputs["ce_count"]
            l2_total += head_outputs["l2_total"]
            l2_count += head_outputs["l2_count"]

    ce_loss = ce_total / jnp.maximum(ce_count, 1.0)
    l2_loss = l2_total / jnp.maximum(l2_count, 1.0)
    total_loss = ce_loss * ce_loss_scale
    if include_l2wrap:
        total_loss += l2_loss * l2_loss_scale
    stats_total = {} if stats_total is None else stats_total
    normalized_stats = {
        key: value / jnp.asarray(token_count, dtype=jnp.float32)
        for key, value in stats_total.items()
    }
    admission_floor_loss = jnp.zeros((), dtype=jnp.float32)
    admission_floor_target = jnp.zeros((), dtype=jnp.float32)
    write_budget_loss = jnp.zeros((), dtype=jnp.float32)
    write_budget_target = jnp.zeros((), dtype=jnp.float32)
    self_index_loss = jnp.zeros((), dtype=jnp.float32)
    self_index_weight = jnp.zeros((), dtype=jnp.float32)
    per_layer_aux_metrics = {}
    if phase == "read_write" and "write_budget_rate" in normalized_stats:
        cfg = active_model.config.screening
        layer_inputs = []
        for layer_idx in cfg.screened_layers:
            prefix = f"screening_layer_{layer_idx}_"
            realized_key = prefix + "write_budget_realized_rate"
            if realized_key in normalized_stats:
                layer_inputs.append(
                    (
                        layer_idx,
                        normalized_stats[realized_key],
                        normalized_stats[prefix + "slot_utilization"],
                        normalized_stats.get(
                            prefix + "self_index_raw_loss",
                            jnp.zeros((), dtype=jnp.float32),
                        ),
                    )
                )
        if not layer_inputs:
            layer_inputs.append(
                (
                    None,
                    normalized_stats.get(
                        "write_budget_realized_rate",
                        normalized_stats["write_budget_rate"],
                    ),
                    normalized_stats["slot_utilization"],
                    normalized_stats.get(
                        "self_index_raw_loss",
                        jnp.zeros((), dtype=jnp.float32),
                    ),
                )
            )

        floor_losses = []
        floor_targets = []
        budget_losses = []
        budget_targets = []
        self_losses = []
        self_weights = []
        activation = normalized_stats.get(
            "screening_activation",
            jnp.ones((), dtype=jnp.float32),
        )
        for layer_idx, realized_rate, utilization, raw_self_loss in layer_inputs:
            layer_floor_loss, layer_floor_target = (
                compute_v5_admission_floor_loss(
                    realized_rate,
                    utilization,
                    cfg,
                    training_step,
                )
            )
            layer_budget_loss, layer_budget_target = (
                compute_v5_write_budget_loss(
                    realized_rate,
                    cfg,
                    slot_utilization=utilization,
                )
            )
            layer_self_loss, layer_self_weight = compute_v5_self_index_loss(
                raw_self_loss,
                cfg,
                training_step,
            )
            layer_floor_loss *= activation
            layer_budget_loss *= activation
            layer_self_loss *= activation
            floor_losses.append(layer_floor_loss)
            floor_targets.append(layer_floor_target * activation)
            budget_losses.append(layer_budget_loss)
            budget_targets.append(layer_budget_target * activation)
            self_losses.append(layer_self_loss)
            self_weights.append(layer_self_weight * activation)
            if layer_idx is not None:
                prefix = f"screening_layer_{layer_idx}_"
                per_layer_aux_metrics.update(
                    {
                        prefix + "admission_floor_loss": layer_floor_loss,
                        prefix + "admission_floor_target": (
                            layer_floor_target * activation
                        ),
                        prefix + "write_budget_loss": layer_budget_loss,
                        prefix + "write_budget_target": (
                            layer_budget_target * activation
                        ),
                        prefix + "self_index_loss": layer_self_loss,
                        prefix + "self_index_weight": (
                            layer_self_weight * activation
                        ),
                    }
                )
        admission_floor_loss = jnp.mean(jnp.stack(floor_losses))
        admission_floor_target = jnp.mean(jnp.stack(floor_targets))
        write_budget_loss = jnp.mean(jnp.stack(budget_losses))
        write_budget_target = jnp.mean(jnp.stack(budget_targets))
        self_index_loss = jnp.mean(jnp.stack(self_losses))
        self_index_weight = jnp.mean(jnp.stack(self_weights))
        total_loss += (
            admission_floor_loss + write_budget_loss + self_index_loss
        ) * aux_loss_scale
    metrics = {
        "loss": ce_loss,
        "admission_floor_loss": admission_floor_loss,
        "admission_floor_target": admission_floor_target,
        "write_budget_loss": write_budget_loss,
        "write_budget_target": write_budget_target,
        "self_index_loss": self_index_loss,
        "self_index_weight": self_index_weight,
        **normalized_stats,
        **per_layer_aux_metrics,
    }
    for key in (
        "rel_read_mean",
        "active_slots_mean",
        "u_norm_mean",
        "rel_write_mean",
        "rel_write_effective_mean",
        "slot_update_norm_mean",
        "slot_usage_ema_mean",
        "matched_route_mass",
        "novel_route_mass",
        "route_entropy",
        "route_top1_concentration",
        "admission_mean",
        "admission_low_rate",
        "admission_high_rate",
        "novel_token_rate",
        "rejected_write_rate",
        "short_bank_write_mass",
        "mid_bank_write_mass",
        "long_bank_write_mass",
        "eviction_age_mean",
        "eviction_usage_mean",
        "slot_utilization",
        "dead_slot_rate",
        "short_bank_slot_utilization",
        "mid_bank_slot_utilization",
        "long_bank_slot_utilization",
        "slot_cosine_redundancy",
        "matched_erase_mass",
        "matched_write_mass",
        "novel_erase_mass",
        "novel_write_mass",
        "accepted_novel_rate",
        "empty_allocation_rate",
        "occupied_eviction_rate",
        "read_energy_mean",
        "write_saturation_rate",
        "write_budget_rate",
        "hard_write_budget_rate",
        "write_budget_realized_rate",
        "write_self_similarity",
        "read_self_similarity",
        "self_index_raw_loss",
        "read_soft_warmup_alpha",
        "screening_residual_rms",
        "base_residual_rms",
        "screening_base_rms_ratio",
        "screening_residual_scale",
        "screening_activation",
    ):
        metrics.setdefault(key, jnp.zeros(()))
    return total_loss, (metrics, current_rwkv, current_screen)


@nnx.jit(
    static_argnames=("phase", "gradient_accumulation_steps"),
    donate_argnums=(0, 1),
)
def _nnx_train_step(
    model: NNXScreenedRWKVModel,
    optimizer: nnx.Optimizer,
    batch,
    rwkv_state,
    screen_state,
    *,
    phase: str,
    gradient_accumulation_steps: int,
):
    batch_size = int(batch["input_ids"].shape[0])
    if batch_size % gradient_accumulation_steps != 0:
        raise ValueError(
            "batch size must be divisible by gradient_accumulation_steps"
        )
    microbatch_size = batch_size // gradient_accumulation_steps
    token_count = int(batch["input_ids"].shape[1])
    accum_dtype = _dtype_from_name(model.config.gradient_accum_dtype)
    update_dtype = _dtype_from_name(model.config.param_update_dtype)
    mask = batch.get("mask")
    total_ce_count = (
        jnp.asarray(batch_size * token_count, dtype=jnp.float32)
        if mask is None
        else jnp.sum(mask, dtype=jnp.float32)
    )
    total_ce_count = jnp.maximum(total_ce_count, 1.0)
    total_positions = jnp.asarray(batch_size * token_count, dtype=jnp.float32)
    accumulated_grads = None
    accumulated_metrics = None
    accumulated_loss = jnp.zeros((), dtype=jnp.float32)
    new_rwkv_states = []
    new_screen_states = []

    for microstep in range(gradient_accumulation_steps):
        start = microstep * microbatch_size
        stop = start + microbatch_size
        microbatch = _slice_batch_tree(batch, start, stop)
        micro_rwkv = _slice_batch_tree(rwkv_state, start, stop)
        micro_screen = _slice_batch_tree(screen_state, start, stop)
        micro_mask = microbatch.get("mask")
        micro_ce_count = (
            jnp.asarray(microbatch_size * token_count, dtype=jnp.float32)
            if micro_mask is None
            else jnp.sum(micro_mask, dtype=jnp.float32)
        )
        ce_weight = micro_ce_count / total_ce_count
        position_weight = (
            jnp.asarray(microbatch_size * token_count, dtype=jnp.float32)
            / total_positions
        )

        def loss_fn(active_model):
            return nnx_model_loss(
                active_model,
                microbatch,
                micro_rwkv,
                micro_screen,
                phase=phase,
                deterministic=False,
                include_l2wrap=True,
                ce_loss_scale=ce_weight,
                l2_loss_scale=position_weight,
                aux_loss_scale=position_weight,
                training_step=optimizer.step[...],
            )

        (loss, (metrics, new_rwkv, new_screen)), grads = nnx.value_and_grad(
            loss_fn, has_aux=True
        )(model)
        grads = _cast_gradient_tree(grads, accum_dtype)
        accumulated_grads = (
            grads
            if accumulated_grads is None
            else jax.tree.map(jnp.add, accumulated_grads, grads)
        )
        weighted_metrics = {
            key: value * (ce_weight if key == "loss" else position_weight)
            for key, value in metrics.items()
        }
        accumulated_metrics = (
            weighted_metrics
            if accumulated_metrics is None
            else jax.tree.map(jnp.add, accumulated_metrics, weighted_metrics)
        )
        accumulated_loss += loss
        new_rwkv_states.append(new_rwkv)
        new_screen_states.append(new_screen)

    accumulated_grads = _cast_gradient_tree(accumulated_grads, update_dtype)
    (
        gradient_all_finite,
        gradient_global_norm,
        gradient_max_abs,
    ) = _tree_gradient_statistics(accumulated_grads)
    optimizer.update(model, accumulated_grads)
    metrics = accumulated_metrics
    metrics["total_loss"] = accumulated_loss
    metrics["gradient_all_finite"] = gradient_all_finite
    metrics["gradient_global_norm"] = gradient_global_norm
    metrics["gradient_max_abs"] = gradient_max_abs
    metrics["parameter_all_finite"] = _tree_finite_flag(
        nnx.state(model, nnx.Param)
    )
    return (
        _concat_batch_trees(new_rwkv_states),
        _concat_batch_trees(new_screen_states),
        metrics,
    )


def nnx_train_step(
    train_state: NNXTrainState,
    batch,
    rwkv_state,
    screen_state,
    *,
    phase="read_screening_only",
    gradient_accumulation_steps=1,
):
    new_rwkv_state, new_screen_state, metrics = _nnx_train_step(
        train_state.model,
        train_state.optimizer,
        batch,
        rwkv_state,
        screen_state,
        phase=phase,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    return train_state, new_rwkv_state, new_screen_state, metrics


__all__ = [
    "NNXTrainState",
    "compute_v5_admission_floor_loss",
    "compute_v5_self_index_loss",
    "compute_v5_write_budget_loss",
    "create_nnx_train_state",
    "initialize_nnx_train_state",
    "nnx_model_loss",
    "nnx_train_step",
]
