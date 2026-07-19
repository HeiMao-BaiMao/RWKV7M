from dataclasses import replace
import json
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import pytest

from rwkv7m.api import tiny_config
from rwkv7m.distributed import make_mesh
from rwkv7m.io import load_model_config, model_config_from_dict, model_config_to_dict
from rwkv7m.model.nnx_model import (
    NNXScreenedRWKVModel,
    NNXShardingConfig,
    NNXStateLevelScreening,
)
from rwkv7m.model.screened_rwkv import cross_entropy_loss
from rwkv7m.model.screening import (
    ScreeningConfig,
    bounded_non_amplifying_aggregate,
    capacity_calibrated_similarity_threshold,
    resolve_semantics_version,
    smooth_trim_square,
)
from rwkv7m.model.screening_v5 import (
    ACCEPTED_NOVEL_RATE,
    EMPTY_ALLOCATION_RATE,
    MATCHED_ROUTE_MASS,
    MATCHED_ERASE_MASS,
    MATCHED_WRITE_MASS,
    NOVEL_ERASE_MASS,
    NOVEL_RATE,
    NOVEL_WRITE_MASS,
    OCCUPANCY_MEAN,
    READ_MAX,
    SELF_INDEX_LOSS,
    Z_NORM,
    ScreeningV5RecurrenceConfig,
    ambiguity_aware_matched_routing,
    deterministic_lowest_argmax,
    screening_v5_recurrence_reference,
)
from rwkv7m.model.state import (
    LayerScreenState,
    init_rwkv_state,
    init_screen_state,
)
from rwkv7m.train import (
    compute_v5_admission_floor_loss,
    compute_v5_self_index_loss,
    compute_v5_write_budget_loss,
    create_nnx_train_state,
    nnx_train_step,
)


def _v5_screening_config(**overrides):
    values = {
        "d_model": 32,
        "d_slot": 16,
        "d_k": 8,
        "d_v": 8,
        "n_slots": 4,
        "screened_layers": (0,),
        "bank_ids": (0, 0, 1, 2),
        "use_write_screening": True,
        "write_mode": "competitive_novel",
        "semantics_version": "screening-v5-core",
        "gate_space": "value",
        "candidate_rank": 4,
        "n_read_tiles": 2,
        "capacity_calibration": "analytic",
        "admission_init": 0.6,
        "checkpoint_interval": None,
        "edit_mode": "tied",
        "allocation_redundancy_weight": 0.0,
    }
    values.update(overrides)
    return ScreeningConfig(**values)


def _v5_recurrence_config(**overrides):
    values = {
        "use_value_unit_norm": True,
        "usage_ema_decay": 0.99,
        "tanh_norm_cap": 1.0,
        "eps": 1e-6,
        "bank_ids": (0, 0, 1, 2),
        "route_power": 2.0,
        "novelty_temperature": 0.1,
        "admission_threshold": 0.5,
        "allocation_temperature": 1.0,
        "bank_route_temperature": 1.0,
        "allocation_age_weight": 1.0,
        "allocation_usage_weight": 1.0,
        "allocation_redundancy_weight": 0.5,
        "n_read_tiles": 2,
        "tau_min": -0.95,
        "tau_max": 0.95,
        "target_false_read_rate": 0.05,
        "target_false_write_rate": 0.05,
        "target_false_match_rate": 0.01,
        "threshold_warmup_by_load": True,
        "threshold_warmup_tau": -0.25,
        "eta_ambiguity": 0.5,
        "edit_mode": "tied",
        "write_accounting_floor": 1e-4,
    }
    values.update(overrides)
    return ScreeningV5RecurrenceConfig(**values)


def _v5_inputs(time=1):
    batch, slots, key, slot, value = 1, 4, 4, 4, 4
    q_read = jnp.tile(
        jnp.asarray([[[1.0, 0.0, 1.0, 0.0]]], dtype=jnp.float32),
        (time, 1, 1),
    )
    q_write = jnp.tile(
        jnp.asarray([[[1.0, 0.0, 0.0, 0.0]]], dtype=jnp.float32),
        (time, 1, 1),
    )
    delta_slots = jnp.zeros(
        (time, batch, slots, slot),
        dtype=jnp.float32,
    )
    delta_slots = delta_slots.at[:, 0, 0].set(
        jnp.asarray([1.0, 2.0, 3.0, 4.0])
    )
    delta_slots = delta_slots.at[:, 0, 1].set(
        jnp.asarray([5.0, 6.0, 7.0, 8.0])
    )
    delta_keys = jnp.zeros(
        (time, batch, slots, key),
        dtype=jnp.float32,
    )
    delta_keys = delta_keys.at[:, 0, :, 0].set(1.0)
    delta_values = jnp.zeros(
        (time, batch, slots, value),
        dtype=jnp.float32,
    )
    delta_values = delta_values.at[:, 0, :, 0].set(1.0)
    return (
        q_read,
        q_write,
        jnp.full((time, batch), 10.0, dtype=jnp.float32),
        jnp.zeros((5,), dtype=jnp.float32),
        jnp.tile(
            jnp.asarray([[[3.0, 2.0, 1.0]]], dtype=jnp.float32),
            (time, 1, 1),
        ),
        jnp.zeros((time, batch, slots), dtype=jnp.float32),
        jnp.zeros((time, batch, slots), dtype=jnp.float32),
        jnp.zeros((time, batch), dtype=jnp.float32),
        jnp.zeros((time, batch), dtype=jnp.float32),
        delta_slots,
        delta_keys,
        delta_values,
        delta_keys,
        jnp.zeros((batch, slots, slot), dtype=jnp.float32),
        jnp.zeros((batch, slots, key), dtype=jnp.float32),
        jnp.zeros((batch, slots, value), dtype=jnp.float32),
        jnp.zeros((batch, slots, key), dtype=jnp.float32),
        jnp.zeros((batch, slots), dtype=jnp.float32),
        jnp.zeros((batch, slots), dtype=jnp.float32),
        jnp.zeros((batch, slots), dtype=jnp.float32),
        jnp.asarray([0.05, 0.05, 0.02, 0.005], dtype=jnp.float32),
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.zeros((), dtype=jnp.float32),
    )


def test_semantics_version_migration_is_explicit():
    assert (
        resolve_semantics_version(
            ScreeningConfig(write_mode="competitive_novel")
        )
        == "screening-v4-competitive"
    )
    assert (
        resolve_semantics_version(ScreeningConfig())
        == "screening-v4-legacy"
    )
    assert (
        resolve_semantics_version(_v5_screening_config())
        == "screening-v5-core"
    )


def test_v5_config_rejects_unsafe_or_incomplete_contracts():
    with pytest.raises(ValueError, match="gate_space"):
        _v5_screening_config(gate_space="model")
    with pytest.raises(ValueError, match="factorized"):
        _v5_screening_config(candidate_rank=None)
    with pytest.raises(ValueError, match="checkpoint redesign"):
        _v5_screening_config(checkpoint_interval=16)
    with pytest.raises(ValueError, match="tau_min and tau_max"):
        _v5_screening_config(tau_max=1.0)
    with pytest.raises(ValueError, match="requires write_mode"):
        _v5_screening_config(write_mode="legacy_threshold")
    with pytest.raises(ValueError, match="read_soft_warmup_temperature"):
        _v5_screening_config(read_soft_warmup_temperature=0.0)
    with pytest.raises(ValueError, match="write_budget_target_max"):
        _v5_screening_config(write_budget_target_max=1.1)
    with pytest.raises(ValueError, match="self_index_loss_steps"):
        _v5_screening_config(
            self_index_loss_weight=0.1,
            self_index_loss_steps=0,
        )


def test_v5_example_configs_roundtrip():
    for path in (
        "configs/rwkv7m-0.185b-screening-v5-core.json.example",
        "configs/rwkv7m-0.3b-screening-v5-core.json.example",
    ):
        config = load_model_config(path)
        assert config.screening.semantics_version == "screening-v5-core"
        restored = model_config_from_dict(model_config_to_dict(config))
        assert restored == config


def test_capacity_calibration_increases_with_test_count():
    small = capacity_calibrated_similarity_threshold(
        4,
        64,
        0.05,
        tau_min=-0.95,
        tau_max=0.95,
    )
    large = capacity_calibrated_similarity_threshold(
        256,
        64,
        0.05,
        tau_min=-0.95,
        tau_max=0.95,
    )
    assert large > small
    assert large < 1.0


def test_capacity_calibration_uses_null_cdf_not_loose_tail_bound():
    threshold = capacity_calibrated_similarity_threshold(
        64,
        16,
        0.05,
        tau_min=-0.95,
        tau_max=0.95,
    )
    loose_bound = jnp.sqrt(2.0 * jnp.log(64.0 / 0.05) / 16.0)
    assert jnp.allclose(threshold, 0.78887224, rtol=1e-5)
    assert threshold < loose_bound


def test_smooth_trim_square_keeps_a_finite_gradient_below_hard_threshold():
    similarity = jnp.asarray(0.2, dtype=jnp.float32)
    threshold = jnp.asarray(0.5, dtype=jnp.float32)

    def relevance(active_similarity):
        return smooth_trim_square(
            active_similarity,
            threshold,
            0.1,
        )

    value, gradient = jax.value_and_grad(relevance)(similarity)
    assert value > 0.0
    assert jnp.isfinite(gradient)
    assert gradient > 0.0


def test_v5_learned_threshold_offsets_start_at_zero():
    module = NNXStateLevelScreening(
        _v5_screening_config(),
        rngs=nnx.Rngs(0),
    )
    assert module.q_proj_r.precision == jax.lax.Precision.HIGH
    assert jnp.array_equal(
        module.tau_r_offset[...],
        jnp.zeros((2,), dtype=module.tau_r_offset[...].dtype),
    )
    assert module.tau_w_offset[...] == 0.0


def test_bounded_aggregate_never_amplifies_and_caps_aligned_values():
    weak = jnp.asarray([[0.25, 0.0]], dtype=jnp.float32)
    weak_output, weak_energy = bounded_non_amplifying_aggregate(weak)
    assert jnp.allclose(weak_output, weak)
    assert jnp.allclose(weak_energy, 0.0625)

    aligned = jnp.asarray([[3.0, 0.0]], dtype=jnp.float32)
    output, energy = bounded_non_amplifying_aggregate(aligned)
    assert jnp.allclose(jnp.linalg.norm(output, axis=-1), 1.0)
    assert jnp.allclose(energy, 9.0)


def test_diffuse_match_has_lower_confidence_than_clear_match():
    clear = jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    diffuse = jnp.asarray([[1.0, 1.0, 0.0, 0.0]], dtype=jnp.float32)
    clear_result = ambiguity_aware_matched_routing(
        clear,
        route_power=1.0,
        eta_ambiguity=1.0,
    )
    diffuse_result = ambiguity_aware_matched_routing(
        diffuse,
        route_power=1.0,
        eta_ambiguity=1.0,
    )
    assert jnp.allclose(clear_result[3], 1.0)
    assert jnp.allclose(diffuse_result[3], 0.5)


def test_empty_matched_route_has_finite_zero_gradient():
    def confidence(eligibility):
        return jnp.sum(
            ambiguity_aware_matched_routing(
                eligibility,
                route_power=1.0,
                eta_ambiguity=0.5,
            )[3]
        )

    gradient = jax.grad(confidence)(jnp.zeros((1, 4), dtype=jnp.float32))
    assert jnp.array_equal(gradient, jnp.zeros_like(gradient))


def test_underflowed_matched_route_has_finite_zero_gradient():
    def confidence(eligibility):
        return jnp.sum(
            ambiguity_aware_matched_routing(
                eligibility,
                route_power=2.0,
                eta_ambiguity=0.5,
            )[3]
        )

    eligibility = jnp.asarray([[1e-30, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    value, gradient = jax.value_and_grad(confidence)(eligibility)
    assert value == 0.0
    assert jnp.array_equal(gradient, jnp.zeros_like(gradient))


def test_empty_read_diagnostic_norm_has_zero_finite_gradient():
    inputs = list(_v5_inputs())

    def objective(q_read):
        values = list(inputs)
        values[0] = q_read
        statistics = screening_v5_recurrence_reference(
            *values,
            _v5_recurrence_config(),
        )[5]
        return statistics[0, 0, Z_NORM]

    gradient = jax.grad(objective)(inputs[0])
    assert jnp.array_equal(gradient, jnp.zeros_like(gradient))


def test_all_empty_uses_lowest_empty_slot_and_updates_occupancy():
    outputs = screening_v5_recurrence_reference(
        *_v5_inputs(),
        _v5_recurrence_config(),
    )
    u, slots, ages, usage, occupancy, statistics, _ = outputs
    assert jnp.allclose(u, 0.0)
    assert jnp.allclose(
        slots[0, 0],
        jnp.asarray([1.0, 2.0, 3.0, 4.0]),
    )
    assert jnp.allclose(slots[0, 1:], 0.0)
    assert jnp.array_equal(
        occupancy,
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]]),
    )
    assert jnp.allclose(ages, 0.0)
    assert jnp.allclose(usage, 0.0)
    assert statistics[0, 0, ACCEPTED_NOVEL_RATE] == 1.0
    assert statistics[0, 0, EMPTY_ALLOCATION_RATE] == 1.0
    assert statistics[0, 0, OCCUPANCY_MEAN] == 0.25


def test_lowest_argmax_only_breaks_exact_ties():
    mask = jnp.asarray([[True, True, True]])
    exact = deterministic_lowest_argmax(
        jnp.asarray([[1.0, 1.0, 0.0]], dtype=jnp.float32),
        mask,
    )
    near = deterministic_lowest_argmax(
        jnp.asarray([[1.0, 1.0000005, 0.0]], dtype=jnp.float32),
        mask,
    )
    assert jnp.array_equal(exact, jnp.asarray([[1.0, 0.0, 0.0]]))
    assert jnp.array_equal(near, jnp.asarray([[0.0, 1.0, 0.0]]))


def test_v5_core_versioned_semantic_golden_vector():
    path = Path("tests/golden/screening-v5-core-v1.json.example")
    golden = json.loads(path.read_text(encoding="utf-8"))
    assert golden["semantics_version"] == "screening-v5-core"
    expected = golden["expected"]
    inputs = _v5_inputs()
    outputs = screening_v5_recurrence_reference(
        *inputs,
        _v5_recurrence_config(),
    )
    _, slots, ages, usage, occupancy, statistics, _ = outputs

    assert statistics[0, 0, READ_MAX] == max(expected["read_relevance"])
    assert statistics[0, 0, MATCHED_ROUTE_MASS] == sum(
        expected["matched_route"]
    )
    assert statistics[0, 0, NOVEL_RATE] == expected["novel_decision"]
    assert statistics[0, 0, ACCEPTED_NOVEL_RATE] == expected[
        "admission_decision"
    ]
    assert statistics[0, 0, MATCHED_ERASE_MASS] == sum(
        expected["erase_mass"]
    )
    assert statistics[0, 0, MATCHED_WRITE_MASS] == 0.0
    assert statistics[0, 0, NOVEL_ERASE_MASS] == sum(
        expected["erase_mass"]
    )
    assert statistics[0, 0, NOVEL_WRITE_MASS] == sum(
        expected["write_mass"]
    )
    assert int(jnp.argmax(jnp.linalg.norm(slots[0], axis=-1))) == expected[
        "victim_address"
    ]
    assert jnp.allclose(slots, jnp.asarray(expected["final_slots"]))
    assert jnp.allclose(ages, jnp.asarray(expected["final_ages"]))
    assert jnp.allclose(usage, jnp.asarray(expected["final_usage"]))
    assert jnp.array_equal(
        occupancy,
        jnp.asarray(expected["final_occupancy"]),
    )

    probe = expected["admission_context_gradient_probe"]

    def objective(admission_context):
        values = list(inputs)
        values[2] = admission_context
        return jnp.sum(
            screening_v5_recurrence_reference(
                *values,
                _v5_recurrence_config(),
            )[1]
        )

    gradient = jax.grad(objective)(jnp.asarray(probe["input"]))
    assert jnp.allclose(
        gradient,
        jnp.asarray(probe["gradient"]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_rejected_novel_token_changes_no_content_or_metadata():
    inputs = list(_v5_inputs())
    inputs[2] = jnp.full((1, 1), -10.0, dtype=jnp.float32)
    initial_ages = jnp.asarray([[2.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    initial_occupancy = jnp.asarray(
        [[1.0, 0.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    initial_slots = jnp.zeros((1, 4, 4), dtype=jnp.float32).at[0, 0, 0].set(2.0)
    inputs[13] = initial_slots
    inputs[17] = initial_ages
    inputs[19] = initial_occupancy
    outputs = screening_v5_recurrence_reference(
        *inputs,
        _v5_recurrence_config(),
    )
    assert jnp.allclose(outputs[1], initial_slots)
    assert jnp.array_equal(outputs[4], initial_occupancy)
    assert outputs[2][0, 0] == 3.0


def test_capacity_conserving_matched_write_never_exceeds_erase():
    inputs = list(_v5_inputs())
    inputs[14] = inputs[14].at[0, 0, 0].set(1.0)
    inputs[16] = inputs[16].at[0, 0, 0].set(1.0)
    inputs[19] = inputs[19].at[0, 0].set(1.0)
    outputs = screening_v5_recurrence_reference(
        *inputs,
        _v5_recurrence_config(edit_mode="capacity_conserving"),
    )
    statistics = outputs[5]
    erase = statistics[0, 0, MATCHED_ERASE_MASS]
    write = statistics[0, 0, MATCHED_WRITE_MASS]
    assert erase > 0.0
    assert 0.0 < write <= erase


def test_v5_route_has_soft_admission_gradient():
    inputs = _v5_inputs()

    def objective(admission_context):
        values = list(inputs)
        values[2] = admission_context
        outputs = screening_v5_recurrence_reference(
            *values,
            _v5_recurrence_config(),
        )
        return jnp.sum(outputs[1])

    gradient = jax.grad(objective)(
        jnp.asarray([[1.0]], dtype=jnp.float32)
    )
    assert jnp.all(jnp.isfinite(gradient))
    assert gradient[0, 0] > 0.0


def test_v5_self_index_loss_trains_rejected_candidate_geometry():
    inputs = list(_v5_inputs())
    orthogonal_keys = jnp.zeros_like(inputs[10])
    orthogonal_keys = orthogonal_keys.at[..., 1].set(1.0)
    orthogonal_keys = orthogonal_keys.at[..., 3].set(1.0)
    inputs[10] = orthogonal_keys
    inputs[12] = orthogonal_keys
    inputs[14] = orthogonal_keys[0]
    inputs[16] = orthogonal_keys[0]
    inputs[19] = jnp.ones_like(inputs[19])
    config = _v5_recurrence_config(
        self_index_margin=0.02,
        target_false_write_rate=0.5,
        target_false_match_rate=0.1,
    )

    def objective(candidate_keys):
        values = list(inputs)
        values[10] = candidate_keys
        values[12] = candidate_keys
        return jnp.sum(
            screening_v5_recurrence_reference(
                *values,
                config,
            )[5][..., SELF_INDEX_LOSS]
        )

    loss, gradient = jax.value_and_grad(objective)(orthogonal_keys)
    assert loss > 0.0
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.linalg.norm(gradient) > 0.0

    def threshold_objective(tau_write_offset):
        values = list(inputs)
        values[22] = tau_write_offset
        return jnp.sum(
            screening_v5_recurrence_reference(
                *values,
                config,
            )[5][..., SELF_INDEX_LOSS]
        )

    threshold_gradient = jax.grad(threshold_objective)(inputs[22])
    assert threshold_gradient == 0.0

    def read_threshold_objective(tau_read_offset):
        values = list(inputs)
        values[21] = tau_read_offset
        return jnp.sum(
            screening_v5_recurrence_reference(
                *values,
                config,
            )[5][..., SELF_INDEX_LOSS]
        )

    read_threshold_gradient = jax.grad(read_threshold_objective)(inputs[21])
    assert jnp.array_equal(
        read_threshold_gradient,
        jnp.zeros_like(read_threshold_gradient),
    )


def test_admission_floor_is_temporary_and_only_uses_empty_capacity():
    config = _v5_screening_config(
        admission_floor_target_initial=0.1,
        admission_floor_weight=0.5,
        admission_floor_steps=100,
    )

    def loss_at_rate(write_rate):
        return compute_v5_admission_floor_loss(
            write_rate,
            jnp.asarray(0.25, dtype=jnp.float32),
            config,
            jnp.asarray(0, dtype=jnp.int32),
        )[0]

    loss, target = compute_v5_admission_floor_loss(
        jnp.asarray(0.01, dtype=jnp.float32),
        jnp.asarray(0.25, dtype=jnp.float32),
        config,
        jnp.asarray(0, dtype=jnp.int32),
    )
    assert jnp.allclose(target, 0.075)
    assert loss > 0.0
    assert jax.grad(loss_at_rate)(jnp.asarray(0.01)) < 0.0

    expired_loss, expired_target = compute_v5_admission_floor_loss(
        jnp.asarray(0.0),
        jnp.asarray(0.25),
        config,
        jnp.asarray(100),
    )
    full_loss, full_target = compute_v5_admission_floor_loss(
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        config,
        jnp.asarray(0),
    )
    assert expired_loss == 0.0
    assert expired_target == 0.0
    assert full_loss == 0.0
    assert full_target == 0.0


def test_write_budget_penalizes_only_rates_above_the_ceiling():
    config = _v5_screening_config(
        write_budget_target_max=0.1,
        write_budget_weight=0.5,
    )

    def budget(rate):
        return compute_v5_write_budget_loss(rate, config)[0]

    assert budget(jnp.asarray(0.05)) == 0.0
    loss, gradient = jax.value_and_grad(budget)(jnp.asarray(0.5))
    assert loss > 0.0
    assert gradient > 0.0


def test_self_index_curriculum_anneals_without_changing_raw_metric():
    config = _v5_screening_config(
        self_index_loss_weight=0.5,
        self_index_loss_steps=100,
    )
    initial, initial_weight = compute_v5_self_index_loss(
        jnp.asarray(2.0), config, jnp.asarray(0)
    )
    expired, expired_weight = compute_v5_self_index_loss(
        jnp.asarray(2.0), config, jnp.asarray(100)
    )
    assert initial == 1.0
    assert initial_weight == 0.5
    assert expired == 0.0
    assert expired_weight == 0.0


def test_nnx_v5_sequence_chunks_preserve_state_and_output():
    config = _v5_screening_config()
    module = NNXStateLevelScreening(config, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (1, 4, 32))
    h = jax.random.normal(jax.random.key(2), (1, 4, 32))
    initial = LayerScreenState(
        slots=jnp.zeros((1, 4, 16), dtype=jnp.float32),
        ages=jnp.zeros((1, 4), dtype=jnp.float32),
        usage_ema=jnp.zeros((1, 4), dtype=jnp.float32),
        occupancy=jnp.zeros((1, 4), dtype=jnp.float32),
    )
    full_h, full_state, full_stats = module(
        x,
        h,
        initial,
        phase="read_write",
        deterministic=False,
    )
    first_h, first_state, _ = module(
        x[:, :2],
        h[:, :2],
        initial,
        phase="read_write",
        deterministic=False,
    )
    second_h, second_state, _ = module(
        x[:, 2:],
        h[:, 2:],
        first_state,
        phase="read_write",
        deterministic=False,
    )
    assert jnp.allclose(
        full_h,
        jnp.concatenate([first_h, second_h], axis=1),
        rtol=1e-5,
        atol=1e-5,
    )
    assert jnp.allclose(full_state.slots, second_state.slots)
    assert jnp.allclose(full_state.ages, second_state.ages)
    assert jnp.allclose(full_state.usage_ema, second_state.usage_ema)
    assert jnp.array_equal(full_state.occupancy, second_state.occupancy)
    assert full_stats["slot_utilization"] > 0.0
    assert full_stats["matched_write_mass"] <= full_stats["matched_erase_mass"]


def test_nnx_v5_lambda_floor_only_applies_during_warmup():
    config = _v5_screening_config(
        lambda_screen_warmup_floor=0.2,
        lambda_screen_warmup_steps=100,
    )
    module = NNXStateLevelScreening(config, rngs=nnx.Rngs(0))
    state = LayerScreenState(
        slots=jnp.zeros((1, 4, 16), dtype=jnp.float32),
        ages=jnp.zeros((1, 4), dtype=jnp.float32),
        usage_ema=jnp.zeros((1, 4), dtype=jnp.float32),
        occupancy=jnp.zeros((1, 4), dtype=jnp.float32),
    )
    values = (
        jnp.zeros((1, 1, 32), dtype=jnp.float32),
        jnp.zeros((1, 1, 32), dtype=jnp.float32),
        state,
    )
    _, _, warm_stats = module(
        *values,
        phase="read_write",
        training_step=jnp.asarray(0),
    )
    _, _, expired_stats = module(
        *values,
        phase="read_write",
        training_step=jnp.asarray(100),
    )
    tile_scale = jnp.sqrt(jnp.asarray(config.n_read_tiles, jnp.float32))
    assert jnp.allclose(warm_stats["lambda_screen_floor"], 0.2 / tile_scale)
    assert expired_stats["lambda_screen_floor"] == 0.0
    assert warm_stats["lambda_screen"] > expired_stats["lambda_screen"]


def test_nnx_v5_read_curriculum_is_training_only_and_expires():
    config = _v5_screening_config(read_soft_warmup_steps=100)
    module = NNXStateLevelScreening(config, rngs=nnx.Rngs(0))
    state = LayerScreenState(
        slots=jnp.ones((1, 4, 16), dtype=jnp.float32),
        ages=jnp.zeros((1, 4), dtype=jnp.float32),
        usage_ema=jnp.zeros((1, 4), dtype=jnp.float32),
        occupancy=jnp.ones((1, 4), dtype=jnp.float32),
    )
    values = (
        jnp.zeros((1, 1, 32), dtype=jnp.float32),
        jnp.zeros((1, 1, 32), dtype=jnp.float32),
        state,
    )
    _, _, warm = module(
        *values,
        phase="read_write",
        deterministic=False,
        training_step=jnp.asarray(0),
    )
    _, _, expired = module(
        *values,
        phase="read_write",
        deterministic=False,
        training_step=jnp.asarray(100),
    )
    _, _, evaluation = module(
        *values,
        phase="read_write",
        deterministic=True,
        training_step=jnp.asarray(0),
    )
    assert warm["read_soft_warmup_alpha"] == 1.0
    assert expired["read_soft_warmup_alpha"] == 0.0
    assert evaluation["read_soft_warmup_alpha"] == 0.0


def test_nnx_v5_memory_off_override_zeroes_only_the_residual():
    config = _v5_screening_config(
        lambda_screen_init=0.2,
        read_soft_warmup_steps=100,
        read_soft_warmup_temperature=1.0,
        target_false_read_rate=0.5,
    )
    module = NNXStateLevelScreening(config, rngs=nnx.Rngs(0))
    state = LayerScreenState(
        slots=jax.random.normal(jax.random.key(20), (1, 4, 16)),
        ages=jnp.zeros((1, 4), dtype=jnp.float32),
        usage_ema=jnp.zeros((1, 4), dtype=jnp.float32),
        occupancy=jnp.ones((1, 4), dtype=jnp.float32),
    )
    x = jax.random.normal(jax.random.key(21), (1, 2, 32))
    h_base = jax.random.normal(jax.random.key(22), (1, 2, 32))
    enabled, enabled_state, _ = module(
        x,
        h_base,
        state,
        phase="read_write",
        deterministic=False,
        training_step=jnp.asarray(0),
    )
    disabled, disabled_state, disabled_stats = module(
        x,
        h_base,
        state,
        phase="read_write",
        deterministic=False,
        training_step=jnp.asarray(0),
        screening_residual_scale=0.0,
    )
    assert jnp.array_equal(disabled, h_base)
    assert jnp.allclose(enabled_state.slots, disabled_state.slots)
    assert disabled_stats["screening_residual_scale"] == 0.0
    assert not jnp.array_equal(enabled, disabled)


def test_nnx_v5_rejects_state_without_occupancy_metadata():
    config = _v5_screening_config()
    module = NNXStateLevelScreening(config, rngs=nnx.Rngs(0))
    state = LayerScreenState(
        slots=jnp.zeros((1, 4, 16), dtype=jnp.float32),
        ages=jnp.zeros((1, 4), dtype=jnp.float32),
        usage_ema=jnp.zeros((1, 4), dtype=jnp.float32),
    )
    with pytest.raises(ValueError, match="explicit occupancy metadata"):
        module(
            jnp.zeros((1, 1, 32), dtype=jnp.float32),
            jnp.zeros((1, 1, 32), dtype=jnp.float32),
            state,
            phase="read_write",
        )


def test_nnx_v5_full_train_step_has_finite_loss_and_gradients():
    config = tiny_config(
        vocab_size=32,
        d_model=32,
        n_layers=1,
        n_heads=4,
        head_size=8,
    )
    config.screening = _v5_screening_config()
    config.dtype = "bfloat16"
    config.param_dtype = "bfloat16"
    config.lm_head_init = "variance_scaled"
    config.remat_blocks = True
    model = NNXScreenedRWKVModel(config, rngs=nnx.Rngs(7))
    screening = model.layer_0.screening_0
    assert screening.compute_dtype == jnp.float32
    assert screening.q_proj_r.dtype == jnp.float32
    assert screening.q_proj_r.kernel[...].dtype == jnp.bfloat16
    graphdef, params = nnx.split(model, nnx.Param)
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    targets = jnp.asarray([[2, 3, 4]], dtype=jnp.int32)
    rwkv_state = init_rwkv_state(1, config)
    screen_state = init_screen_state(1, config.screening)

    def loss_fn(active_params):
        active_model = nnx.merge(graphdef, active_params)
        logits, _, next_screen_state, statistics = active_model(
            input_ids,
            rwkv_state,
            screen_state,
            phase="read_write",
            deterministic=False,
        )
        return cross_entropy_loss(logits, targets), (
            next_screen_state,
            statistics,
        )

    (loss, (next_screen_state, statistics)), gradients = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )(params)
    assert jnp.isfinite(loss)
    nonfinite_paths = [
        "/".join(str(part) for part in path)
        for path, value in nnx.to_flat_state(gradients)
        if not bool(jnp.all(jnp.isfinite(value)))
    ]
    assert not nonfinite_paths, nonfinite_paths
    assert next_screen_state.layers[0].occupancy is not None
    assert statistics["accepted_novel_rate"] >= 0.0
    assert statistics["write_saturation_rate"] >= 0.0


def test_nnx_train_step_applies_enabled_admission_floor():
    config = tiny_config(
        vocab_size=16,
        d_model=32,
        n_layers=1,
        n_heads=4,
        head_size=8,
    )
    config.screening = _v5_screening_config(
        admission_init=0.9,
        admission_floor_target_initial=1.0,
        admission_floor_weight=0.5,
        admission_floor_steps=10,
        read_soft_warmup_steps=10,
        write_budget_target_max=0.0,
        write_budget_weight=0.5,
        self_index_margin=0.1,
        self_index_loss_weight=0.5,
        self_index_loss_steps=10,
    )
    config.lm_head_init = "variance_scaled"
    model = NNXScreenedRWKVModel(config, rngs=nnx.Rngs(9))
    train_state = create_nnx_train_state(model, config, total_steps=10)
    batch = {
        "input_ids": jnp.asarray([[1, 2]], dtype=jnp.int32),
        "target_ids": jnp.asarray([[2, 3]], dtype=jnp.int32),
    }
    _, _, _, metrics = nnx_train_step(
        train_state,
        batch,
        init_rwkv_state(1, config),
        init_screen_state(1, config.screening),
        phase="read_write",
    )
    assert metrics["admission_floor_target"] > 0.0
    assert metrics["admission_floor_loss"] > 0.0
    assert metrics["write_budget_loss"] > 0.0
    assert metrics["self_index_weight"] > 0.0
    assert jnp.isfinite(metrics["self_index_loss"])
    assert metrics["read_soft_warmup_alpha"] == 1.0
    assert metrics["total_loss"] > metrics["loss"]


def test_nnx_v5_size_one_explicit_mesh_forward():
    config = _v5_screening_config()
    mesh = make_mesh(
        ("data", "model"),
        axis_sizes=(1, 1),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
        ),
    )
    sharding = NNXShardingConfig(mesh)
    module = NNXStateLevelScreening(
        config,
        rngs=nnx.Rngs(11),
        sharding=sharding,
    )
    state = LayerScreenState(
        slots=jax.device_put(
            jnp.zeros((1, 4, 16), dtype=jnp.float32),
            sharding.named("data", None, "model"),
        ),
        ages=jax.device_put(
            jnp.zeros((1, 4), dtype=jnp.float32),
            sharding.named("data", None),
        ),
        usage_ema=jax.device_put(
            jnp.zeros((1, 4), dtype=jnp.float32),
            sharding.named("data", None),
        ),
        occupancy=jax.device_put(
            jnp.zeros((1, 4), dtype=jnp.float32),
            sharding.named("data", None),
        ),
    )
    with jax.set_mesh(mesh):
        outputs = module(
            jax.device_put(
                jnp.zeros((1, 2, 32), dtype=jnp.float32),
                sharding.named("data", None, "model"),
            ),
            jax.device_put(
                jnp.zeros((1, 2, 32), dtype=jnp.float32),
                sharding.named("data", None, "model"),
            ),
            state,
            phase="read_write",
            training_step=jnp.asarray(0),
        )
    assert outputs[0].shape == (1, 2, 32)
    assert outputs[1].occupancy is not None
    assert jnp.all(jnp.isfinite(outputs[0]))


def test_retention_profile_is_not_silently_mapped_to_core():
    config = _v5_screening_config(
        semantics_version="screening-v5-retention"
    )
    with pytest.raises(NotImplementedError, match="gated"):
        NNXStateLevelScreening(config, rngs=nnx.Rngs(0))
