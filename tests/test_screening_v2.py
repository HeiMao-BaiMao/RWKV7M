from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from rwkv7m.distributed.mesh import make_mesh
from rwkv7m.io import model_config_from_dict, model_config_to_dict
from rwkv7m.model.nnx_model import NNXShardingConfig, NNXStateLevelScreening
from rwkv7m.model.screened_rwkv import ModelConfig
from rwkv7m.model.screening import (
    ScreeningConfig,
    competitive_write_routing,
    resolve_write_mode,
    unit_norm,
)
from rwkv7m.model.screening_recurrence import (
    REJECTED_RATE,
    ScreeningRecurrenceConfig,
    screening_recurrence,
    screening_recurrence_reference,
)
from rwkv7m.model.state import LayerScreenState


def _v2_config(**overrides):
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
        "gate_space": "value",
        "candidate_rank": 4,
        "n_read_tiles": 2,
        "checkpoint_interval": 2,
    }
    values.update(overrides)
    return ScreeningConfig(**values)


def _recurrence_config(**overrides):
    values = {
        "write_enabled": True,
        "use_value_unit_norm": True,
        "use_leaky_warmup": False,
        "leaky_alpha": 0.0,
        "leaky_gamma": 8.0,
        "use_age_mask": False,
        "age_ref": 32.0,
        "age_sigma": 8.0,
        "write_rel_floor": 1e-3,
        "usage_ema_decay": 0.99,
        "tanh_norm_cap": 1.0,
        "eps": 1e-6,
        "write_mode": "competitive_novel",
        "bank_ids": (0, 1, 2),
        "n_read_tiles": 1,
    }
    values.update(overrides)
    return ScreeningRecurrenceConfig(**values)


def _recurrence_inputs(time=2, batch=1, slots=3, key=2, slot=4, value=2):
    rngs = jax.random.split(jax.random.key(12), 12)

    def normal(index, shape, scale=0.1):
        return jax.random.normal(rngs[index], shape) * scale

    q_read = unit_norm(normal(0, (time, batch, key)))
    q_write = unit_norm(normal(1, (time, batch, key)))
    return (
        q_read,
        q_write,
        jax.nn.sigmoid(normal(2, (time, batch))),
        normal(3, (time, batch, 3)),
        normal(4, (time, batch, slots, slot)),
        normal(5, (time, batch, slots, key)),
        normal(6, (time, batch, slots, value)),
        normal(7, (time, batch, slots, key)),
        normal(8, (batch, slots, slot)),
        normal(9, (batch, slots, key)),
        normal(10, (batch, slots, value)),
        normal(11, (batch, slots, key)),
        jnp.zeros((batch, slots), dtype=jnp.float32),
        jnp.zeros((batch, slots), dtype=jnp.float32),
        jnp.asarray([0.05, 0.02, 0.005], dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )


def test_legacy_phase_mapping_is_explicit_and_stable():
    legacy = ScreeningConfig(use_write_screening=True)
    assert resolve_write_mode(legacy, "read_screening_only") == "legacy_unconditional"
    assert resolve_write_mode(legacy, "read_write") == "legacy_threshold"
    assert (
        resolve_write_mode(
            ScreeningConfig(write_mode="disabled"), "read_write"
        )
        == "disabled"
    )


def test_screening_v2_config_roundtrip_and_validation():
    screening = _v2_config()
    config = ModelConfig(
        d_model=32,
        d_ffn=64,
        n_layers=1,
        n_heads=2,
        head_size=16,
        vocab_size=64,
        max_seq_len=8,
        dtype="float32",
        use_screening=True,
        screening=screening,
    )
    restored = model_config_from_dict(model_config_to_dict(config))
    assert restored == config
    with pytest.raises(ValueError, match="reduce candidate projection"):
        _v2_config(candidate_rank=32)
    with pytest.raises(ValueError, match="divisible by n_read_tiles"):
        _v2_config(n_read_tiles=3)
    with pytest.raises(ValueError, match="Unknown write mode"):
        screening_recurrence_reference(
            *_recurrence_inputs(),
            config=_recurrence_config(write_mode="unknown"),
        )
    with pytest.raises(ValueError, match="bank_ids must contain only"):
        screening_recurrence_reference(
            *_recurrence_inputs(),
            config=_recurrence_config(bank_ids=(0, 1, 3)),
        )


def test_matched_route_preserves_absolute_confidence():
    route, matched, novel, confidence, is_novel, *_ = (
        competitive_write_routing(
            jnp.asarray([[0.01, 0.005, 0.0]], dtype=jnp.float32),
            jnp.zeros((1, 3), dtype=jnp.float32),
            jnp.zeros((1, 3), dtype=jnp.float32),
            jnp.asarray([0.8], dtype=jnp.float32),
            jnp.zeros((1, 3), dtype=jnp.float32),
            (0, 1, 2),
            route_power=2.0,
            novelty_threshold=0.0,
            allocation_temperature=1.0,
            bank_route_temperature=1.0,
            allocation_age_weight=1.0,
            allocation_usage_weight=1.0,
            hard_admission=False,
            admission_threshold=0.5,
            eps=1e-6,
        )
    )
    assert not bool(is_novel[0])
    assert jnp.allclose(confidence, 0.01)
    assert jnp.allclose(jnp.sum(route, axis=-1), confidence, atol=2e-6)
    assert jnp.allclose(route, matched)
    assert jnp.all(novel == 0.0)


def test_novel_route_is_sparse_forward_with_soft_gradient():
    ages = jnp.asarray([[4.0, 2.0, 8.0]], dtype=jnp.float32)
    usage = jnp.asarray([[0.4, 0.1, 0.0]], dtype=jnp.float32)
    eligibility = jnp.zeros((1, 3), dtype=jnp.float32)

    def objective(bank_logits):
        route, *_ = competitive_write_routing(
            eligibility,
            ages,
            usage,
            jnp.asarray([0.7], dtype=jnp.float32),
            bank_logits,
            (0, 1, 2),
            route_power=1.0,
            novelty_threshold=0.1,
            allocation_temperature=1.0,
            bank_route_temperature=1.0,
            allocation_age_weight=1.0,
            allocation_usage_weight=1.0,
            hard_admission=False,
            admission_threshold=0.5,
            eps=1e-6,
        )
        return jnp.sum(route * jnp.asarray([[1.0, 2.0, 3.0]]))

    logits = jnp.asarray([[0.3, 0.2, 0.1]], dtype=jnp.float32)
    route, *_ = competitive_write_routing(
        eligibility,
        ages,
        usage,
        jnp.asarray([0.7], dtype=jnp.float32),
        logits,
        (0, 1, 2),
        route_power=1.0,
        novelty_threshold=0.1,
        allocation_temperature=1.0,
        bank_route_temperature=1.0,
        allocation_age_weight=1.0,
        allocation_usage_weight=1.0,
        hard_admission=False,
        admission_threshold=0.5,
        eps=1e-6,
    )
    assert jnp.count_nonzero(route) == 1
    assert jnp.allclose(jnp.sum(route), 0.7)
    assert jnp.linalg.norm(jax.grad(objective)(logits)) > 0.0


def test_rejected_novel_write_does_not_change_content_or_reset_age():
    inputs = list(_recurrence_inputs(time=1))
    inputs[0] = jnp.asarray([[[1.0, 0.0]]], dtype=jnp.float32)
    inputs[1] = jnp.asarray([[[1.0, 0.0]]], dtype=jnp.float32)
    inputs[2] = jnp.asarray([[0.1]], dtype=jnp.float32)
    inputs[9] = jnp.asarray(
        [[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]], dtype=jnp.float32
    )
    inputs[11] = inputs[9]
    inputs[12] = jnp.asarray([[3.0, 4.0, 5.0]], dtype=jnp.float32)
    initial_slots = inputs[8]
    outputs = screening_recurrence_reference(
        *inputs,
        _recurrence_config(
            hard_admission=True,
            admission_threshold=0.5,
            novelty_threshold=0.1,
            bank_ids=(0, 1, 2),
        ),
    )
    assert jnp.allclose(outputs[1], initial_slots)
    assert jnp.allclose(outputs[2], inputs[12] + 1.0)
    assert outputs[4][0, 0, REJECTED_RATE] == 1.0


def test_legacy_modes_match_implicit_pre_v2_mapping():
    inputs = _recurrence_inputs()
    implicit = _recurrence_config(
        write_enabled=False,
        write_mode=None,
        bank_ids=(0, 1, 2),
    )
    explicit = replace(implicit, write_mode="legacy_unconditional")
    expected = screening_recurrence_reference(*inputs, implicit)
    actual = screening_recurrence_reference(*inputs, explicit)
    for left, right in zip(actual, expected, strict=True):
        assert jnp.allclose(left, right)


def test_screening_v2_sequence_chunks_preserve_semantics():
    config = _v2_config(checkpoint_interval=None)
    module = NNXStateLevelScreening(config, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (1, 6, 32))
    h = jax.random.normal(jax.random.key(2), (1, 6, 32))
    initial = LayerScreenState(
        slots=jnp.zeros((1, 4, 16), dtype=jnp.float32),
        ages=jnp.zeros((1, 4), dtype=jnp.float32),
        usage_ema=jnp.zeros((1, 4), dtype=jnp.float32),
    )
    full_h, full_state, _ = module(
        x, h, initial, phase="read_write", deterministic=False
    )
    first_h, first_state, _ = module(
        x[:, :2], h[:, :2], initial, phase="read_write", deterministic=False
    )
    second_h, second_state, _ = module(
        x[:, 2:], h[:, 2:], first_state, phase="read_write", deterministic=False
    )
    assert jnp.allclose(full_h, jnp.concatenate([first_h, second_h], axis=1))
    assert jnp.allclose(full_state.slots, second_state.slots)
    assert jnp.allclose(full_state.ages, second_state.ages)
    assert jnp.allclose(full_state.usage_ema, second_state.usage_ema)


def test_factorized_candidate_respects_explicit_data_model_mesh():
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
        _v2_config(checkpoint_interval=None),
        rngs=nnx.Rngs(0),
        sharding=sharding,
    )
    hidden_sharding = sharding.named("data", None, "model")
    scalar_sharding = sharding.named("data", None)
    x = jax.device_put(jnp.ones((1, 2, 32)), hidden_sharding)
    state = LayerScreenState(
        slots=jax.device_put(jnp.zeros((1, 4, 16)), hidden_sharding),
        ages=jax.device_put(jnp.zeros((1, 4)), scalar_sharding),
        usage_ema=jax.device_put(jnp.zeros((1, 4)), scalar_sharding),
    )
    output, _, _ = module(
        x,
        x,
        state,
        phase="read_write",
        deterministic=False,
    )
    jax.block_until_ready(output)
    assert output.sharding.spec == hidden_sharding.spec


@pytest.mark.parametrize("backend", ["pallas_gpu_triton", "pallas_tpu"])
def test_screening_v2_pallas_interpret_checkpoint_matches_reference(backend):
    inputs = _recurrence_inputs(time=4, key=4, value=4)
    config = _recurrence_config(
        bank_ids=(0, 1, 2),
        n_read_tiles=2,
        checkpoint_interval=2,
    )
    inputs = list(inputs)
    q_tiled = inputs[0].reshape(4, 1, 2, 2)
    inputs[0] = unit_norm(q_tiled).reshape(4, 1, 4)
    inputs[15] = jnp.asarray([0.0, 0.0], dtype=jnp.float32)
    inputs = tuple(inputs)
    expected = screening_recurrence_reference(*inputs, config)
    actual = screening_recurrence(
        *inputs, config, backend=backend, interpret=True
    )
    for left, right in zip(actual, expected, strict=True):
        assert jnp.allclose(
            left.astype(jnp.float32),
            right.astype(jnp.float32),
            rtol=2e-5,
            atol=2e-5,
        )

    def loss(function):
        outputs = function(*inputs)
        return sum(jnp.sum(value.astype(jnp.float32) ** 2) for value in outputs[:4])

    expected_grad = jax.grad(
        lambda *values: loss(
            lambda *_: screening_recurrence_reference(*values, config)
        ),
        argnums=tuple(range(17)),
    )(*inputs)
    actual_grad = jax.grad(
        lambda *values: loss(
            lambda *_: screening_recurrence(
                *values, config, backend=backend, interpret=True
            )
        ),
        argnums=tuple(range(17)),
    )(*inputs)
    for left, right in zip(actual_grad, expected_grad, strict=True):
        assert jnp.allclose(
            left.astype(jnp.float32),
            right.astype(jnp.float32),
            rtol=2e-4,
            atol=2e-4,
        )
