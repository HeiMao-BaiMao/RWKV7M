import jax
import jax.numpy as jnp
from flax import nnx, traverse_util
import pytest

from rwkv7m.api import create_train_runtime, tiny_config, train_batch
from rwkv7m.distributed import (
    make_mesh,
    save_orbax_train_state,
)
from rwkv7m.model.nnx_conversion import load_linen_params_into_nnx
from rwkv7m.model.nnx_model import (
    NNXScreenedRWKVModel,
    NNXShardingConfig,
    _apply_norm_in_float32,
    initialize_nnx_model,
)
from rwkv7m.model.screened_rwkv import (
    create_model_variables,
    cross_entropy_components,
    cross_entropy_loss,
)
from rwkv7m.model.state import init_rwkv_state, init_screen_state


def _screened_config(*, dtype="float32"):
    config = tiny_config(
        vocab_size=32,
        d_model=16,
        n_layers=2,
        n_heads=2,
        head_size=8,
    )
    config.dtype = dtype
    config.screening.use_write_screening = True
    return config


def _states(config):
    return (
        init_rwkv_state(1, config),
        init_screen_state(1, config.screening),
    )


def _bfloat16_config():
    config = tiny_config(
        vocab_size=16,
        d_model=8,
        n_layers=1,
        n_heads=1,
        head_size=8,
        use_screening=False,
    )
    config.dtype = "bfloat16"
    config.param_dtype = "bfloat16"
    config.lm_head_init = "variance_scaled"
    return config


def _nnx_flat(state):
    return {
        "/".join(str(part) for part in path): value[...]
        for path, value in nnx.to_flat_state(state)
    }


def test_nnx_full_model_matches_linen_reference_for_read_and_write():
    config = _screened_config()
    linen_variables, linen_model = create_model_variables(
        jax.random.key(0), config, 1
    )
    nnx_model = NNXScreenedRWKVModel(config, rngs=nnx.Rngs(1))
    load_linen_params_into_nnx(nnx_model, linen_variables["params"])
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)

    for phase in ("read_screening_only", "read_write"):
        rwkv_state, screen_state = _states(config)
        expected = linen_model.apply(
            linen_variables,
            input_ids,
            rwkv_state,
            screen_state,
            phase=phase,
        )
        actual = nnx_model(
            input_ids,
            rwkv_state,
            screen_state,
            phase=phase,
        )
        hidden, split_rwkv, split_screen, split_stats = (
            nnx_model.compute_recurrent_hidden(
                input_ids,
                rwkv_state,
                screen_state,
                phase=phase,
            )
        )
        split_logits = nnx_model.compute_logits(hidden)
        assert jnp.allclose(split_logits, actual[0], rtol=1e-5, atol=1e-6)
        assert all(
            jax.tree.leaves(
                jax.tree.map(
                    lambda left, right: jnp.allclose(
                        left, right, rtol=1e-5, atol=1e-6
                    ),
                    (split_rwkv, split_screen),
                    actual[1:3],
                )
            )
        )
        assert set(split_stats) == set(actual[3])
        for key in split_stats:
            assert jnp.allclose(split_stats[key], actual[3][key])
        assert jnp.allclose(actual[0], expected[0], rtol=1e-5, atol=1e-6)
        assert all(
            jax.tree.leaves(
                jax.tree.map(
                    lambda left, right: jnp.allclose(
                        left, right, rtol=1e-5, atol=1e-6
                    ),
                    actual[1:3],
                    expected[1:3],
                )
            )
        )
        assert set(actual[3]) == set(expected[3])
        for key in actual[3]:
            assert jnp.allclose(actual[3][key], expected[3][key])


def test_nnx_full_model_gradient_matches_linen_reference():
    config = _screened_config()
    linen_variables, linen_model = create_model_variables(
        jax.random.key(2), config, 1
    )
    nnx_model = NNXScreenedRWKVModel(config, rngs=nnx.Rngs(3))
    load_linen_params_into_nnx(nnx_model, linen_variables["params"])
    input_ids = jnp.asarray([[1, 2]], dtype=jnp.int32)
    targets = jnp.asarray([[2, 3]], dtype=jnp.int32)
    rwkv_state, screen_state = _states(config)

    def linen_loss(params):
        logits, *_ = linen_model.apply(
            {"params": params},
            input_ids,
            rwkv_state,
            screen_state,
            phase="read_write",
        )
        return cross_entropy_loss(logits, targets)

    graphdef, nnx_params = nnx.split(nnx_model, nnx.Param)

    def nnx_loss(params):
        logits, *_ = nnx.merge(graphdef, params)(
            input_ids,
            rwkv_state,
            screen_state,
            phase="read_write",
        )
        return cross_entropy_loss(logits, targets)

    linen_grads = traverse_util.flatten_dict(
        jax.grad(linen_loss)(linen_variables["params"]), sep="/"
    )
    nnx_grads = _nnx_flat(jax.grad(nnx_loss)(nnx_params))
    assert set(nnx_grads) == set(linen_grads)
    for path, expected in linen_grads.items():
        assert jnp.allclose(
            nnx_grads[path], expected, rtol=3e-5, atol=3e-6
        ), path


@pytest.mark.parametrize("param_dtype", ["bfloat16", "float32"])
def test_bfloat16_compute_boundaries_keep_state_and_loss_statistics_in_fp32(
    param_dtype,
):
    config = _bfloat16_config()
    config.param_dtype = param_dtype
    model = NNXScreenedRWKVModel(config, rngs=nnx.Rngs(10))
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    targets = jnp.asarray([[2, 3, 4]], dtype=jnp.int32)
    rwkv_state, screen_state = _states(config)

    logits, new_rwkv_state, _, _ = model(
        input_ids,
        rwkv_state,
        screen_state,
    )
    assert logits.dtype == jnp.bfloat16
    assert new_rwkv_state[0].time_mix_x.dtype == jnp.float32
    assert new_rwkv_state[0].channel_mix_x.dtype == jnp.float32
    assert new_rwkv_state[0].wkv.dtype == jnp.float32

    block = model.layer_0.rwkv_block_0
    hidden = model.token_embedding(input_ids).astype(jnp.bfloat16)
    normalized = _apply_norm_in_float32(
        block.ln1,
        hidden,
        output_dtype=jnp.bfloat16,
    )
    assert normalized.dtype == jnp.bfloat16
    time_mix_output, _, _, wkv = block.att(
        normalized,
        jnp.zeros_like(normalized),
        rwkv_state[0],
    )
    assert time_mix_output.dtype == jnp.bfloat16
    assert wkv.dtype == jnp.float32
    projection_output = block.att.output(hidden.astype(jnp.float32))
    assert projection_output.dtype == jnp.bfloat16

    ce_total, ce_count = cross_entropy_components(
        logits,
        targets,
        jnp.ones_like(targets, dtype=jnp.bfloat16),
    )
    loss = cross_entropy_loss(logits, targets)
    assert ce_total.dtype == jnp.float32
    assert ce_count.dtype == jnp.float32
    assert loss.dtype == jnp.float32
    recurrent_hidden, *_ = model.compute_recurrent_hidden(
        input_ids,
        rwkv_state,
        screen_state,
    )
    training_components = model.compute_training_loss(
        recurrent_hidden,
        targets,
    )
    assert set(training_components) == {
        "ce_total",
        "ce_count",
        "l2_total",
        "l2_count",
    }
    assert all(
        value.dtype == jnp.float32 for value in training_components.values()
    )
    assert jnp.allclose(
        loss,
        training_components["ce_total"]
        / jnp.maximum(training_components["ce_count"], 1.0),
    )
    expected_loss = -jnp.mean(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1),
            targets[..., None],
            axis=-1,
        )
    )
    assert jnp.allclose(loss, expected_loss)
    logits_grad = jax.grad(cross_entropy_loss)(logits, targets)
    assert jnp.all(jnp.isfinite(logits_grad))

    graphdef, params = nnx.split(model, nnx.Param)

    def model_loss(active_params):
        active_model = nnx.merge(graphdef, active_params)
        active_logits, *_ = active_model(
            input_ids,
            rwkv_state,
            screen_state,
        )
        return cross_entropy_loss(active_logits, targets)

    model_grads = jax.grad(model_loss)(params)
    assert all(
        bool(jnp.all(jnp.isfinite(value)))
        for value in jax.tree.leaves(model_grads)
    )


def test_explicit_nnx_forward_matches_linen_reference():
    config = _screened_config()
    config.lm_head_init = "variance_scaled"
    linen_variables, linen_model = create_model_variables(
        jax.random.key(20), config, 1
    )
    mesh = make_mesh(
        ("data", "model"),
        axis_sizes=(1, 1),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
        ),
    )
    sharding = NNXShardingConfig(mesh)
    nnx_model = initialize_nnx_model(
        jax.random.key(21),
        config,
        sharding=sharding,
    )
    load_linen_params_into_nnx(nnx_model, linen_variables["params"])
    input_ids = jnp.asarray([[1, 2]], dtype=jnp.int32)
    rwkv_state, screen_state = _states(config)
    expected = linen_model.apply(
        linen_variables,
        input_ids,
        rwkv_state,
        screen_state,
        phase="read_write",
    )
    actual = nnx_model(
        jax.device_put(input_ids, sharding.named("data", None)),
        rwkv_state,
        screen_state,
        phase="read_write",
    )
    assert jnp.allclose(actual[0], expected[0], rtol=1e-5, atol=1e-6)
    assert all(
        jax.tree.leaves(
            jax.tree.map(
                lambda left, right: jnp.allclose(
                    left, right, rtol=1e-5, atol=1e-6
                ),
                actual[1:3],
                expected[1:3],
            )
        )
    )


def test_nnx_sharded_init_optimizer_orbax_restore_and_next_step(tmp_path):
    config = tiny_config(
        vocab_size=16,
        d_model=8,
        n_layers=1,
        n_heads=1,
        head_size=8,
        use_screening=False,
    )
    config.lm_head_init = "variance_scaled"
    mesh = make_mesh(("data", "model"), axis_sizes=(1, 1))
    sharding = NNXShardingConfig(mesh)
    runtime, train_state = create_train_runtime(
        jax.random.key(4),
        config,
        batch_size=1,
        total_steps=3,
        sharding=sharding,
    )
    parameter_metadata = {
        "/".join(path): variable.get_metadata().get("out_sharding")
        for path, variable in nnx.to_flat_state(
            nnx.state(train_state.model, nnx.Param)
        )
    }
    assert parameter_metadata["token_embedding/embedding"] == (None, "model")
    assert parameter_metadata["lm_head/kernel"] == ("model", None)

    batch = {
        "input_ids": jnp.asarray([[1, 2]], dtype=jnp.int32),
        "target_ids": jnp.asarray([[2, 3]], dtype=jnp.int32),
        "mask": jnp.ones((1, 2), dtype=jnp.float32),
    }
    train_state, _ = train_batch(train_state, batch, runtime)
    save_orbax_train_state(tmp_path, train_state)

    restored_runtime, template = create_train_runtime(
        jax.random.key(99),
        config,
        batch_size=1,
        total_steps=3,
        sharding=sharding,
    )
    from rwkv7m.distributed.checkpoint import load_orbax_train_state

    restored = load_orbax_train_state(tmp_path, template)
    assert int(restored.step) == 1
    expected_params = traverse_util.flatten_dict(train_state.params, sep="/")
    actual_params = traverse_util.flatten_dict(restored.params, sep="/")
    assert set(expected_params) == set(actual_params)
    for path in expected_params:
        assert jnp.allclose(actual_params[path], expected_params[path]), path
    restored, metrics = train_batch(restored, batch, restored_runtime)
    assert int(restored.step) == 2
    assert jnp.isfinite(metrics["loss"])
