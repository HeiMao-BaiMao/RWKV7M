import copy
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from rwkv7m import (
    create_train_runtime,
    load_model_config,
    model_config_to_dict,
    tiny_config,
)
from rwkv7m.cli.train_binidx import build_config
from rwkv7m.cli.train_binidx_distributed import parse_args
from rwkv7m.distributed import DTypePolicy, abstract_parameter_summary
from rwkv7m.train.train_step import train_step
from rwkv7m.train.nnx_train import (
    _prepare_gradient_update,
    _tree_gradient_statistics,
)


def _training_config():
    config = tiny_config(
        vocab_size=16,
        d_model=8,
        n_layers=1,
        n_heads=1,
        head_size=8,
        use_screening=False,
    )
    config.lm_head_init = "variance_scaled"
    return config


def _batch(batch_size=4, tokens=4):
    input_ids = (
        jnp.arange(batch_size * tokens, dtype=jnp.int32)
        .reshape(batch_size, tokens)
        % 16
    )
    return {
        "input_ids": input_ids,
        "target_ids": (input_ids + 1) % 16,
        "mask": jnp.ones_like(input_ids, dtype=jnp.float32),
    }


def _parameter_values(model):
    return {
        path: variable.get_value()
        for path, variable in nnx.to_flat_state(nnx.state(model, nnx.Param))
    }


def _run_step(config, *, accumulation_steps, batch=None):
    runtime, state = create_train_runtime(
        jax.random.key(0),
        config,
        batch_size=4,
        total_steps=2,
    )
    state, rwkv_state, screen_state, metrics = train_step(
        state,
        _batch() if batch is None else batch,
        runtime.initial_rwkv_state,
        runtime.initial_screen_state,
        gradient_accumulation_steps=accumulation_steps,
    )
    jax.block_until_ready(state.ready_state())
    return state, rwkv_state, screen_state, metrics


def test_public_runtime_accepts_the_same_json_contract_for_bf16_model(tmp_path):
    config = _training_config()
    config.param_dtype = "bfloat16"
    config.param_update_dtype = "float32"
    config.optimizer_state_dtype = "float32"
    config.gradient_accum_dtype = "float32"
    config_path = tmp_path / "model.json"
    config_path.write_text(
        json.dumps(model_config_to_dict(config)),
        encoding="utf-8",
    )

    runtime, state = create_train_runtime(
        jax.random.key(1),
        config_path,
        batch_size=2,
        total_steps=2,
    )

    assert runtime.config == load_model_config(config_path)
    assert {
        str(variable.get_value().dtype)
        for _, variable in nnx.to_flat_state(nnx.state(state.model, nnx.Param))
    } == {"bfloat16"}
    optimizer_float_dtypes = {
        str(value.dtype)
        for value in jax.tree.leaves(state.opt_state)
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.inexact)
    }
    assert optimizer_float_dtypes == {"float32"}


def test_exact_sequence_chunking_and_block_remat_match_unchunked_update():
    unchunked = _training_config()
    batch = _batch()
    batch["mask"] = batch["mask"].at[:, 2:].set(0.0)
    expected = _run_step(unchunked, accumulation_steps=1, batch=batch)
    expected_params = _parameter_values(expected[0].model)

    for recurrent_chunk_size, head_chunk_size in ((2, 4), (3, 2)):
        chunked = copy.deepcopy(unchunked)
        chunked.sequence_chunk_size = recurrent_chunk_size
        chunked.head_chunk_size = head_chunk_size
        chunked.remat_blocks = True
        actual = _run_step(chunked, accumulation_steps=1, batch=batch)

        assert jnp.allclose(
            actual[3]["loss"], expected[3]["loss"], rtol=1e-5, atol=1e-6
        )
        actual_params = _parameter_values(actual[0].model)
        for path, expected_value in expected_params.items():
            assert jnp.allclose(
                actual_params[path],
                expected_value,
                rtol=2e-5,
                atol=2e-6,
            ), (recurrent_chunk_size, head_chunk_size, path)


def test_microbatch_accumulation_matches_full_batch_optimizer_update():
    config = _training_config()
    batch = _batch()
    batch["mask"] = batch["mask"].at[2, 1:].set(0.0)
    batch["mask"] = batch["mask"].at[3, :].set(0.0)
    full_batch = _run_step(config, accumulation_steps=1, batch=batch)
    accumulated = _run_step(
        copy.deepcopy(config), accumulation_steps=2, batch=batch
    )

    assert jnp.allclose(
        accumulated[3]["total_loss"],
        full_batch[3]["total_loss"],
        rtol=1e-5,
        atol=1e-6,
    )
    accumulated_params = _parameter_values(accumulated[0].model)
    for path, expected_value in _parameter_values(full_batch[0].model).items():
        assert jnp.allclose(
            accumulated_params[path],
            expected_value,
            rtol=2e-5,
            atol=2e-6,
        ), path


def test_stable_gradient_statistics_do_not_overflow_for_finite_spike():
    gradients = {
        "a": jnp.asarray([1e22, -1e22], dtype=jnp.float32),
        "b": jnp.asarray([3.0], dtype=jnp.float32),
    }
    all_finite, global_norm, max_abs = _tree_gradient_statistics(gradients)
    assert all_finite == 1.0
    assert jnp.isfinite(global_norm)
    assert jnp.allclose(global_norm / 1e22, jnp.sqrt(2.0), rtol=1e-5)
    assert max_abs == jnp.asarray(1e22, dtype=jnp.float32)


def test_gradient_spike_guard_zeroes_update_before_optimizer():
    gradients = {
        "a": jnp.asarray([3.0, 4.0], dtype=jnp.float32),
    }
    (
        prepared,
        all_finite,
        global_norm,
        max_abs,
        clip_scale,
        spike_detected,
        update_applied,
    ) = _prepare_gradient_update(
        gradients,
        max_grad_norm=1.0,
        spike_max_abs=3.5,
    )
    assert all_finite == 1.0
    assert global_norm == 5.0
    assert max_abs == 4.0
    assert clip_scale == 0.0
    assert spike_detected == 1.0
    assert update_applied == 0.0
    assert jnp.array_equal(prepared["a"], jnp.zeros((2,), jnp.float32))


def test_nnx_train_step_skips_complete_optimizer_update_on_spike():
    config = _training_config()
    config.gradient_spike_max_abs = 1e-20
    runtime, state = create_train_runtime(
        jax.random.key(0),
        config,
        batch_size=4,
        total_steps=2,
    )
    before = {
        path: jax.device_get(value).copy()
        for path, value in _parameter_values(state.model).items()
    }
    before_optimizer = [
        jax.device_get(value).copy()
        for value in jax.tree.leaves(state.opt_state)
        if hasattr(value, "dtype")
    ]
    state, _, _, metrics = train_step(
        state,
        _batch(),
        runtime.initial_rwkv_state,
        runtime.initial_screen_state,
    )
    jax.block_until_ready(state.ready_state())
    assert metrics["gradient_spike_detected"] == 1.0
    assert metrics["gradient_update_applied"] == 0.0
    assert state.step == 0
    after = _parameter_values(state.model)
    for path, expected in before.items():
        assert jnp.array_equal(after[path], expected), path
    after_optimizer = [
        value
        for value in jax.tree.leaves(state.opt_state)
        if hasattr(value, "dtype")
    ]
    assert len(after_optimizer) == len(before_optimizer)
    for actual, expected in zip(
        after_optimizer,
        before_optimizer,
        strict=True,
    ):
        assert jnp.array_equal(actual, expected)


def test_tracked_7b_config_is_shared_by_planner_and_distributed_cli():
    small_config = load_model_config("configs/rwkv7m-small.json.example")
    assert small_config.d_model == 128
    assert small_config.vocab_parallel is False

    config_path = Path("configs/rwkv7m-7b-tpu.json.example")
    config = load_model_config(config_path)
    args = parse_args(
        [
            "--data-file",
            "unused",
            "--model-config",
            str(config_path),
            "--ctx-len",
            "4096",
            "--global-batch-size",
            "1",
            "--steps",
            "0",
        ]
    )

    assert build_config(args) == config
    assert DTypePolicy.from_config(config).param_storage_dtype == "bfloat16"
    summary = abstract_parameter_summary(config)
    assert summary.total == 6_994_788_376
    assert config.vocab_parallel is True
    assert config.remat_blocks is True
    assert config.sequence_chunk_size == 256

    run_args = parse_args(
        ["--config", "configs/rwkv7m-7b-tpu-train.json.example"]
    )
    assert build_config(run_args) == config
    assert run_args.gradient_accumulation_steps == 8
    assert run_args.mesh_axis_names == ["data", "model"]


def test_preset_execution_controls_are_explicit_opt_in_overrides():
    base_args = [
        "--data-file",
        "unused",
        "--model-preset",
        "0.185b",
        "--ctx-len",
        "512",
        "--global-batch-size",
        "1",
        "--steps",
        "0",
    ]

    preset_config = build_config(parse_args(base_args))
    assert preset_config.remat_blocks is True
    assert preset_config.sequence_chunk_size == 128
    assert preset_config.head_chunk_size is None

    unchunked_config = build_config(
        parse_args(
            base_args
            + [
                "--no-remat-blocks",
                "--no-sequence-chunking",
                "--no-head-chunking",
            ]
        )
    )
    assert unchunked_config.remat_blocks is False
    assert unchunked_config.sequence_chunk_size is None
    assert unchunked_config.head_chunk_size == 512

    comparison_config = build_config(
        parse_args(
            base_args
            + [
                "--remat-blocks",
                "--sequence-chunk-size",
                "256",
                "--head-chunk-size",
                "512",
            ]
        )
    )
    assert comparison_config.remat_blocks is True
    assert comparison_config.sequence_chunk_size == 256
    assert comparison_config.head_chunk_size == 512
