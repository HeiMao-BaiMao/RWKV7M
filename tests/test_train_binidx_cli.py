import json
from pathlib import Path

import numpy as np
import pytest

from rwkv7m import load_train_checkpoint_metadata
from rwkv7m.cli.train_binidx import build_config, parse_args, run_training
from rwkv7m.data import MMapIndexedDatasetBuilder, data_file_path, index_file_path


def write_train_binidx(tmp_path):
    prefix = str(tmp_path / "train")
    builder = MMapIndexedDatasetBuilder(data_file_path(prefix), dtype=np.uint16)
    builder.add_item((np.arange(257, dtype=np.uint16) % 64).astype(np.uint16))
    builder.end_document()
    builder.finalize(index_file_path(prefix))
    return prefix


def base_args(prefix, output_dir, extra=None):
    args = [
        "--data-file",
        prefix,
        "--ctx-len",
        "4",
        "--batch-size",
        "1",
        "--steps",
        "1",
        "--vocab-size",
        "64",
        "--d-model",
        "32",
        "--d-ffn",
        "64",
        "--n-layers",
        "2",
        "--n-heads",
        "2",
        "--head-size",
        "16",
        "--d-slot",
        "16",
        "--d-k",
        "16",
        "--d-v",
        "16",
        "--output-dir",
        str(output_dir),
        "--save-every",
        "1",
        "--eval-every",
        "1",
        "--eval-steps",
        "1",
        "--print-every",
        "0",
    ]
    if extra:
        args.extend(extra)
    return parse_args(args)


def test_train_binidx_cli_saves_and_resumes(tmp_path):
    prefix = write_train_binidx(tmp_path)
    output_dir = tmp_path / "out"

    state, _, checkpoint = run_training(base_args(prefix, output_dir))
    assert int(state.step) == 1
    assert checkpoint == output_dir / "ckpt-00000001"
    assert (checkpoint / "checkpoint.json").exists()
    assert (checkpoint / "model.safetensors").exists()

    _, payload = load_train_checkpoint_metadata(checkpoint)
    assert payload["step"] == 1
    assert payload["dataset_position"] == {"step": 1}

    resumed_state, _, resumed_checkpoint = run_training(
        base_args(
            prefix,
            output_dir,
            ["--resume", str(checkpoint), "--remat-blocks"],
        )
    )
    assert int(resumed_state.step) == 2
    assert resumed_state.model.config.remat_blocks is True
    assert resumed_checkpoint == output_dir / "ckpt-00000002"
    resumed_config, _ = load_train_checkpoint_metadata(resumed_checkpoint)
    assert resumed_config.remat_blocks is True


def test_train_binidx_cli_carry_state_requires_sequential_sampling(tmp_path):
    prefix = write_train_binidx(tmp_path)
    output_dir = tmp_path / "out"
    args = base_args(prefix, output_dir, ["--carry-state"])

    try:
        run_training(args)
    except ValueError as exc:
        assert "--carry-state requires --sampling-mode sequential" in str(exc)
    else:
        raise AssertionError("carry-state with magic sampling should fail")


def test_train_binidx_cli_carry_state_saves_runtime_state(tmp_path):
    prefix = write_train_binidx(tmp_path)
    output_dir = tmp_path / "out"
    args = base_args(
        prefix,
        output_dir,
        [
            "--carry-state",
            "--sampling-mode",
            "sequential",
            "--eval-every",
            "0",
        ],
    )

    state, _, checkpoint = run_training(args)

    assert int(state.step) == 1
    assert (checkpoint / "runtime_state.msgpack").exists()
    _, payload = load_train_checkpoint_metadata(checkpoint)
    assert payload["metadata"]["carry_state"] is True
    assert payload["metadata"]["sampling_mode"] == "sequential"


def test_model_config_allows_experiment_only_recovery_overrides(tmp_path):
    prefix = write_train_binidx(tmp_path)
    args = parse_args(
        [
            "--data-file",
            prefix,
            "--ctx-len",
            "4",
            "--model-config",
            "configs/rwkv7m-0.185b-screening-v5-core.json.example",
            "--warmup-steps",
            "10",
            "--lr-init",
            "0.0001",
            "--lr-final",
            "0.0001",
            "--lr-schedule",
            "rwkv",
            "--max-grad-norm",
            "0.5",
            "--gradient-spike-max-abs",
            "250000",
            "--weight-decay",
            "0.0",
            "--screening-activation-step",
            "0",
            "--screening-activation-warmup-steps",
            "0",
            "--screening-optimizer-lr-multiplier",
            "1.0",
            "--screening-write-budget-min-slot-utilization",
            "0.0",
            "--no-screening",
        ]
    )
    config = build_config(args)
    assert config.warmup_steps == 10
    assert config.lr_init == 1e-4
    assert config.lr_final == 1e-4
    assert config.lr_schedule == "rwkv"
    assert config.max_grad_norm == 0.5
    assert config.gradient_spike_max_abs == 250000.0
    assert config.weight_decay == 0.0
    assert config.screening.activation_step == 0
    assert config.screening.activation_warmup_steps == 0
    assert config.screening.optimizer_lr_multiplier == 1.0
    assert config.screening.write_budget_min_slot_utilization == 0.0
    assert config.use_screening is False


def test_model_config_preserves_optimizer_settings_without_cli_overrides(tmp_path):
    payload = json.loads(
        Path(
            "configs/rwkv7m-0.185b-screening-v5-core.json.example"
        ).read_text(encoding="utf-8")
    )
    payload.update(
        {
            "lr_init": 3e-4,
            "lr_final": 2e-5,
            "lr_schedule": "rwkv",
            "max_grad_norm": 0.75,
            "gradient_spike_max_abs": 123456.0,
            "weight_decay": 0.002,
        }
    )
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = build_config(
        parse_args(
            [
                "--data-file",
                "unused",
                "--ctx-len",
                "512",
                "--model-config",
                str(path),
            ]
        )
    )
    assert config.lr_init == 3e-4
    assert config.lr_final == 2e-5
    assert config.lr_schedule == "rwkv"
    assert config.max_grad_norm == 0.75
    assert config.gradient_spike_max_abs == 123456.0
    assert config.weight_decay == 0.002


def test_model_config_preserves_admission_controller_settings(tmp_path):
    config = build_config(
        parse_args(
            [
                "--data-file",
                "unused",
                "--ctx-len",
                "512",
                "--model-config",
                "configs/rwkv7m-0.185b-screening-v5-core.json.example",
                "--screening-admission-controller",
                "--no-screening-admission-quota",
                "--screening-admission-controller-target",
                "0.075",
                "--screening-admission-controller-kp",
                "0.2",
                "--screening-admission-controller-ki",
                "0.03",
                "--screening-detach-inputs-steps",
                "300",
            ]
        )
    )
    assert config.screening.admission_controller_enabled is True
    assert config.screening.admission_controller_target == 0.075
    assert config.screening.admission_controller_kp == 0.2
    assert config.screening.admission_controller_ki == 0.03
    assert config.screening.detach_screening_inputs_steps == 300


def test_model_config_rejects_invalid_gradient_spike_override():
    with pytest.raises(ValueError, match="gradient_spike_max_abs"):
        build_config(
            parse_args(
                [
                    "--data-file",
                    "unused",
                    "--ctx-len",
                    "512",
                    "--model-config",
                    "configs/rwkv7m-0.185b-screening-v5-core.json.example",
                    "--gradient-spike-max-abs",
                    "0",
                ]
            )
        )


def test_model_config_honors_explicit_screening_curriculum_overrides(tmp_path):
    payload = json.loads(
        Path(
            "configs/rwkv7m-0.185b-screening-v5-core.json.example"
        ).read_text(encoding="utf-8")
    )
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = build_config(
        parse_args(
            [
                "--data-file",
                "unused",
                "--ctx-len",
                "512",
                "--model-config",
                str(path),
                "--screening-activation-step",
                "300",
                "--screening-admission-floor-weight",
                "0",
                "--screening-write-budget-weight",
                "0",
                "--screening-self-index-loss-weight",
                "0",
            ]
        )
    )
    assert config.screening.activation_step == 300
    assert config.screening.admission_floor_weight == 0.0
    assert config.screening.write_budget_weight == 0.0
    assert config.screening.self_index_loss_weight == 0.0


def test_train_cli_requires_finite_metrics_by_default():
    args = parse_args(["--data-file", "unused", "--ctx-len", "4"])
    assert args.require_finite is True
    args = parse_args(
        [
            "--data-file",
            "unused",
            "--ctx-len",
            "4",
            "--no-require-finite",
        ]
    )
    assert args.require_finite is False
