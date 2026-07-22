import numpy as np

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
        base_args(prefix, output_dir, ["--resume", str(checkpoint)])
    )
    assert int(resumed_state.step) == 2
    assert resumed_checkpoint == output_dir / "ckpt-00000002"


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
    assert config.screening.activation_step == 0
    assert config.screening.activation_warmup_steps == 0
    assert config.screening.optimizer_lr_multiplier == 1.0
    assert config.screening.write_budget_min_slot_utilization == 0.0
    assert config.use_screening is False
