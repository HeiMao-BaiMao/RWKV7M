import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from rwkv7m import (
    load_train_checkpoint_metadata,
    model_config_to_dict,
    tiny_config,
)
from rwkv7m.cli.train_binidx_distributed import parse_args, run_distributed_training
from rwkv7m.data import MMapIndexedDatasetBuilder, data_file_path, index_file_path
from rwkv7m.distributed import load_distributed_checkpoint_metadata


def write_dp_train_data(tmp_path):
    prefix = str(tmp_path / "dp-train")
    builder = MMapIndexedDatasetBuilder(data_file_path(prefix), dtype=np.uint16)
    builder.add_item((np.arange(257, dtype=np.uint16) % 32).astype(np.uint16))
    builder.end_document()
    builder.finalize(index_file_path(prefix))
    return prefix


def test_distributed_binidx_training_cli_runs_one_step(tmp_path):
    prefix = write_dp_train_data(tmp_path)
    args = parse_args(
        [
            "--data-file",
            prefix,
            "--ctx-len",
            "4",
            "--global-batch-size",
            "1",
            "--steps",
            "1",
            "--vocab-size",
            "32",
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
            "--print-every",
            "0",
        ]
    )
    dist, _ = run_distributed_training(args)
    assert int(dist.train_state.step) == 1


def test_distributed_binidx_training_cli_carry_state_requires_sequential_sampling(tmp_path):
    prefix = write_dp_train_data(tmp_path)
    args = parse_args(
        [
            "--data-file",
            prefix,
            "--ctx-len",
            "4",
            "--global-batch-size",
            "1",
            "--steps",
            "1",
            "--vocab-size",
            "32",
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
            "--carry-state",
            "--print-every",
            "0",
        ]
    )

    try:
        run_distributed_training(args)
    except ValueError as exc:
        assert "--carry-state requires --sampling-mode sequential" in str(exc)
    else:
        raise AssertionError("carry-state with magic sampling should fail")


def test_distributed_binidx_training_cli_saves_and_resumes(tmp_path):
    prefix = write_dp_train_data(tmp_path)
    output_dir = tmp_path / "out"
    base = [
        "--data-file",
        prefix,
        "--ctx-len",
        "4",
        "--global-batch-size",
        "1",
        "--steps",
        "1",
        "--vocab-size",
        "32",
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
        "--print-every",
        "0",
    ]
    dist, checkpoint = run_distributed_training(parse_args(base))
    assert int(dist.train_state.step) == 1
    assert checkpoint == output_dir / "ckpt-00000001"

    _, payload = load_train_checkpoint_metadata(checkpoint)
    assert payload["metadata"]["distributed"] is True
    assert payload["metadata"]["global_batch_size"] == 1
    assert payload["metadata"]["local_device_count"] >= 1
    assert payload["metadata"]["device_count"] >= 1

    checkpoint_payload = load_distributed_checkpoint_metadata(checkpoint)
    assert checkpoint_payload.start_step == 1
    assert checkpoint_payload.dataset_position == {"step": 1}

    resumed, resumed_checkpoint = run_distributed_training(
        parse_args([*base, "--resume", str(checkpoint)])
    )
    assert int(resumed.train_state.step) == 2
    assert resumed_checkpoint == output_dir / "ckpt-00000002"


def test_distributed_binidx_training_cli_carry_state_saves_and_restores_runtime_state(tmp_path):
    prefix = write_dp_train_data(tmp_path)
    output_dir = tmp_path / "carry"
    first_args = [
        "--data-file",
        prefix,
        "--ctx-len",
        "4",
        "--global-batch-size",
        "1",
        "--steps",
        "2",
        "--vocab-size",
        "32",
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
        "2",
        "--sampling-mode",
        "sequential",
        "--carry-state",
        "--print-every",
        "0",
    ]
    dist, checkpoint = run_distributed_training(parse_args(first_args))
    assert int(dist.train_state.step) == 2
    assert checkpoint == output_dir / "ckpt-00000002"
    assert (checkpoint / "runtime_state.msgpack").exists()

    _, payload = load_train_checkpoint_metadata(checkpoint)
    assert payload["metadata"]["carry_state"] is True
    assert payload["metadata"]["runtime_state"] is True

    resume_args = [
        "--data-file",
        prefix,
        "--ctx-len",
        "4",
        "--global-batch-size",
        "1",
        "--steps",
        "0",
        "--vocab-size",
        "32",
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
        "0",
        "--sampling-mode",
        "sequential",
        "--carry-state",
        "--resume",
        str(checkpoint),
        "--print-every",
        "0",
    ]
    resumed, _ = run_distributed_training(parse_args(resume_args))
    assert int(resumed.train_state.step) == 2
    assert not jnp.allclose(
        resumed.rwkv_state[0].time_mix_x,
        resumed.initial_rwkv_state[0].time_mix_x,
    )


def test_distributed_binidx_training_cli_logs_eval_and_rotates(tmp_path):
    prefix = write_dp_train_data(tmp_path)
    output_dir = tmp_path / "logged"
    args = parse_args(
        [
            "--data-file",
            prefix,
            "--ctx-len",
            "4",
            "--global-batch-size",
            "1",
            "--steps",
            "3",
            "--vocab-size",
            "32",
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
            "--keep-last-checkpoints",
            "1",
            "--eval-every",
            "2",
            "--eval-steps",
            "1",
            "--summary-every",
            "1",
            "--save-best-checkpoint",
            "--param-axis-name",
            "data",
            "--print-every",
            "0",
        ]
    )
    dist, checkpoint = run_distributed_training(args)
    assert int(dist.train_state.step) == 3
    assert checkpoint == output_dir / "ckpt-00000003"
    assert not (output_dir / "ckpt-00000001").exists()
    assert (output_dir / "ckpt-00000002").exists()
    assert (output_dir / "ckpt-00000003").exists()
    run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["args"]["global_batch_size"] == 1
    assert run_config["model_config"]["vocab_size"] == 32
    assert run_config["environment"]["packages"]["jax"]
    assert run_config["environment"]["packages"]["flax"]
    assert "token_embedding/embedding" in run_config["parameter_partition_summary"]

    jsonl_path = output_dir / "metrics.jsonl"
    csv_path = output_dir / "metrics.csv"
    assert jsonl_path.exists()
    assert csv_path.exists()
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["split"] for record in records].count("train") == 3
    assert any(record["split"] == "eval" for record in records)

    summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["current_step"] == 3
    assert summary["completed_steps"] == 3
    assert summary["tokens_per_step"] == 4
    assert summary["tokens_seen"] == 12
    assert summary["latest_checkpoint"].endswith("ckpt-00000003")
    assert summary["best_eval"]["step"] == 2
    assert summary["best_eval"]["metric"] == "loss"
    assert summary["best_eval"]["checkpoint"].endswith("ckpt-00000002")

    best_eval = json.loads((output_dir / "best_eval.json").read_text(encoding="utf-8"))
    assert best_eval["step"] == 2
    assert best_eval["checkpoint"].endswith("ckpt-00000002")


def test_distributed_binidx_training_cli_orbax_checkpoint_roundtrip(tmp_path):
    prefix = write_dp_train_data(tmp_path)
    output_dir = tmp_path / "orbax"
    base = [
        "--data-file",
        prefix,
        "--ctx-len",
        "4",
        "--global-batch-size",
        "1",
        "--steps",
        "1",
        "--vocab-size",
        "32",
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
        "--checkpoint-backend",
        "orbax",
        "--print-every",
        "0",
    ]
    dist, checkpoint = run_distributed_training(parse_args(base))
    assert int(dist.train_state.step) == 1
    assert checkpoint == output_dir / "ckpt-00000001"
    assert (checkpoint / "checkpoint.json").exists()
    assert (checkpoint / "orbax_train_state").exists()

    _, payload = load_train_checkpoint_metadata(checkpoint)
    assert payload["backend"] == "orbax"
    assert payload["metadata"]["distributed"] is True

    resumed, resumed_checkpoint = run_distributed_training(
        parse_args([*base, "--resume", str(checkpoint)])
    )
    assert int(resumed.train_state.step) == 2
    assert resumed_checkpoint == output_dir / "ckpt-00000002"


def test_distributed_binidx_training_cli_orbax_accepts_relative_output_dir(
    tmp_path,
    monkeypatch,
):
    prefix = write_dp_train_data(tmp_path)
    monkeypatch.chdir(tmp_path)
    output_dir = "relative-orbax"
    base = [
        "--data-file",
        prefix,
        "--ctx-len",
        "4",
        "--global-batch-size",
        "1",
        "--steps",
        "1",
        "--vocab-size",
        "32",
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
        output_dir,
        "--save-every",
        "1",
        "--checkpoint-backend",
        "orbax",
        "--print-every",
        "0",
    ]

    dist, checkpoint = run_distributed_training(parse_args(base))

    assert int(dist.train_state.step) == 1
    assert checkpoint == Path(output_dir) / "ckpt-00000001"
    assert (checkpoint / "checkpoint.json").exists()
    assert (checkpoint / "orbax_train_state").exists()

    resumed, resumed_checkpoint = run_distributed_training(
        parse_args([*base, "--resume", str(checkpoint)])
    )

    assert int(resumed.train_state.step) == 2
    assert resumed_checkpoint == Path(output_dir) / "ckpt-00000002"


def test_distributed_cli_uses_shared_model_config_and_microbatches(tmp_path):
    prefix = write_dp_train_data(tmp_path)
    config = tiny_config(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=2,
        head_size=8,
        use_screening=False,
    )
    config.lm_head_init = "variance_scaled"
    config.sequence_chunk_size = 2
    config.remat_blocks = True
    config_path = tmp_path / "model.json"
    config_path.write_text(
        json.dumps(model_config_to_dict(config)),
        encoding="utf-8",
    )
    args = parse_args(
        [
            "--data-file",
            prefix,
            "--model-config",
            str(config_path),
            "--ctx-len",
            "4",
            "--global-batch-size",
            "2",
            "--gradient-accumulation-steps",
            "2",
            "--steps",
            "1",
            "--print-every",
            "0",
        ]
    )

    dist, _ = run_distributed_training(args)

    assert int(dist.train_state.step) == 1
    assert dist.train_state.model.config == config
