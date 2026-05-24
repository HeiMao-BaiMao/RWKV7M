import numpy as np

from rwkv7m import load_train_checkpoint_metadata
from rwkv7m.cli.train_binidx_distributed import parse_args, run_distributed_training
from rwkv7m.data import MMapIndexedDatasetBuilder, data_file_path, index_file_path


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

    resumed, resumed_checkpoint = run_distributed_training(
        parse_args([*base, "--resume", str(checkpoint)])
    )
    assert int(resumed.train_state.step) == 2
    assert resumed_checkpoint == output_dir / "ckpt-00000002"
