import numpy as np

from rwkv7m import load_train_checkpoint_metadata
from rwkv7m.cli.train_binidx import parse_args, run_training
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
