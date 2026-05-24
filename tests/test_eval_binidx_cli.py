import numpy as np

from rwkv7m.data import MMapIndexedDatasetBuilder, data_file_path, index_file_path
from rwkv7m.cli.eval_binidx import evaluate_binidx, parse_args


def write_eval_binidx(tmp_path):
    prefix = str(tmp_path / "eval")
    builder = MMapIndexedDatasetBuilder(data_file_path(prefix), dtype=np.uint16)
    builder.add_item((np.arange(257, dtype=np.uint16) % 64).astype(np.uint16))
    builder.end_document()
    builder.finalize(index_file_path(prefix))
    return prefix


def test_eval_binidx_runs_one_step(tmp_path):
    prefix = write_eval_binidx(tmp_path)
    args = parse_args(
        [
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
            "--print-every",
            "0",
        ]
    )
    metrics = evaluate_binidx(args)
    assert metrics["steps"] == 1
    assert metrics["tokens"] == 4
    assert metrics["loss"] > 0
    assert metrics["perplexity"] > 1
