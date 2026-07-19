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


def test_eval_binidx_reports_memory_off_counterfactual(tmp_path):
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
            "--phase",
            "read_write",
            "--memory-off-counterfactual",
            "--print-every",
            "0",
        ]
    )
    metrics = evaluate_binidx(args)
    assert metrics["memory_off_loss"] > 0.0
    assert np.isfinite(metrics["memory_loss_delta"])
    assert np.isfinite(metrics["prediction_rms_delta"])
    assert metrics["prediction_rms_delta"] >= 0.0


def test_eval_binidx_can_carry_state_on_sequential_sampling(tmp_path):
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
            "2",
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
            "--sampling-mode",
            "sequential",
            "--carry-state",
            "--print-every",
            "0",
        ]
    )
    metrics = evaluate_binidx(args)
    assert metrics["steps"] == 2
    assert metrics["tokens"] == 8
    assert metrics["carry_state"] is True
    assert metrics["sampling_mode"] == "sequential"
    assert metrics["loss"] > 0


def test_eval_binidx_carry_state_requires_sequential_sampling(tmp_path):
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
            "--carry-state",
            "--print-every",
            "0",
        ]
    )
    try:
        evaluate_binidx(args)
    except ValueError as exc:
        assert "--carry-state requires --sampling-mode sequential" in str(exc)
    else:
        raise AssertionError("carry-state eval with magic sampling should fail")
