import importlib.util
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from flax import nnx
import jax.numpy as jnp
import numpy as np
import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COMMON = _load("benchmark_common")
MATRIX = _load("benchmark_gpu_train_matrix")
PROFILE = _load("profile_gpu_train_compute")
COMPARE = _load("compare_compute_benchmarks")
CUDA_RANGE = _load("cuda_profiler_range")
LOCAL = _load("benchmark_local_train_compute")
DIAGNOSE = _load("diagnose_local_train_step")


def test_fixed_batch_content_hash_ignores_container_and_mapping_order(tmp_path):
    first = {
        "input_ids": np.arange(8, dtype=np.int64).reshape(2, 4),
        "target_ids": np.arange(1, 9, dtype=np.int64).reshape(2, 4),
        "mask": np.ones((2, 4), dtype=np.float32),
    }
    second = {key: first[key] for key in reversed(tuple(first))}
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    np.savez(first_path, **first)
    np.savez_compressed(second_path, **second)
    assert first_path.read_bytes() != second_path.read_bytes()
    assert COMMON.fixed_batch_content_sha256(first) == (
        COMMON.fixed_batch_content_sha256(second)
    )


def _record(batch, head, optimizer, throughput, optimizer_ms):
    return {
        "batch_size": batch,
        "head": head,
        "optimizer": optimizer,
        "status": "ok",
        "tokens_per_second": throughput,
        "full_step_median_ms": 1000.0 * batch / throughput,
        "optimizer_median_ms": optimizer_ms,
    }


def test_gpu_matrix_summary_reports_head_optimizer_and_scaling_ratios():
    records = [
        _record(1, "full_xla", "optax", 100.0, 10.0),
        _record(1, "pallas_tiled", "optax", 90.0, 10.0),
        _record(1, "full_xla", "pallas_gpu_triton", 125.0, 5.0),
        _record(1, "pallas_tiled", "pallas_gpu_triton", 110.0, 5.0),
        _record(2, "full_xla", "optax", 180.0, 10.0),
        _record(2, "pallas_tiled", "optax", 160.0, 10.0),
        _record(2, "full_xla", "pallas_gpu_triton", 220.0, 5.0),
        _record(2, "pallas_tiled", "pallas_gpu_triton", 200.0, 5.0),
    ]
    summary = MATRIX.summarize(
        records,
        batch_sizes=(1, 2),
        head_modes=("full_xla", "pallas_tiled"),
        optimizer_backends=("optax", "pallas_gpu_triton"),
    )
    assert summary["head"][0]["tiled_over_full_throughput"] == pytest.approx(0.9)
    assert summary["optimizer"][0]["fused_over_optax_throughput"] == pytest.approx(
        1.25
    )
    scaling = [
        value
        for value in summary["scaling"]
        if value["batch_size"] == 2
        and value["head"] == "full_xla"
        and value["optimizer"] == "optax"
    ][0]
    assert scaling["parallel_efficiency"] == pytest.approx(0.9)


def test_nsight_systems_command_uses_post_warmup_cuda_range(tmp_path):
    benchmark = ["python", "scripts/benchmark_local_train_compute.py"]
    ranged = PROFILE._ensure_profile_range(benchmark)
    command = PROFILE._systems_command("nsys", tmp_path / "trace", ranged)
    assert "--capture-range=cudaProfilerApi" in command
    assert "--capture-range-end=stop" in command
    assert command[-2:] == ["--profile-mode", "cuda_profiler_api"]


def test_cuda_runtime_candidate_search_is_safe_without_cuda_installation():
    candidates = tuple(CUDA_RANGE._cudart_candidates())
    assert len(candidates) == len(set(candidates))


def test_local_benchmark_reports_the_executed_v5_screening_backend():
    config = SimpleNamespace(
        use_screening=True,
        screening=SimpleNamespace(
            semantics_version="screening-v5-core",
            write_mode="competitive_novel",
        ),
    )
    assert LOCAL._screening_execution(config) == (
        "portable_jax_v5",
        "screening-v5-core",
    )
    config.use_screening = False
    assert LOCAL._screening_execution(config) == ("disabled", None)


def test_local_benchmark_finite_gate_checks_every_inexact_leaf():
    assert LOCAL._tree_all_finite(
        {"finite": jnp.asarray([1.0]), "step": jnp.asarray(1)}
    )
    assert not LOCAL._tree_all_finite(
        {"invalid": jnp.asarray([jnp.nan]), "finite": jnp.asarray([1.0])}
    )


def test_gradient_diagnostic_reports_nonfinite_values_and_largest_leaf():
    gradients = nnx.State(
        {
            "finite": jnp.asarray([1.0, -3.0], dtype=jnp.float32),
            "invalid": jnp.asarray([jnp.nan, jnp.inf], dtype=jnp.float32),
        }
    )
    summary = DIAGNOSE.summarize_gradient_state(gradients, top_k=1)
    assert summary["leaf_count"] == 2
    assert summary["nonfinite_leaf_count"] == 1
    assert summary["nonfinite_value_count"] == 2
    assert summary["nonfinite_gradients"][0]["path"] == "invalid"
    assert summary["largest_finite_gradients"][0]["path"] == "finite"


def test_gradient_diagnostic_accepts_checkpoint_path(tmp_path):
    checkpoint = tmp_path / "ckpt-00000007"
    args = DIAGNOSE.parse_args(
        [
            "--fixed-batch",
            str(tmp_path / "batch.npz"),
            "--model-config",
            "config.json",
            "--checkpoint",
            str(checkpoint),
            "--sequence-chunk-size",
            "0",
            "--float32-model",
            "--output",
            str(tmp_path / "diagnostic.json"),
        ]
    )
    assert args.checkpoint == checkpoint
    assert args.sequence_chunk_size == 0
    assert args.float32_model is True


def test_nsight_rejects_conflicting_profile_mode():
    with pytest.raises(SystemExit, match="cuda_profiler_api"):
        PROFILE._ensure_profile_range(
            ["python", "benchmark.py", "--profile-mode", "xprof"]
        )


def _compute_record():
    timing = {"median_ms": 1.0}
    return {
        "schema_version": COMMON.COMPUTE_BENCHMARK_SCHEMA_VERSION,
        "benchmark_kind": "train_compute_only",
        "method": {"warmup": 5, "iterations": 20, "python_gc_disabled": True},
        "devices": [{"device_kind": "NVIDIA L40S"}],
        "shape": {
            "batch": 1,
            "tokens": 512,
            "dtype": "bfloat16",
            "n_layers": 12,
            "d_model": 768,
            "d_ffn": 3072,
            "n_heads": 12,
            "head_size": 64,
            "vocab_size": 65536,
        },
        "fixed_batch": {"sha256": "container-a", "content_sha256": "logical"},
        "timings": {
            "forward": timing,
            "backward": timing,
            "optimizer": timing,
            "full_step": timing,
        },
    }


def test_compute_comparison_uses_content_hash_and_rejects_model_mismatch(tmp_path):
    local = _compute_record()
    official = copy.deepcopy(local)
    official["fixed_batch"]["sha256"] = "container-b"
    local_path = tmp_path / "local.json"
    official_path = tmp_path / "official.json"
    local_path.write_text(json.dumps(local), encoding="utf-8")
    official_path.write_text(json.dumps(official), encoding="utf-8")
    assert COMPARE.main(
        ["--local", str(local_path), "--official", str(official_path)]
    ) == 0
    official["shape"]["d_ffn"] = 2688
    official_path.write_text(json.dumps(official), encoding="utf-8")
    with pytest.raises(SystemExit, match="shape.d_ffn differs"):
        COMPARE.main(
            ["--local", str(local_path), "--official", str(official_path)]
        )
