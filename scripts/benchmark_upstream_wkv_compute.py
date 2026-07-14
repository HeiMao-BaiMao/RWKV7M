"""Measure the official RWKV-7 fused CUDA WKV operator in isolation."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from benchmark_common import (
    COMPUTE_BENCHMARK_SCHEMA_VERSION,
    gc_policy,
    timing_summary,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-repo", required=True)
    parser.add_argument("--time", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--kernel", default="@rwkv3")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--disable-python-gc", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ("time", "batch", "heads", "head_size", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.head_size != 64:
        parser.error("official x070 fused CUDA currently requires head size 64")
    if args.time % 16:
        parser.error("--time must be divisible by the official chunk length 16")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


def _measure(function, *, warmup, iterations, torch):
    for _ in range(warmup):
        function()
        torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return timing_summary(samples)


def main(argv=None):
    args = parse_args(argv)
    upstream = Path(args.upstream_repo).resolve()
    model_file = upstream / "src" / "model.py"
    if not model_file.is_file():
        raise SystemExit(f"official model.py not found: {model_file}")
    os.environ.update(
        RWKV_MY_TESTING="x070",
        RWKV_KERNEL=args.kernel,
        RWKV_CTXLEN=str(args.time),
        RWKV_HEAD_SIZE=str(args.head_size),
        RWKV_FLOAT_MODE="bf16",
        RWKV_JIT_ON="1",
    )
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("the official PyTorch environment is required") from exc
    if not torch.cuda.is_available():
        raise SystemExit("an NVIDIA CUDA device is required")

    rng = np.random.default_rng(args.seed)
    numpy_shape = (args.time, args.batch, args.heads, args.head_size)
    numpy_inputs = tuple(
        rng.standard_normal(numpy_shape, dtype=np.float32) * 0.03
        for _ in range(6)
    )
    input_sha256 = hashlib.sha256(
        b"".join(np.ascontiguousarray(value).tobytes() for value in numpy_inputs)
    ).hexdigest()
    torch_inputs = tuple(
        torch.as_tensor(
            np.transpose(value, (1, 0, 2, 3)).reshape(
                args.batch, args.time, args.heads * args.head_size
            ),
            device="cuda",
            dtype=torch.bfloat16,
        ).contiguous()
        for value in numpy_inputs
    )

    old_cwd = Path.cwd()
    sys.path.insert(0, str(upstream))
    try:
        os.chdir(upstream)
        from src.model import RWKV7_CLAMPW_CUDA

        def fresh_inputs():
            return tuple(value.detach().requires_grad_(True) for value in torch_inputs)

        def forward_phase():
            return RWKV7_CLAMPW_CUDA(*fresh_inputs())

        def prepared_backward():
            values = fresh_inputs()
            output = RWKV7_CLAMPW_CUDA(*values)
            cotangent = (2.0 * output.detach().float()).to(torch.bfloat16)
            torch.cuda.synchronize()
            return output, cotangent

        def measure_backward():
            samples = []
            total = args.warmup + args.iterations
            for index in range(total):
                output, cotangent = prepared_backward()
                started = time.perf_counter_ns()
                output.backward(cotangent)
                torch.cuda.synchronize()
                if index >= args.warmup:
                    samples.append(
                        (time.perf_counter_ns() - started) / 1_000_000.0
                    )
            return timing_summary(samples)

        def forward_backward():
            output = RWKV7_CLAMPW_CUDA(*fresh_inputs())
            output.float().square().sum().backward()

        torch.cuda.synchronize()
        with gc_policy(args.disable_python_gc):
            timings = {
                "forward": _measure(
                    forward_phase,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    torch=torch,
                ),
                "backward": measure_backward(),
                "forward_backward": _measure(
                    forward_backward,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    torch=torch,
                ),
            }
    finally:
        os.chdir(old_cwd)

    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    diff = subprocess.check_output(
        ["git", "-C", str(upstream), "diff", "--binary"]
    )
    payload = {
        "schema_version": COMPUTE_BENCHMARK_SCHEMA_VERSION,
        "benchmark_kind": "wkv_compute_only",
        "framework": "official_pytorch_cuda",
        "recorded_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "upstream_revision": commit,
        "upstream_dirty": bool(diff),
        "upstream_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda},
        "devices": [
            {
                "platform": "gpu",
                "device_kind": torch.cuda.get_device_name(0),
                "id": 0,
                "capability": list(torch.cuda.get_device_capability(0)),
            }
        ],
        "shape": {
            "time": args.time,
            "batch": args.batch,
            "heads": args.heads,
            "head_size": args.head_size,
            "dtype": "bfloat16",
        },
        "inputs": {
            "numpy_rng": "PCG64",
            "seed": args.seed,
            "sha256_float32_before_bf16_cast": input_sha256,
            "initial_state": "zero (official operator contract)",
        },
        "method": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "fixed_inputs": True,
            "inputs_device_resident_before_warmup": True,
            "synchronized_each_iteration": True,
            "python_gc_disabled": args.disable_python_gc,
            "backward_objective": "sum(square(float32(activations)))",
            "excluded": [
                "input_generation",
                "host_to_device_transfer",
                "extension_compilation",
                "logging",
            ],
            "phase_windows": (
                "backward graph and cotangent are prepared before its window; "
                "forward_backward is measured independently"
            ),
        },
        "timings": timings,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
