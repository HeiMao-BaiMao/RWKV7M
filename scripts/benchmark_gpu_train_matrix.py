"""Run the controlled single-GPU head, optimizer, and batch-size matrix."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time


HEAD_MODES = ("full_xla", "pallas_tiled")
OPTIMIZER_BACKENDS = ("optax", "pallas_gpu_triton", "pallas_gpu_mosaic")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-config")
    source.add_argument("--model-preset")
    parser.add_argument(
        "--variant",
        choices=("baseline", "screening", "read_write"),
        default="baseline",
    )
    parser.add_argument("--ctx-len", type=int, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument(
        "--head-modes", choices=HEAD_MODES, nargs="+", default=HEAD_MODES
    )
    parser.add_argument(
        "--optimizer-backends",
        choices=OPTIMIZER_BACKENDS,
        nargs="+",
        default=("optax", "pallas_gpu_triton"),
    )
    parser.add_argument("--training-vocab-tile-size", type=int, default=16_384)
    parser.add_argument("--benchmark-warmup", type=int, default=5)
    parser.add_argument("--benchmark-iterations", type=int, default=20)
    parser.add_argument("--disable-python-gc", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    if args.ctx_len <= 0 or any(size <= 0 for size in args.batch_sizes):
        parser.error("context and batch sizes must be positive")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("--batch-sizes must not contain duplicates")
    if args.training_vocab_tile_size <= 0:
        parser.error("--training-vocab-tile-size must be positive")
    if args.benchmark_warmup < 0 or args.benchmark_iterations <= 0:
        parser.error("benchmark warmup must be non-negative and iterations positive")
    return args


def _run(command, *, log_path):
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    return completed.returncode, elapsed


def _benchmark_command(args, *, batch_size, fixed_batch, output, head, optimizer):
    command = [
        sys.executable,
        "scripts/benchmark_local_train_compute.py",
        "--fixed-batch",
        str(fixed_batch),
        "--variant",
        args.variant,
        "--ctx-len",
        str(args.ctx_len),
        "--batch-size",
        str(batch_size),
        "--benchmark-warmup",
        str(args.benchmark_warmup),
        "--benchmark-iterations",
        str(args.benchmark_iterations),
        "--output",
        str(output),
        "--no-remat-blocks",
        "--no-sequence-chunking",
        "--no-head-chunking",
        "--optimizer-backend",
        optimizer,
    ]
    if args.model_config:
        command.extend(("--model-config", args.model_config))
    else:
        command.extend(("--model-preset", args.model_preset))
    if args.disable_python_gc:
        command.append("--disable-python-gc")
    if head == "full_xla":
        command.append("--no-training-vocab-tiling")
    else:
        command.extend(
            (
                "--training-vocab-tile-size",
                str(args.training_vocab_tile_size),
            )
        )
    return command


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(records, *, batch_sizes, head_modes, optimizer_backends):
    successful = {
        (record["batch_size"], record["head"], record["optimizer"]): record
        for record in records
        if record["status"] == "ok"
    }
    comparisons = {"head": [], "optimizer": [], "scaling": []}
    for batch_size in batch_sizes:
        for optimizer in optimizer_backends:
            full = successful.get((batch_size, "full_xla", optimizer))
            tiled = successful.get((batch_size, "pallas_tiled", optimizer))
            if full is not None and tiled is not None:
                comparisons["head"].append(
                    {
                        "batch_size": batch_size,
                        "optimizer": optimizer,
                        "tiled_over_full_throughput": _ratio(
                            tiled["tokens_per_second"], full["tokens_per_second"]
                        ),
                        "full_over_tiled_step_latency": _ratio(
                            full["full_step_median_ms"],
                            tiled["full_step_median_ms"],
                        ),
                    }
                )
        for head in head_modes:
            optax_record = successful.get((batch_size, head, "optax"))
            for optimizer in optimizer_backends:
                if optimizer == "optax":
                    continue
                fused = successful.get((batch_size, head, optimizer))
                if optax_record is not None and fused is not None:
                    comparisons["optimizer"].append(
                        {
                            "batch_size": batch_size,
                            "head": head,
                            "optimizer": optimizer,
                            "fused_over_optax_throughput": _ratio(
                                fused["tokens_per_second"],
                                optax_record["tokens_per_second"],
                            ),
                            "optax_over_fused_optimizer_latency": _ratio(
                                optax_record["optimizer_median_ms"],
                                fused["optimizer_median_ms"],
                            ),
                        }
                    )
    base_batch = min(batch_sizes)
    for head in head_modes:
        for optimizer in optimizer_backends:
            base = successful.get((base_batch, head, optimizer))
            if base is None:
                continue
            for batch_size in batch_sizes:
                record = successful.get((batch_size, head, optimizer))
                if record is None:
                    continue
                batch_ratio = batch_size / base_batch
                throughput_ratio = _ratio(
                    record["tokens_per_second"], base["tokens_per_second"]
                )
                comparisons["scaling"].append(
                    {
                        "batch_size": batch_size,
                        "head": head,
                        "optimizer": optimizer,
                        "throughput_over_base_batch": throughput_ratio,
                        "parallel_efficiency": _ratio(
                            throughput_ratio, batch_ratio
                        ),
                    }
                )
    return comparisons


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for batch_size in args.batch_sizes:
        fixed_batch = args.output_dir / f"fixed-b{batch_size}-t{args.ctx_len}.npz"
        prepare_log = args.output_dir / f"prepare-b{batch_size}.log"
        prepare_command = [
            sys.executable,
            "scripts/prepare_compute_benchmark_batch.py",
            "--data-file",
            args.data_file,
            "--ctx-len",
            str(args.ctx_len),
            "--batch-size",
            str(batch_size),
            "--output",
            str(fixed_batch),
        ]
        returncode, _ = _run(prepare_command, log_path=prepare_log)
        if returncode:
            raise SystemExit(f"fixed-batch preparation failed: {prepare_log}")
        for head in args.head_modes:
            for optimizer in args.optimizer_backends:
                stem = f"b{batch_size}-{head}-{optimizer}"
                output = args.output_dir / f"{stem}.json"
                log = args.output_dir / f"{stem}.log"
                command = _benchmark_command(
                    args,
                    batch_size=batch_size,
                    fixed_batch=fixed_batch,
                    output=output,
                    head=head,
                    optimizer=optimizer,
                )
                print(f"running {stem}", flush=True)
                returncode, elapsed = _run(command, log_path=log)
                record = {
                    "batch_size": batch_size,
                    "head": head,
                    "optimizer": optimizer,
                    "result": str(output.resolve()),
                    "log": str(log.resolve()),
                    "elapsed_seconds": elapsed,
                }
                if returncode == 0 and output.is_file():
                    payload = json.loads(output.read_text(encoding="utf-8"))
                    if (
                        len(payload["devices"]) != 1
                        or payload["devices"][0]["platform"] != "gpu"
                    ):
                        raise SystemExit(
                            "GPU matrix requires exactly one visible GPU"
                        )
                    record.update(
                        status="ok",
                        tokens_per_second=payload["timings"]["full_step"][
                            "tokens_per_second_median"
                        ],
                        full_step_median_ms=payload["timings"]["full_step"][
                            "median_ms"
                        ],
                        optimizer_median_ms=payload["timings"]["optimizer"][
                            "median_ms"
                        ],
                        fixed_batch_content_sha256=payload["fixed_batch"][
                            "content_sha256"
                        ],
                    )
                else:
                    log_tail = log.read_text(encoding="utf-8", errors="replace")[-4000:]
                    failure_kind = (
                        "out_of_memory"
                        if any(
                            marker in log_tail.lower()
                            for marker in (
                                "out of memory",
                                "resource_exhausted",
                                "cuda_error_out_of_memory",
                            )
                        )
                        else "benchmark_error"
                    )
                    record.update(
                        status="failed",
                        returncode=returncode,
                        failure_kind=failure_kind,
                    )
                records.append(record)
                partial = {
                    "schema_version": 1,
                    "benchmark_kind": "single_gpu_train_matrix",
                    "recorded_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                    "records": records,
                    "comparisons": summarize(
                        records,
                        batch_sizes=args.batch_sizes,
                        head_modes=args.head_modes,
                        optimizer_backends=args.optimizer_backends,
                    ),
                }
                args.summary.parent.mkdir(parents=True, exist_ok=True)
                args.summary.write_text(
                    json.dumps(partial, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                if record["status"] == "failed" and args.fail_fast:
                    raise SystemExit(f"benchmark failed: {log}")
    failed = sum(record["status"] == "failed" for record in records)
    print(f"completed {len(records)} matrix cells; failed={failed}")
    # Capacity probing intentionally permits larger batches to OOM. Every
    # failed cell remains explicit in the summary; --fail-fast is available
    # when a fully successful matrix is required.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
