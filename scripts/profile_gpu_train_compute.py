"""Launch the fixed-batch workload under Nsight Systems or Nsight Compute."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", choices=("systems", "compute"), required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ncu-kernel-regex", default=None)
    parser.add_argument("--ncu-set", default="full")
    parser.add_argument("--ncu-launch-count", type=int, default=1)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a benchmark command is required after --")
    if args.tool == "compute" and not args.ncu_kernel_regex:
        parser.error("--ncu-kernel-regex is required for Nsight Compute")
    if args.ncu_launch_count <= 0:
        parser.error("--ncu-launch-count must be positive")
    return args


def _tool_version(executable):
    completed = subprocess.run(
        [executable, "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout.strip()


def _ensure_profile_range(command):
    if "--profile-mode" in command:
        index = command.index("--profile-mode")
        if index + 1 >= len(command) or command[index + 1] != "cuda_profiler_api":
            raise SystemExit(
                "Nsight requires --profile-mode cuda_profiler_api"
            )
        return command
    return [*command, "--profile-mode", "cuda_profiler_api"]


def _systems_command(nsys, prefix, benchmark):
    return [
        nsys,
        "profile",
        "--trace=cuda,nvtx,osrt,cublas,cudnn",
        "--sample=none",
        "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--force-overwrite=true",
        f"--output={prefix}",
        *benchmark,
    ]


def _compute_command(ncu, args, benchmark):
    return [
        ncu,
        "--target-processes=all",
        "--profile-from-start=off",
        f"--set={args.ncu_set}",
        f"--kernel-name=regex:{args.ncu_kernel_regex}",
        f"--launch-count={args.ncu_launch_count}",
        "--force-overwrite",
        f"--export={args.output_prefix}",
        *benchmark,
    ]


def main(argv=None):
    args = parse_args(argv)
    executable_name = "nsys" if args.tool == "systems" else "ncu"
    executable = shutil.which(executable_name)
    if executable is None:
        raise SystemExit(f"{executable_name} was not found on PATH")
    benchmark = _ensure_profile_range(args.command)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    command = (
        _systems_command(executable, args.output_prefix, benchmark)
        if args.tool == "systems"
        else _compute_command(executable, args, benchmark)
    )
    completed = subprocess.run(command, check=False)
    stats_path = None
    if args.tool == "systems" and completed.returncode == 0:
        report = args.output_prefix.with_suffix(".nsys-rep")
        stats_path = args.output_prefix.with_name(
            args.output_prefix.name + "-stats.csv"
        )
        stats = subprocess.run(
            [
                executable,
                "stats",
                "--report=cuda_gpu_kern_sum,cuda_api_sum",
                "--format=csv",
                str(report),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        stats_path.write_text(stats.stdout, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "profile_kind": f"nsight_{args.tool}",
        "recorded_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "tool_version": _tool_version(executable),
        "command": command,
        "returncode": completed.returncode,
        "output_prefix": str(args.output_prefix.resolve()),
        "stats": str(stats_path.resolve()) if stats_path is not None else None,
        "limitations": [
            "JAX/XProf collection must not run in the same process as Nsight",
            "Nsight Compute requires GPU performance-counter permission",
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
