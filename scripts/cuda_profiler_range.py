"""Small CUDA Profiler API range used by external Nsight tools."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import ctypes.util
import importlib.util
import os
from pathlib import Path
import sys


def _cudart_candidates():
    candidates = [
        ctypes.util.find_library("cudart"),
        "libcudart.so",
        "libcudart.so.13",
        "libcudart.so.12",
        "cudart64_13.dll",
        "cudart64_12.dll",
    ]
    for directory in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if directory:
            candidates.extend(sorted(Path(directory).glob("libcudart.so*")))
    try:
        runtime_spec = importlib.util.find_spec("nvidia.cuda_runtime")
    except ModuleNotFoundError:
        runtime_spec = None
    if runtime_spec is not None and runtime_spec.submodule_search_locations:
        for location in runtime_spec.submodule_search_locations:
            candidates.extend(
                sorted((Path(location) / "lib").glob("libcudart.so*"))
            )
    candidates.extend(
        sorted(
            Path(sys.prefix).glob(
                "lib/python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so*"
            )
        )
    )
    seen = set()
    for candidate in candidates:
        if candidate is None:
            continue
        rendered = str(candidate)
        if rendered not in seen:
            seen.add(rendered)
            yield rendered


def _load_cudart():
    errors = []
    for candidate in _cudart_candidates():
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    rendered = "; ".join(errors) if errors else "no library candidates"
    raise RuntimeError(f"CUDA runtime library was not found: {rendered}")


def _check_cuda(status, operation):
    if status != 0:
        raise RuntimeError(f"{operation} failed with CUDA status {status}")


@contextmanager
def cuda_profiler_range():
    runtime = _load_cudart()
    runtime.cudaProfilerStart.restype = ctypes.c_int
    runtime.cudaProfilerStop.restype = ctypes.c_int
    _check_cuda(runtime.cudaProfilerStart(), "cudaProfilerStart")
    try:
        yield
    finally:
        _check_cuda(runtime.cudaProfilerStop(), "cudaProfilerStop")


__all__ = ["cuda_profiler_range"]
