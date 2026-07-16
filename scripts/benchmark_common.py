"""Shared, framework-neutral reporting helpers for compute-only benchmarks."""

from __future__ import annotations

from contextlib import contextmanager
import gc
import hashlib
import math
import statistics

import numpy as np


COMPUTE_BENCHMARK_SCHEMA_VERSION = 2


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def timing_summary(milliseconds):
    return {
        "iterations": len(milliseconds),
        "mean_ms": statistics.fmean(milliseconds),
        "median_ms": statistics.median(milliseconds),
        "stdev_ms": (
            statistics.stdev(milliseconds) if len(milliseconds) > 1 else 0.0
        ),
        "min_ms": min(milliseconds),
        "p95_ms": percentile(milliseconds, 0.95),
        "max_ms": max(milliseconds),
    }


def measurement_contract(*, warmup, iterations, disable_python_gc):
    return {
        "warmup": warmup,
        "iterations": iterations,
        "fixed_batch": True,
        "batch_device_resident_before_warmup": True,
        "synchronized_each_iteration": True,
        "python_gc_disabled": disable_python_gc,
        "excluded": [
            "dataset_sampling",
            "host_to_device_transfer",
            "compilation",
            "checkpoint_io",
            "logging",
            "host_metrics",
        ],
        "phase_windows": (
            "forward, backward, and optimizer are independent windows; "
            "full_step is measured separately without phase barriers"
        ),
    }


def fixed_batch_content_sha256(batch):
    """Hash logical batch arrays independently of the NPZ container bytes."""

    digest = hashlib.sha256()
    for name in sorted(batch):
        value = np.ascontiguousarray(np.asarray(batch[name]))
        encoded_name = name.encode("utf-8")
        encoded_dtype = value.dtype.str.encode("ascii")
        digest.update(len(encoded_name).to_bytes(4, "little"))
        digest.update(encoded_name)
        digest.update(len(encoded_dtype).to_bytes(4, "little"))
        digest.update(encoded_dtype)
        digest.update(value.ndim.to_bytes(4, "little"))
        for dimension in value.shape:
            digest.update(int(dimension).to_bytes(8, "little", signed=False))
        digest.update(value.nbytes.to_bytes(8, "little", signed=False))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@contextmanager
def gc_policy(disable_python_gc):
    was_enabled = gc.isenabled()
    if disable_python_gc:
        gc.collect()
        gc.disable()
    try:
        yield
    finally:
        if disable_python_gc and was_enabled:
            gc.enable()


__all__ = [
    "COMPUTE_BENCHMARK_SCHEMA_VERSION",
    "fixed_batch_content_sha256",
    "gc_policy",
    "measurement_contract",
    "percentile",
    "timing_summary",
]
