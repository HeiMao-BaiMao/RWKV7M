"""Selection policy for WKV accelerator kernels."""

from __future__ import annotations

import os
from typing import Literal, cast

import jax


WKVBackend = Literal[
    "reference",
    "pallas_tpu",
    "pallas_gpu_mosaic",
    "pallas_gpu_triton",
    "pallas_gpu_triton_reference_vjp",
    "ffi",
]

_BACKENDS = frozenset(
    {
        "reference",
        "pallas_tpu",
        "pallas_gpu_mosaic",
        "pallas_gpu_triton",
        "pallas_gpu_triton_reference_vjp",
        "ffi",
    }
)
_HOPPER_OR_NEWER_MARKERS = (
    "h100",
    "h200",
    "gh200",
    "b100",
    "b200",
    "gb200",
    "blackwell",
)
_AMD_MARKERS = ("amd", "instinct", "mi300", "radeon", "gfx")


def available_wkv_backends() -> tuple[str, ...]:
    """Return backend names accepted by the dispatcher."""

    return tuple(sorted(_BACKENDS))


def _default_device_kind() -> str:
    devices = jax.devices()
    if not devices:
        return ""
    return str(getattr(devices[0], "device_kind", "")).lower()


def resolve_wkv_backend(
    requested: str | None = None,
    *,
    platform: str | None = None,
    device_kind: str | None = None,
) -> WKVBackend:
    """Resolve an explicit or automatic WKV backend.

    ``auto`` is Pallas-first on accelerators. CPU intentionally keeps the
    portable reference implementation. FFI is never selected automatically.
    ``RWKV7M_WKV_BACKEND`` provides a process-level override for benchmarking
    and rollback without changing model configs.
    """

    selected = requested
    if selected is None:
        selected = os.environ.get("RWKV7M_WKV_BACKEND", "auto")
    selected = selected.strip().lower()
    if selected != "auto":
        if selected not in _BACKENDS:
            choices = ", ".join(("auto", *sorted(_BACKENDS)))
            raise ValueError(
                f"unknown WKV backend {selected!r}; expected one of {choices}"
            )
        return cast(WKVBackend, selected)

    current_platform = platform or jax.default_backend()
    if current_platform == "tpu":
        return "pallas_tpu"
    if current_platform == "gpu":
        kind = (device_kind or _default_device_kind()).lower()
        if any(marker in kind for marker in _AMD_MARKERS):
            # The Triton forward is numerically valid on MI300X, but a
            # production full graph with multiple Pallas WKV pullbacks can
            # corrupt cotangents even though every captured pullback passes
            # in isolation. Keep Pallas forward while fail-closing training
            # to the portable VJP until an AMD-specific full-graph gate passes.
            return "pallas_gpu_triton_reference_vjp"
        if any(marker in kind for marker in _HOPPER_OR_NEWER_MARKERS):
            return "pallas_gpu_mosaic"
        return "pallas_gpu_triton"
    return "reference"


__all__ = [
    "WKVBackend",
    "available_wkv_backends",
    "resolve_wkv_backend",
]
