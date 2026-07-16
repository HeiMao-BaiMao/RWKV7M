"""Backend selection for the vocabulary-tiled training head."""

from __future__ import annotations

import os
from typing import Literal, cast

import jax


TrainingLossBackend = Literal[
    "reference",
    "pallas_tpu",
    "pallas_gpu_mosaic",
    "pallas_gpu_triton",
]

_BACKENDS = frozenset(
    {
        "reference",
        "pallas_tpu",
        "pallas_gpu_mosaic",
        "pallas_gpu_triton",
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


def available_training_loss_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def _default_device_kind() -> str:
    devices = jax.devices()
    if not devices:
        return ""
    return str(getattr(devices[0], "device_kind", "")).lower()


def resolve_training_loss_backend(
    requested: str | None = None,
    *,
    platform: str | None = None,
    device_kind: str | None = None,
) -> TrainingLossBackend:
    selected = requested
    if selected is None:
        selected = os.environ.get("RWKV7M_TRAINING_LOSS_BACKEND", "auto")
    selected = selected.strip().lower()
    if selected != "auto":
        if selected not in _BACKENDS:
            choices = ", ".join(("auto", *sorted(_BACKENDS)))
            raise ValueError(
                f"unknown training-loss backend {selected!r}; "
                f"expected one of {choices}"
            )
        return cast(TrainingLossBackend, selected)

    current_platform = platform or jax.default_backend()
    if current_platform == "tpu":
        return "pallas_tpu"
    if current_platform == "gpu":
        kind = (device_kind or _default_device_kind()).lower()
        if any(marker in kind for marker in _HOPPER_OR_NEWER_MARKERS):
            return "pallas_gpu_mosaic"
        return "pallas_gpu_triton"
    return "reference"


__all__ = [
    "TrainingLossBackend",
    "available_training_loss_backends",
    "resolve_training_loss_backend",
]
