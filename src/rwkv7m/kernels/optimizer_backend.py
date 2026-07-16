"""Selection policy for the experimental fused GPU optimizer."""

from __future__ import annotations

import os
from typing import Literal, cast


OptimizerBackend = Literal[
    "optax",
    "pallas_gpu_mosaic",
    "pallas_gpu_triton",
]

_BACKENDS = frozenset(
    {"optax", "pallas_gpu_mosaic", "pallas_gpu_triton"}
)


def available_optimizer_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def resolve_optimizer_backend(requested: str | None = None) -> OptimizerBackend:
    selected = requested
    if selected is None:
        selected = os.environ.get("RWKV7M_OPTIMIZER_BACKEND", "auto")
    selected = selected.strip().lower()
    # Keep the measured Optax path as the default until a real-GPU complete
    # step gate proves that the Pallas path is a net win.
    if selected == "auto":
        selected = "optax"
    if selected not in _BACKENDS:
        choices = ", ".join(("auto", *sorted(_BACKENDS)))
        raise ValueError(
            f"unknown optimizer backend {selected!r}; expected one of {choices}"
        )
    return cast(OptimizerBackend, selected)


__all__ = [
    "OptimizerBackend",
    "available_optimizer_backends",
    "resolve_optimizer_backend",
]
