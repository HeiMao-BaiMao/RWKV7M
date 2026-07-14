"""Optional FFI registration boundary for external WKV implementations.

This module deliberately contains no native implementation. Pallas remains the
accelerator default; applications may register a measured FFI implementation
without making it a package dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


WKVForward = Callable[..., tuple[object, object]]
WKVForwardWithAux = Callable[
    ..., tuple[tuple[object, object], tuple[object, ...]]
]
WKVBackward = Callable[..., tuple[object, ...]]


@dataclass(frozen=True)
class WKVFFIBackend:
    """Callable contract implemented by an optional native WKV extension.

    ``forward`` receives the six vectors and initial state and returns
    ``(activation, final_state)``. ``forward_with_aux`` returns that pair plus
    an auxiliary tuple. ``backward`` receives the seven primal inputs, two
    output cotangents, then the unpacked auxiliary tuple, and returns seven
    input cotangents in primal order.
    """

    forward: WKVForward
    forward_with_aux: WKVForwardWithAux
    backward: WKVBackward


_registered_backend: WKVFFIBackend | None = None


def register_wkv_ffi_backend(
    backend: WKVFFIBackend,
    *,
    replace: bool = False,
) -> None:
    """Register an external backend, rejecting accidental replacement."""

    global _registered_backend
    if _registered_backend is not None and not replace:
        raise RuntimeError("a WKV FFI backend is already registered")
    _registered_backend = backend


def unregister_wkv_ffi_backend() -> None:
    """Remove the process-local FFI backend registration."""

    global _registered_backend
    _registered_backend = None


def require_wkv_ffi_backend() -> WKVFFIBackend:
    """Return the registered backend or fail before JAX tracing starts."""

    if _registered_backend is None:
        raise RuntimeError(
            "WKV backend 'ffi' was requested, but no FFI backend is registered"
        )
    return _registered_backend


__all__ = [
    "WKVFFIBackend",
    "register_wkv_ffi_backend",
    "require_wkv_ffi_backend",
    "unregister_wkv_ffi_backend",
]
