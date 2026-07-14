"""Accelerator kernels and backend dispatch support."""

from .screening_backend import (
    ScreeningBackend,
    available_screening_backends,
    resolve_screening_backend,
)
from .wkv_backend import (
    WKVBackend,
    available_wkv_backends,
    resolve_wkv_backend,
)
from .wkv_ffi import (
    WKVFFIBackend,
    register_wkv_ffi_backend,
    unregister_wkv_ffi_backend,
)

__all__ = [
    "ScreeningBackend",
    "WKVBackend",
    "WKVFFIBackend",
    "available_screening_backends",
    "available_wkv_backends",
    "register_wkv_ffi_backend",
    "resolve_screening_backend",
    "resolve_wkv_backend",
    "unregister_wkv_ffi_backend",
]
