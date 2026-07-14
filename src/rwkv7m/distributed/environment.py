import platform
from importlib import metadata

import jax


_PACKAGE_NAMES = (
    "jax",
    "jaxlib",
    "flax",
    "optax",
    "orbax-checkpoint",
    "safetensors",
    "libtpu",
)


def _package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def runtime_version_manifest():
    """Return reproducibility-relevant runtime and accelerator versions."""
    devices = jax.devices()
    return {
        "python": platform.python_version(),
        "packages": {
            name: version
            for name in _PACKAGE_NAMES
            if (version := _package_version(name)) is not None
        },
        "jax_backend": jax.default_backend(),
        "process_count": jax.process_count(),
        "device_count": jax.device_count(),
        "devices": [str(device) for device in devices],
        "device_kinds": sorted(
            {
                str(getattr(device, "device_kind", type(device).__name__))
                for device in devices
            }
        ),
    }
