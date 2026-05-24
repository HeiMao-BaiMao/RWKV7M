import importlib.util
import json
from pathlib import Path

from safetensors import safe_open

from ...io.config import model_config_from_dict
from ...io.safetensors import CONFIG_METADATA_KEY


def is_torch_available():
    return importlib.util.find_spec("torch") is not None


def require_torch():
    if not is_torch_available():
        raise ImportError(
            "PyTorch is not installed. Install torch to use rwkv7m.backends.torch."
        )
    import torch

    return torch


def flax_key_to_torch_key(key: str):
    return key.replace("/", ".")


def _read_metadata(path):
    with safe_open(path, framework="flax") as handle:
        return dict(handle.metadata() or {})


def load_torch_safetensors(path, *, rename_keys=True):
    require_torch()
    from safetensors.torch import load_file

    path = Path(path)
    metadata = _read_metadata(path)
    tensors = load_file(path)
    if rename_keys:
        tensors = {flax_key_to_torch_key(key): value for key, value in tensors.items()}

    config_json = metadata.get(CONFIG_METADATA_KEY)
    config = model_config_from_dict(json.loads(config_json)) if config_json else None
    return tensors, config, metadata
