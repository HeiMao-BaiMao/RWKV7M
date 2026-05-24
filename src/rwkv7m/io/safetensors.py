import json
from pathlib import Path

import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
from safetensors import safe_open
from safetensors.flax import load_file, save_file

from ..model import ModelConfig
from .config import model_config_from_dict, model_config_to_dict


FORMAT_NAME = "rwkv7m"
FORMAT_VERSION = "1"
CONFIG_METADATA_KEY = "rwkv7m_config_json"


def _flatten_params(params):
    flat = flatten_dict(params, sep="/")
    return {name: jnp.asarray(value) for name, value in flat.items()}


def _unflatten_params(tensors):
    flat = {tuple(name.split("/")): jnp.asarray(value) for name, value in tensors.items()}
    return unflatten_dict(flat)


def _read_metadata(path):
    with safe_open(path, framework="flax") as handle:
        return dict(handle.metadata() or {})


def save_model_safetensors(path, params, config: ModelConfig, *, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export_metadata = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        CONFIG_METADATA_KEY: json.dumps(model_config_to_dict(config), sort_keys=True),
    }
    if metadata:
        export_metadata.update({str(key): str(value) for key, value in metadata.items()})
    save_file(_flatten_params(params), path, metadata=export_metadata)
    return path


def load_model_safetensors(path):
    path = Path(path)
    metadata = _read_metadata(path)
    tensors = load_file(path)
    params = _unflatten_params(tensors)
    config_json = metadata.get(CONFIG_METADATA_KEY)
    config = model_config_from_dict(json.loads(config_json)) if config_json else None
    return params, config, metadata
