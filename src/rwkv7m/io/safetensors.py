import json
from pathlib import Path

import jax.numpy as jnp
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict
from safetensors import safe_open
from safetensors.flax import load_file, save_file

from ..model import ModelConfig
from ..tokenizer import tokenizer_metadata as default_tokenizer_metadata
from .config import model_config_from_dict, model_config_to_dict


FORMAT_NAME = "rwkv7m"
FORMAT_VERSION = "1"
CONFIG_METADATA_KEY = "rwkv7m_config_json"


def _standard_metadata(config: ModelConfig, tokenizer_metadata=None):
    screening = config.screening
    export_metadata = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "architecture": "rwkv7m",
        "dtype": config.dtype,
        "d_model": str(config.d_model),
        "d_ffn": str(config.d_ffn),
        "n_layers": str(config.n_layers),
        "n_heads": str(config.n_heads),
        "head_size": str(config.head_size),
        "vocab_size": str(config.vocab_size),
        "max_seq_len": str(config.max_seq_len),
        "use_screening": str(bool(config.use_screening)).lower(),
        "screening_n_slots": str(screening.n_slots),
        "screening_d_slot": str(screening.d_slot),
        "screening_layers": ",".join(str(layer) for layer in screening.screened_layers),
        "screening_write_mode": (
            screening.write_mode
            if screening.write_mode is not None
            else "legacy_phase_mapping"
        ),
        "screening_gate_space": screening.gate_space,
        "screening_gate_activation": screening.gate_activation,
        "screening_candidate_rank": (
            "none"
            if screening.candidate_rank is None
            else str(screening.candidate_rank)
        ),
        "screening_read_tiles": str(screening.n_read_tiles),
        "screening_checkpoint_interval": (
            "none"
            if screening.checkpoint_interval is None
            else str(screening.checkpoint_interval)
        ),
        CONFIG_METADATA_KEY: json.dumps(model_config_to_dict(config), sort_keys=True),
    }
    tokenizer_data = (
        default_tokenizer_metadata()
        if tokenizer_metadata is None
        else dict(tokenizer_metadata)
    )
    export_metadata.update({str(key): str(value) for key, value in tokenizer_data.items()})
    return export_metadata


def _flatten_params(params):
    if isinstance(params, nnx.State):
        return {
            "/".join(str(part) for part in path): jnp.asarray(value[...])
            for path, value in nnx.to_flat_state(params)
        }
    flat = flatten_dict(params, sep="/")
    return {name: jnp.asarray(value) for name, value in flat.items()}


def _unflatten_params(tensors):
    flat = {tuple(name.split("/")): jnp.asarray(value) for name, value in tensors.items()}
    return unflatten_dict(flat)


def _read_metadata(path):
    with safe_open(path, framework="flax") as handle:
        return dict(handle.metadata() or {})


def save_model_safetensors(
    path,
    params,
    config: ModelConfig,
    *,
    metadata=None,
    tokenizer_metadata=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export_metadata = _standard_metadata(config, tokenizer_metadata=tokenizer_metadata)
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
