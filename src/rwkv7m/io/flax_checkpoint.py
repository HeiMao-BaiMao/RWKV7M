import json
from pathlib import Path

import jax.numpy as jnp
from flax import serialization

from ..model import ModelConfig
from .config import model_config_from_dict, model_config_to_dict
from .safetensors import save_model_safetensors


CHECKPOINT_JSON = "checkpoint.json"
TRAIN_STATE_MSGPACK = "train_state.msgpack"
RUNTIME_STATE_MSGPACK = "runtime_state.msgpack"
MODEL_SAFETENSORS = "model.safetensors"


def _rng_key_to_list(rng_key):
    if rng_key is None:
        return None
    return [int(x) for x in jnp.asarray(rng_key).reshape(-1).tolist()]


def save_train_checkpoint(
    checkpoint_dir,
    train_state,
    config: ModelConfig,
    *,
    rng_key=None,
    dataset_position=None,
    metadata=None,
    runtime_state=None,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state_path = checkpoint_dir / TRAIN_STATE_MSGPACK
    state_path.write_bytes(serialization.to_bytes(train_state))
    if runtime_state is not None:
        runtime_state_path = checkpoint_dir / RUNTIME_STATE_MSGPACK
        runtime_state_path.write_bytes(serialization.to_bytes(runtime_state))

    save_model_safetensors(
        checkpoint_dir / MODEL_SAFETENSORS,
        train_state.params,
        config,
        metadata={
            "checkpoint_step": int(train_state.step),
            **({} if metadata is None else metadata),
        },
    )

    payload = {
        "format": "rwkv7m_train_checkpoint",
        "format_version": 1,
        "step": int(train_state.step),
        "config": model_config_to_dict(config),
        "rng_key": _rng_key_to_list(rng_key),
        "dataset_position": dataset_position,
        "metadata": {} if metadata is None else metadata,
    }
    with open(checkpoint_dir / CHECKPOINT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return checkpoint_dir


def load_train_checkpoint_metadata(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / CHECKPOINT_JSON, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return model_config_from_dict(payload["config"]), payload


def load_train_checkpoint(checkpoint_dir, train_state_template):
    checkpoint_dir = Path(checkpoint_dir)
    config, payload = load_train_checkpoint_metadata(checkpoint_dir)
    state_bytes = (checkpoint_dir / TRAIN_STATE_MSGPACK).read_bytes()
    train_state = serialization.from_bytes(train_state_template, state_bytes)
    return train_state, config, payload


def load_train_runtime_state(checkpoint_dir, runtime_state_template):
    checkpoint_dir = Path(checkpoint_dir)
    state_path = checkpoint_dir / RUNTIME_STATE_MSGPACK
    if not state_path.exists():
        return None
    return serialization.from_bytes(runtime_state_template, state_path.read_bytes())
