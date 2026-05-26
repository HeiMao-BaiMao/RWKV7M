from dataclasses import dataclass
import json
from pathlib import Path
import shutil

from flax import serialization
import jax
import jax.numpy as jnp

from ..io import (
    load_train_checkpoint,
    load_train_checkpoint_metadata,
    load_train_runtime_state,
    save_train_checkpoint,
)
from ..io.config import model_config_from_dict, model_config_to_dict
from ..io.flax_checkpoint import RUNTIME_STATE_MSGPACK


CHECKPOINT_JSON = "checkpoint.json"
ORBAX_TRAIN_STATE_DIR = "orbax_train_state"
ORBAX_RUNTIME_STATE_DIR = "orbax_runtime_state"


@dataclass(frozen=True)
class DistributedCheckpointPayload:
    config: object
    metadata: dict
    start_step: int
    dataset_position: dict | None


def checkpoint_path(output_dir, step):
    return Path(output_dir) / f"ckpt-{int(step):08d}"


def list_checkpoint_dirs(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []
    checkpoints = []
    for path in output_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("ckpt-"):
            continue
        try:
            step = int(path.name.removeprefix("ckpt-"))
        except ValueError:
            continue
        checkpoints.append((step, path))
    return [path for _, path in sorted(checkpoints)]


def load_distributed_checkpoint_metadata(checkpoint_dir):
    config, payload = load_train_checkpoint_metadata(checkpoint_dir)
    return DistributedCheckpointPayload(
        config=config,
        metadata=payload,
        start_step=int(payload.get("step", 0)),
        dataset_position=payload.get("dataset_position"),
    )


def restore_distributed_train_state(checkpoint_dir, train_state_template):
    checkpoint_dir = Path(checkpoint_dir)
    config, payload = load_distributed_checkpoint_raw_metadata(checkpoint_dir)
    if payload.get("backend") == "orbax":
        train_state = load_orbax_train_state(checkpoint_dir, train_state_template)
    else:
        train_state, config, payload = load_train_checkpoint(
            checkpoint_dir,
            train_state_template,
        )
    return train_state, DistributedCheckpointPayload(
        config=config,
        metadata=payload,
        start_step=int(payload.get("step", 0)),
        dataset_position=payload.get("dataset_position"),
    )


def restore_distributed_runtime_state(checkpoint_dir, runtime_state_template):
    checkpoint_dir = Path(checkpoint_dir)
    _, payload = load_distributed_checkpoint_raw_metadata(checkpoint_dir)
    if payload.get("backend") == "orbax":
        state_dir = checkpoint_dir / ORBAX_RUNTIME_STATE_DIR
        if state_dir.exists():
            return load_orbax_runtime_state(checkpoint_dir, runtime_state_template)
        return None
    if (checkpoint_dir / RUNTIME_STATE_MSGPACK).exists():
        return load_train_runtime_state(checkpoint_dir, runtime_state_template)
    return None


def _rng_key_to_list(rng_key):
    if rng_key is None:
        return None
    return [int(x) for x in jax.device_get(jnp.asarray(rng_key)).reshape(-1).tolist()]


def _write_checkpoint_metadata(
    checkpoint_dir,
    train_state,
    config,
    *,
    backend,
    rng_key=None,
    dataset_position=None,
    metadata=None,
):
    payload = {
        "format": "rwkv7m_train_checkpoint",
        "format_version": 1,
        "backend": backend,
        "step": int(jax.device_get(train_state.step)),
        "config": model_config_to_dict(config),
        "rng_key": _rng_key_to_list(rng_key),
        "dataset_position": dataset_position,
        "metadata": {} if metadata is None else metadata,
    }
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / CHECKPOINT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return payload


def load_distributed_checkpoint_raw_metadata(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / CHECKPOINT_JSON, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return model_config_from_dict(payload["config"]), payload


def _require_orbax():
    try:
        import orbax.checkpoint as ocp
    except ImportError as exc:
        raise ImportError(
            "orbax-checkpoint is required for checkpoint_backend='orbax'."
        ) from exc
    return ocp


def save_orbax_train_state(checkpoint_dir, train_state, *, force=True):
    ocp = _require_orbax()
    state_dir = Path(checkpoint_dir) / ORBAX_TRAIN_STATE_DIR
    checkpointer = ocp.StandardCheckpointer()
    try:
        checkpointer.save(state_dir, train_state, force=force)
        checkpointer.wait_until_finished()
    finally:
        checkpointer.close()
    return state_dir


def save_orbax_runtime_state(checkpoint_dir, runtime_state, *, force=True):
    ocp = _require_orbax()
    state_dir = Path(checkpoint_dir) / ORBAX_RUNTIME_STATE_DIR
    checkpointer = ocp.StandardCheckpointer()
    try:
        checkpointer.save(state_dir, runtime_state, force=force)
        checkpointer.wait_until_finished()
    finally:
        checkpointer.close()
    return state_dir


def load_orbax_train_state(checkpoint_dir, train_state_template):
    ocp = _require_orbax()
    state_dir = Path(checkpoint_dir) / ORBAX_TRAIN_STATE_DIR
    checkpointer = ocp.StandardCheckpointer()
    try:
        return checkpointer.restore(state_dir, train_state_template)
    finally:
        checkpointer.close()


def load_orbax_runtime_state(checkpoint_dir, runtime_state_template):
    ocp = _require_orbax()
    state_dir = Path(checkpoint_dir) / ORBAX_RUNTIME_STATE_DIR
    checkpointer = ocp.StandardCheckpointer()
    try:
        return checkpointer.restore(state_dir, runtime_state_template)
    finally:
        checkpointer.close()


def save_data_parallel_checkpoint(
    output_dir,
    step,
    dist,
    config,
    process_info,
    *,
    rng_key=None,
    dataset_position=None,
    metadata=None,
    backend="flax",
    runtime_state=None,
):
    if output_dir is None:
        return None

    checkpoint_dir = checkpoint_path(output_dir, step)
    checkpoint_metadata = {
        "distributed": True,
        "process_count": process_info["process_count"],
        "local_device_count": process_info["local_device_count"],
        "device_count": process_info["device_count"],
    }
    if metadata:
        checkpoint_metadata.update(metadata)
    checkpoint_metadata["runtime_state"] = runtime_state is not None

    if backend == "orbax":
        save_orbax_train_state(checkpoint_dir, dist.train_state)
        if runtime_state is not None:
            save_orbax_runtime_state(checkpoint_dir, runtime_state)
        if process_info["process_index"] == 0:
            _write_checkpoint_metadata(
                checkpoint_dir,
                dist.train_state,
                config,
                backend="orbax",
                rng_key=rng_key,
                dataset_position=dataset_position,
                metadata=checkpoint_metadata,
            )
            return checkpoint_dir
        return None
    if backend != "flax":
        raise ValueError(f"unknown checkpoint backend: {backend}")
    if process_info["process_index"] != 0:
        return None

    return save_train_checkpoint(
        checkpoint_dir,
        jax.device_get(dist.train_state),
        config,
        rng_key=rng_key,
        dataset_position=dataset_position,
        metadata=checkpoint_metadata,
        runtime_state=(
            serialization.from_state_dict(
                runtime_state,
                jax.device_get(serialization.to_state_dict(runtime_state)),
            )
            if runtime_state is not None
            else None
        ),
    )


def rotate_checkpoints(output_dir, keep_last, process_info, *, protected_paths=None):
    if output_dir is None or keep_last is None or keep_last <= 0:
        return []
    if process_info["process_index"] != 0:
        return []

    protected = set()
    if protected_paths is not None:
        protected = {
            Path(path).resolve()
            for path in protected_paths
            if path is not None
        }
    checkpoints = list_checkpoint_dirs(output_dir)
    to_remove = checkpoints[: max(0, len(checkpoints) - int(keep_last))]
    if protected:
        to_remove = [
            path
            for path in to_remove
            if path.resolve() not in protected
        ]
    for path in to_remove:
        shutil.rmtree(path)
    return to_remove
