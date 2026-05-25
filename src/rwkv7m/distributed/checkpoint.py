from dataclasses import dataclass
from pathlib import Path
import shutil

import jax

from ..io import load_train_checkpoint, load_train_checkpoint_metadata, save_train_checkpoint


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
    train_state, config, payload = load_train_checkpoint(checkpoint_dir, train_state_template)
    return train_state, DistributedCheckpointPayload(
        config=config,
        metadata=payload,
        start_step=int(payload.get("step", 0)),
        dataset_position=payload.get("dataset_position"),
    )


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
):
    if output_dir is None or process_info["process_index"] != 0:
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

    return save_train_checkpoint(
        checkpoint_dir,
        jax.device_get(dist.train_state),
        config,
        rng_key=rng_key,
        dataset_position=dataset_position,
        metadata=checkpoint_metadata,
    )


def rotate_checkpoints(output_dir, keep_last, process_info):
    if output_dir is None or keep_last is None or keep_last <= 0:
        return []
    if process_info["process_index"] != 0:
        return []

    checkpoints = list_checkpoint_dirs(output_dir)
    to_remove = checkpoints[: max(0, len(checkpoints) - int(keep_last))]
    for path in to_remove:
        shutil.rmtree(path)
    return to_remove
