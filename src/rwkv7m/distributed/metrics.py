import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp


def aggregate_metrics(metrics):
    """Normalize metric leaves to scalar JAX arrays for host reporting."""

    def aggregate(value):
        value = jnp.asarray(value)
        if value.shape == ():
            return value
        return jnp.mean(value)

    return jax.tree_util.tree_map(aggregate, metrics)


def metrics_to_host_dict(metrics):
    metrics = aggregate_metrics(metrics)
    host_metrics = jax.device_get(metrics)
    return {
        str(key): float(value) if jnp.asarray(value).shape == () else value
        for key, value in host_metrics.items()
    }


def mean_metric_dict(metric_dicts):
    if not metric_dicts:
        return {}
    keys = sorted({key for metrics in metric_dicts for key in metrics})
    result = {}
    for key in keys:
        values = [float(metrics[key]) for metrics in metric_dicts if key in metrics]
        if values:
            result[key] = sum(values) / len(values)
    return result


def write_metric_record(jsonl_path=None, csv_path=None, record=None, *, process_info=None):
    if record is None:
        record = {}
    if process_info is not None and process_info["process_index"] != 0:
        return

    normalized = {
        str(key): (
            float(value)
            if isinstance(value, jnp.ndarray) and jnp.asarray(value).shape == ()
            else value
        )
        for key, value in record.items()
    }

    if jsonl_path is not None:
        path = Path(jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(normalized, sort_keys=True))
            f.write("\n")

    if csv_path is not None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted(normalized)
        write_header = not path.exists() or path.stat().st_size == 0
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(normalized)
