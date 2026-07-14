import jax
import jax.numpy as jnp
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import NamedSharding, PartitionSpec


def mesh_axis_size(mesh, axis_name):
    axis_names = tuple(mesh.axis_names)
    if axis_name not in axis_names:
        raise ValueError(f"mesh does not contain axis: {axis_name}")
    axis_index = axis_names.index(axis_name)
    return int(mesh.devices.shape[axis_index])


def _partition_spec(ndim, axis_name, shard_dim=None):
    if shard_dim is None:
        return PartitionSpec()
    spec = [None] * ndim
    spec[int(shard_dim)] = axis_name
    return PartitionSpec(*spec)


def _first_divisible_dim(value, axis_size, preferred_dims):
    shape = tuple(int(dim) for dim in jnp.asarray(value).shape)
    for dim in preferred_dims:
        if dim < len(shape) and shape[dim] % axis_size == 0:
            return dim
    return None


def parameter_partition_spec(path, value, *, axis_name, axis_size):
    value = jnp.asarray(value)
    if value.ndim < 2:
        return PartitionSpec()

    name = "/".join(str(part) for part in path)
    if "token_embedding/embedding" in name:
        shard_dim = _first_divisible_dim(value, axis_size, (0, 1))
        return _partition_spec(value.ndim, axis_name, shard_dim)

    if "lm_head/kernel" in name:
        shard_dim = _first_divisible_dim(value, axis_size, (1, 0))
        return _partition_spec(value.ndim, axis_name, shard_dim)

    if name.endswith("/kernel") and value.ndim == 2:
        shard_dim = _first_divisible_dim(value, axis_size, (1, 0))
        return _partition_spec(value.ndim, axis_name, shard_dim)

    dims_by_size = sorted(range(value.ndim), key=lambda dim: value.shape[dim], reverse=True)
    shard_dim = _first_divisible_dim(value, axis_size, dims_by_size)
    return _partition_spec(value.ndim, axis_name, shard_dim)


def parameter_sharding(path, value, mesh, *, axis_name):
    spec = parameter_partition_spec(
        path,
        value,
        axis_name=axis_name,
        axis_size=mesh_axis_size(mesh, axis_name),
    )
    return NamedSharding(mesh, spec)


def parameter_partition_summary(params, mesh, *, axis_name):
    axis_size = mesh_axis_size(mesh, axis_name)
    if isinstance(params, nnx.State):
        flat = {
            path: value[...]
            for path, value in nnx.to_flat_state(params)
        }
    else:
        flat = flatten_dict(params)
    summary = {}
    for path, value in flat.items():
        spec = parameter_partition_spec(
            path,
            value,
            axis_name=axis_name,
            axis_size=axis_size,
        )
        name = "/".join(str(part) for part in path)
        summary[name] = {
            "shape": [int(dim) for dim in jnp.asarray(value).shape],
            "partition_spec": str(spec),
        }
    return summary


def place_parameter_tree(params, mesh, *, axis_name):
    flat = flatten_dict(params)
    placed = {
        path: jax.device_put(value, parameter_sharding(path, value, mesh, axis_name=axis_name))
        for path, value in flat.items()
    }
    return unflatten_dict(placed)
