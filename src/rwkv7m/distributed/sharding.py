import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec


def axis_sharding(mesh, axis_name):
    return NamedSharding(mesh, PartitionSpec(axis_name))


def data_parallel_sharding(mesh, axis_name="data"):
    return axis_sharding(mesh, axis_name)


def model_parallel_sharding(mesh, axis_name="model"):
    return axis_sharding(mesh, axis_name)


def replicated_sharding(mesh):
    return NamedSharding(mesh, PartitionSpec())


def put_to_devices(tree, sharding):
    return jax.tree_util.tree_map(lambda x: jax.device_put(x, sharding), tree)


def mesh_has_axis(mesh, axis_name):
    return axis_name in tuple(mesh.axis_names)


def auto_model_sharding(mesh, value, axis_name="model"):
    value = jnp.asarray(value)
    if not mesh_has_axis(mesh, axis_name) or value.ndim < 2:
        return replicated_sharding(mesh)
    shard_dim = max(range(value.ndim), key=lambda dim: value.shape[dim])
    spec = [None] * value.ndim
    spec[shard_dim] = axis_name
    return NamedSharding(mesh, PartitionSpec(*spec))


def put_tree_auto_model_parallel(tree, mesh, axis_name="model"):
    def place(value):
        if not hasattr(value, "shape"):
            return value
        return jax.device_put(value, auto_model_sharding(mesh, value, axis_name=axis_name))

    return jax.tree_util.tree_map(place, tree)


def local_data_to_global_array(local_data, sharding, *, global_shape=None):
    return jax.make_array_from_process_local_data(
        sharding,
        local_data,
        global_shape=global_shape,
    )


def host_batch_to_global_arrays(batch, sharding, layout):
    def convert(leaf):
        global_shape = (layout.global_batch_size, *leaf.shape[1:])
        return local_data_to_global_array(leaf, sharding, global_shape=global_shape)

    return jax.tree_util.tree_map(convert, batch)
