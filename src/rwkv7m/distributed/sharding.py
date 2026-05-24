import jax
from jax.sharding import NamedSharding, PartitionSpec


def data_parallel_sharding(mesh, axis_name="data"):
    return NamedSharding(mesh, PartitionSpec(axis_name))


def replicated_sharding(mesh):
    return NamedSharding(mesh, PartitionSpec())


def put_to_devices(tree, sharding):
    return jax.tree_util.tree_map(lambda x: jax.device_put(x, sharding), tree)


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
