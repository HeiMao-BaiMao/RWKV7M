import jax
from jax.sharding import NamedSharding, PartitionSpec


def data_parallel_sharding(mesh, axis_name="data"):
    return NamedSharding(mesh, PartitionSpec(axis_name))


def replicated_sharding(mesh):
    return NamedSharding(mesh, PartitionSpec())


def put_to_devices(tree, sharding):
    return jax.tree_util.tree_map(lambda x: jax.device_put(x, sharding), tree)
