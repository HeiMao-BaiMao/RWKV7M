from dataclasses import dataclass

from .mesh import make_1d_mesh
from .sharding import data_parallel_sharding, put_to_devices, replicated_sharding


@dataclass
class DistributedTrainObjects:
    train_state: object
    rwkv_state: object
    screen_state: object
    mesh: object
    batch_sharding: object
    state_sharding: object


def replicate_train_objects(runtime, train_state, *, mesh=None, axis_name="data"):
    mesh = make_1d_mesh(axis_name) if mesh is None else mesh
    state_sharding = replicated_sharding(mesh)
    batch_sharding = data_parallel_sharding(mesh, axis_name=axis_name)
    return DistributedTrainObjects(
        train_state=put_to_devices(train_state, state_sharding),
        rwkv_state=put_to_devices(runtime.initial_rwkv_state, state_sharding),
        screen_state=put_to_devices(runtime.initial_screen_state, state_sharding),
        mesh=mesh,
        batch_sharding=batch_sharding,
        state_sharding=state_sharding,
    )
