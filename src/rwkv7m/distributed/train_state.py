from dataclasses import dataclass

import jax

from .mesh import make_1d_mesh
from .partitioning import place_parameter_tree
from .sharding import (
    data_parallel_sharding,
    mesh_has_axis,
    put_to_devices,
    put_tree_auto_model_parallel,
    replicated_sharding,
)


@dataclass
class DistributedTrainObjects:
    train_state: object
    initial_rwkv_state: object
    initial_screen_state: object
    rwkv_state: object
    screen_state: object
    mesh: object
    batch_sharding: object
    state_sharding: object


def place_train_objects(
    runtime,
    train_state,
    *,
    mesh=None,
    axis_name="data",
    param_axis_name=None,
):
    mesh = make_1d_mesh(axis_name) if mesh is None else mesh
    state_sharding = replicated_sharding(mesh)
    batch_sharding = data_parallel_sharding(mesh, axis_name=axis_name)
    initial_rwkv_state = put_to_devices(runtime.initial_rwkv_state, batch_sharding)
    initial_screen_state = put_to_devices(runtime.initial_screen_state, batch_sharding)

    if param_axis_name is not None:
        if not mesh_has_axis(mesh, param_axis_name):
            raise ValueError(f"mesh does not contain parameter axis: {param_axis_name}")
        placed_train_state = train_state.replace(
            step=jax.device_put(train_state.step, state_sharding),
            params=place_parameter_tree(
                train_state.params,
                mesh,
                axis_name=param_axis_name,
            ),
            opt_state=put_tree_auto_model_parallel(
                train_state.opt_state,
                mesh,
                axis_name=param_axis_name,
            ),
        )
    else:
        placed_train_state = put_to_devices(train_state, state_sharding)

    return DistributedTrainObjects(
        train_state=placed_train_state,
        initial_rwkv_state=initial_rwkv_state,
        initial_screen_state=initial_screen_state,
        rwkv_state=initial_rwkv_state,
        screen_state=initial_screen_state,
        mesh=mesh,
        batch_sharding=batch_sharding,
        state_sharding=state_sharding,
    )


def replicate_train_objects(runtime, train_state, *, mesh=None, axis_name="data"):
    return place_train_objects(
        runtime,
        train_state,
        mesh=mesh,
        axis_name=axis_name,
        param_axis_name=None,
    )
