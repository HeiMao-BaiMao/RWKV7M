from dataclasses import dataclass

import jax
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

from .mesh import make_1d_mesh
from .partitioning import place_parameter_tree
from .partitioning import mesh_axis_size
from .sharding import (
    data_parallel_sharding,
    mesh_has_axis,
    put_to_devices,
    put_tree_auto_model_parallel,
    replicated_sharding,
)
from ..train.nnx_train import NNXTrainState
from ..model.state import LayerRWKVState, LayerScreenState, ModelScreenState


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


def _place_recurrent_states(tree, mesh, *, data_axis, model_axis):
    """Place carried activations according to the Phase 3 state contract."""
    hidden = NamedSharding(mesh, P(data_axis, model_axis))
    wkv = NamedSharding(mesh, P(data_axis, model_axis, None, None))
    slots = NamedSharding(mesh, P(data_axis, None, model_axis))
    token_state = NamedSharding(mesh, P(data_axis, None))

    if isinstance(tree, tuple):
        return tuple(
            LayerRWKVState(
                time_mix_x=jax.device_put(layer.time_mix_x, hidden),
                channel_mix_x=jax.device_put(layer.channel_mix_x, hidden),
                wkv=jax.device_put(layer.wkv, wkv),
            )
            for layer in tree
        )
    if isinstance(tree, ModelScreenState):
        return ModelScreenState(
            layers=tuple(
                LayerScreenState(
                    slots=jax.device_put(layer.slots, slots),
                    ages=jax.device_put(layer.ages, token_state),
                    usage_ema=jax.device_put(layer.usage_ema, token_state),
                )
                for layer in tree.layers
            )
        )
    raise TypeError(f"unsupported recurrent state type: {type(tree)!r}")


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
    if param_axis_name is not None and param_axis_name != axis_name:
        recurrent_placement = lambda tree: _place_recurrent_states(
            tree,
            mesh,
            data_axis=axis_name,
            model_axis=param_axis_name,
        )
    else:
        recurrent_placement = lambda tree: put_to_devices(tree, batch_sharding)
    initial_rwkv_state = recurrent_placement(runtime.initial_rwkv_state)
    initial_screen_state = recurrent_placement(runtime.initial_screen_state)
    rwkv_state = recurrent_placement(runtime.rwkv_state)
    screen_state = recurrent_placement(runtime.screen_state)

    if isinstance(train_state, NNXTrainState):
        if param_axis_name is not None:
            if not mesh_has_axis(mesh, param_axis_name):
                raise ValueError(
                    f"mesh does not contain parameter axis: {param_axis_name}"
                )
            # The NNX scale path initializes params and Adam state under the
            # target mesh before this placement boundary. Validate that the
            # requested axis is present in logical metadata rather than
            # re-sharding with the legacy path heuristic.
            missing = []
            for path, variable in nnx.to_flat_state(
                nnx.state(train_state.model, nnx.Param)
            ):
                axes = variable.get_metadata().get("out_sharding", ())
                if variable[...].ndim >= 2 and param_axis_name not in axes:
                    name = "/".join(str(part) for part in path)
                    if name.endswith(("/kernel", "/embedding")):
                        missing.append(name)
            if missing:
                if mesh_axis_size(mesh, param_axis_name) != 1:
                    raise ValueError(
                        "NNX model was not initialized for the requested parameter "
                        f"axis {param_axis_name!r}: {missing[:5]}"
                    )
                # A size-one mesh is a local compatibility/smoke case. There
                # is no physical unsharded-to-sharded materialization, so
                # replication is both exact and sufficient.
                model_state = put_to_devices(
                    nnx.state(train_state.model), state_sharding
                )
                optimizer_state = put_to_devices(
                    nnx.state(train_state.optimizer), state_sharding
                )
                nnx.update(train_state.model, model_state)
                nnx.update(train_state.optimizer, optimizer_state)
        else:
            model_state = put_to_devices(nnx.state(train_state.model), state_sharding)
            optimizer_state = put_to_devices(
                nnx.state(train_state.optimizer), state_sharding
            )
            nnx.update(train_state.model, model_state)
            nnx.update(train_state.optimizer, optimizer_state)
        placed_train_state = train_state
        return DistributedTrainObjects(
            train_state=placed_train_state,
            initial_rwkv_state=initial_rwkv_state,
            initial_screen_state=initial_screen_state,
            rwkv_state=rwkv_state,
            screen_state=screen_state,
            mesh=mesh,
            batch_sharding=batch_sharding,
            state_sharding=state_sharding,
        )

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
        rwkv_state=rwkv_state,
        screen_state=screen_state,
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
