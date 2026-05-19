from flax import struct
import jax.numpy as jnp


@struct.dataclass
class LayerScreenState:
    slots: jnp.ndarray      # [B, M, d_s]
    ages: jnp.ndarray       # [B, M]
    usage_ema: jnp.ndarray  # [B, M]


@struct.dataclass
class ModelScreenState:
    layers: tuple  # tuple[LayerScreenState, ...] length = len(screened_layers)


def init_screen_state(batch_size, config):
    layer_states = []
    for _ in config.screened_layers:
        layer_states.append(
            LayerScreenState(
                slots=jnp.zeros((batch_size, config.n_slots, config.d_slot), dtype=jnp.float32),
                ages=jnp.zeros((batch_size, config.n_slots), dtype=jnp.float32),
                usage_ema=jnp.zeros((batch_size, config.n_slots), dtype=jnp.float32),
            )
        )
    return ModelScreenState(layers=tuple(layer_states))


def tuple_set(xs, i, x):
    return xs[:i] + (x,) + xs[i + 1 :]
