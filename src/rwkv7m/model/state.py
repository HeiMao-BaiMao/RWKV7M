from flax import struct
import jax
import jax.numpy as jnp

from .screening import is_v5_semantics


@struct.dataclass
class LayerRWKVState:
    time_mix_x: jnp.ndarray     # [B, d_model]
    channel_mix_x: jnp.ndarray  # [B, d_model]
    wkv: jnp.ndarray            # [B, n_heads, head_size, head_size]


@struct.dataclass
class LayerScreenState:
    slots: jnp.ndarray      # [B, M, d_s]
    ages: jnp.ndarray       # [B, M]
    usage_ema: jnp.ndarray  # [B, M]
    occupancy: jnp.ndarray | None = None  # v5 hard state [B, M]


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
                occupancy=(
                    jnp.zeros(
                        (batch_size, config.n_slots),
                        dtype=jnp.float32,
                    )
                    if is_v5_semantics(config)
                    else None
                ),
            )
        )
    return ModelScreenState(layers=tuple(layer_states))


def init_rwkv_state(batch_size, config):
    layer_states = []
    for _ in range(config.n_layers):
        layer_states.append(
            LayerRWKVState(
                time_mix_x=jnp.zeros((batch_size, config.d_model), dtype=jnp.float32),
                channel_mix_x=jnp.zeros((batch_size, config.d_model), dtype=jnp.float32),
                wkv=jnp.zeros(
                    (
                        batch_size,
                        config.n_heads,
                        config.head_size,
                        config.head_size,
                    ),
                    dtype=jnp.float32,
                ),
            )
        )
    return tuple(layer_states)


def tuple_set(xs, i, x):
    return xs[:i] + (x,) + xs[i + 1 :]


def reset_state_rows(current_state, initial_state, reset_mask):
    """Reset selected batch rows without disturbing other stream lanes."""

    reset_mask = jnp.asarray(reset_mask, dtype=jnp.bool_)
    if reset_mask.ndim != 1:
        raise ValueError("reset_mask must have shape [batch]")

    def reset_leaf(current, initial):
        if current is None:
            return None
        if current.ndim == 0 or current.shape[0] != reset_mask.shape[0]:
            return current
        broadcast_shape = (reset_mask.shape[0],) + (1,) * (current.ndim - 1)
        return jnp.where(
            reset_mask.reshape(broadcast_shape),
            initial,
            current,
        )

    return jax.tree.map(reset_leaf, current_state, initial_state)
