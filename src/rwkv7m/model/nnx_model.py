"""Flax NNX implementation of the complete RWKV7M model.

The Linen implementation remains a frozen numerical reference. New runtime,
training, sharding, and checkpoint code uses the classes in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from flax.linen import initializers
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from rwkv7m.kernels.training_loss_backend import (
    resolve_training_loss_backend,
)

from .losses import cross_entropy_components, l2wrap_components
from .rwkv_core import _get_ffn_dim, _time_shift, symmetric_uniform_init
from .screened_rwkv import ModelConfig, _get_model_dtype
from .screening import (
    ScreeningConfig,
    apply_screening_gate,
    bounded_tau,
    compute_slot_delta,
    is_v5_semantics,
    normalize_phase,
    resolve_semantics_version,
    resolve_write_mode,
    theta_from_tau,
    unit_norm,
    update_rate_from_half_life,
    write_mode_uses_write_projection,
)
from .screening_recurrence import (
    ACTIVE_SLOTS,
    ADMISSION_HIGH_RATE,
    ADMISSION_LOW_RATE,
    ADMISSION_MEAN,
    BANK_LONG_WRITE_MASS,
    BANK_MID_WRITE_MASS,
    BANK_SHORT_WRITE_MASS,
    EVICTION_AGE_MEAN,
    EVICTION_USAGE_MEAN,
    MATCHED_ROUTE_MASS,
    NOVEL_RATE,
    NOVEL_ROUTE_MASS,
    READ_MAX,
    READ_MEAN,
    REJECTED_RATE,
    ROUTE_ENTROPY,
    ROUTE_TOP1,
    U_NORM,
    USAGE_MEAN,
    WRITE_EFFECTIVE_MEAN,
    WRITE_MEAN,
    Z_NORM,
    ScreeningRecurrenceConfig,
    screening_recurrence,
    screening_recurrence_sharded,
)
from .state import LayerRWKVState, LayerScreenState, ModelScreenState
from .training_head import tiled_training_loss_components
from .wkv import wkv7, wkv7_sharded


Initializer = Callable[[jax.Array, Sequence[int], jnp.dtype], jax.Array]


def _value(variable):
    """Return an NNX variable's raw JAX value without deprecated coercion."""
    return variable[...] if isinstance(variable, nnx.Variable) else variable


def _dtype_from_name(name: str):
    if name == "float32":
        return jnp.float32
    if name == "bfloat16":
        return jnp.bfloat16
    raise ValueError(f"unsupported dtype: {name!r}")


@dataclass(frozen=True)
class NNXShardingConfig:
    """Concrete data/model mesh contract for NNX initialization and calls."""

    mesh: Mesh
    data_axis: str = "data"
    model_axis: str = "model"

    def named(self, *axes: str | None) -> NamedSharding:
        return NamedSharding(self.mesh, P(*axes))

    @property
    def uses_explicit_axes(self) -> bool:
        return any(
            axis_type == jax.sharding.AxisType.Explicit
            for axis_type in self.mesh.axis_types
        )

    def activation(self, rank: int, *, model_sharded: bool) -> NamedSharding:
        if rank < 1:
            raise ValueError(f"activation rank must be positive, got {rank}")
        axes: list[str | None] = [self.data_axis]
        axes.extend([None] * (rank - 1))
        if model_sharded:
            axes[-1] = self.model_axis
        return self.named(*axes)


class _NNXParallelLinear(nnx.Linear):
    """NNX Linear that makes its output feature placement explicit.

    Row-parallel kernels produce a replicated feature dimension and
    column-parallel kernels produce a model-sharded feature dimension. JAX
    cannot infer the former when both contracting dimensions are sharded, so
    Phase 3 treats the output placement as part of the layer contract.
    """

    def __init__(
        self,
        *args,
        sharding: NNXShardingConfig | None,
        output_model_sharded: bool,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._rwkv7m_sharding = sharding
        self._rwkv7m_output_model_sharded = output_model_sharded

    def __call__(self, inputs, out_sharding=None):
        sharding = self._rwkv7m_sharding
        if out_sharding is None and sharding is not None and sharding.uses_explicit_axes:
            out_sharding = sharding.activation(
                inputs.ndim,
                model_sharded=self._rwkv7m_output_model_sharded,
            )
        return super().__call__(inputs, out_sharding=out_sharding)


def _partitioned_init(
    initializer: Initializer,
    axes: tuple[str | None, ...] | None,
    sharding: NNXShardingConfig | None,
) -> Initializer:
    def storage_safe_init(key, shape, dtype=jnp.float32):
        # QR-backed and some random initializers do not support BF16 directly
        # on every backend. Initialize in FP32 globally, then store in the
        # configured parameter dtype. This also keeps small/large seeds and
        # initializer semantics on one path.
        compute_dtype = jnp.float32 if jnp.dtype(dtype) == jnp.bfloat16 else dtype
        return initializer(key, shape, compute_dtype).astype(dtype)

    if sharding is None or axes is None:
        return storage_safe_init
    return nnx.with_partitioning(storage_safe_init, axes, mesh=sharding.mesh)


def _param(
    rngs: nnx.Rngs,
    initializer: Initializer,
    shape: Sequence[int],
    *,
    axes: tuple[str | None, ...] | None = None,
    sharding: NNXShardingConfig | None = None,
    dtype=jnp.float32,
) -> nnx.Param:
    init = _partitioned_init(initializer, axes, sharding)
    return nnx.Param(init(rngs.params(), tuple(shape), dtype))


def _linear(
    in_features: int,
    out_features: int,
    *,
    rngs: nnx.Rngs,
    use_bias: bool = True,
    kernel_init: Initializer = initializers.lecun_normal(),
    bias_init: Initializer = initializers.zeros_init(),
    kernel_axes: tuple[str | None, ...] | None = None,
    bias_axes: tuple[str | None, ...] | None = None,
    sharding: NNXShardingConfig | None = None,
    dtype=None,
    param_dtype=jnp.float32,
    precision=None,
) -> nnx.Linear:
    output_model_sharded = bool(
        sharding is not None
        and kernel_axes is not None
        and kernel_axes[-1] == sharding.model_axis
    )
    return _NNXParallelLinear(
        in_features,
        out_features,
        sharding=sharding,
        output_model_sharded=output_model_sharded,
        use_bias=use_bias,
        dtype=dtype,
        param_dtype=param_dtype,
        precision=precision,
        kernel_init=_partitioned_init(kernel_init, kernel_axes, sharding),
        bias_init=_partitioned_init(bias_init, bias_axes, sharding),
        rngs=rngs,
    )


def _layer_norm(
    features: int,
    *,
    rngs: nnx.Rngs,
    epsilon: float = 1e-6,
    dtype=None,
    sharding: NNXShardingConfig | None = None,
    param_dtype=jnp.float32,
) -> nnx.LayerNorm:
    axes = (sharding.model_axis,) if sharding is not None else None
    return nnx.LayerNorm(
        features,
        epsilon=epsilon,
        dtype=dtype,
        param_dtype=param_dtype,
        scale_init=_partitioned_init(initializers.ones_init(), axes, sharding),
        bias_init=_partitioned_init(initializers.zeros_init(), axes, sharding),
        rngs=rngs,
    )


def _apply_norm_in_float32(module, x, *, output_dtype=None):
    """Compute normalization statistics in FP32 and restore activation dtype."""
    if output_dtype is None:
        output_dtype = x.dtype
    return module(x.astype(jnp.float32)).astype(output_dtype)


def _constrain_hidden(x, sharding: NNXShardingConfig | None):
    if sharding is None:
        return x
    if x.ndim == 3:
        spec = sharding.named(sharding.data_axis, None, sharding.model_axis)
    elif x.ndim == 2:
        spec = sharding.named(sharding.data_axis, sharding.model_axis)
    else:
        raise ValueError(f"hidden activation must have rank 2 or 3, got {x.ndim}")
    return _apply_activation_sharding(x, spec, sharding)


def _apply_activation_sharding(x, named_sharding, sharding):
    if sharding.uses_explicit_axes:
        return jax.reshard(x, named_sharding)
    return jax.lax.with_sharding_constraint(x, named_sharding)


def _dot_last(
    lhs,
    rhs,
    *,
    sharding: NNXShardingConfig | None,
    output_model_sharded: bool,
    leading_data_axis: bool = True,
    dtype=None,
):
    """Contract the last/first axes with an explicit output placement."""
    if dtype is not None:
        lhs = lhs.astype(dtype)
        rhs = rhs.astype(dtype)
    if sharding is None or not sharding.uses_explicit_axes:
        return lhs @ rhs
    if leading_data_axis:
        out_sharding = sharding.activation(
            lhs.ndim,
            model_sharded=output_model_sharded,
        )
    else:
        axes: list[str | None] = [None] * lhs.ndim
        if output_model_sharded:
            axes[-1] = sharding.model_axis
        out_sharding = sharding.named(*axes)
    return jax.lax.dot_general(
        lhs,
        rhs,
        (((lhs.ndim - 1,), (0,)), ((), ())),
        out_sharding=out_sharding,
    )


def _compute_slot_delta(
    x,
    h_base,
    slot_embed,
    kernel,
    bias,
    sharding: NNXShardingConfig | None,
    *,
    dtype=None,
):
    """Screening delta projection with a model-sharded d_slot output."""
    if sharding is None or not sharding.uses_explicit_axes:
        if dtype is not None:
            x = x.astype(dtype)
            h_base = h_base.astype(dtype)
            slot_embed = slot_embed.astype(dtype)
            kernel = kernel.astype(dtype)
            bias = bias.astype(dtype)
        return compute_slot_delta(x, h_base, slot_embed, kernel, bias)
    C = x.shape[-1]
    d_slot = slot_embed.shape[-1]
    x_kernel = kernel[:C, :]
    h_kernel = kernel[C : 2 * C, :]
    slot_kernel = kernel[2 * C : 2 * C + d_slot, :]
    x_term = _dot_last(
        x,
        x_kernel,
        sharding=sharding,
        output_model_sharded=True,
        dtype=dtype,
    )
    h_term = _dot_last(
        h_base,
        h_kernel,
        sharding=sharding,
        output_model_sharded=True,
        dtype=dtype,
    )
    slot_term = _dot_last(
        slot_embed,
        slot_kernel,
        sharding=sharding,
        output_model_sharded=True,
        leading_data_axis=False,
        dtype=dtype,
    )
    slot_leading_shape = (1,) * (x_term.ndim - 1)
    result = jnp.tanh(
        x_term[..., None, :]
        + h_term[..., None, :]
        + jnp.reshape(slot_term, (*slot_leading_shape, *slot_term.shape))
        + bias
    )
    result_axes: list[str | None] = [sharding.data_axis]
    result_axes.extend([None] * (result.ndim - 2))
    result_axes.append(sharding.model_axis)
    return _apply_activation_sharding(
        result,
        sharding.named(*result_axes),
        sharding,
    )


def _constrain_wkv(x, sharding: NNXShardingConfig | None):
    if sharding is None:
        return x
    return _apply_activation_sharding(
        x,
        sharding.named(sharding.data_axis, sharding.model_axis, None, None),
        sharding,
    )


def _constrain_slots(x, sharding: NNXShardingConfig | None):
    if sharding is None:
        return x
    return _apply_activation_sharding(
        x,
        sharding.named(sharding.data_axis, None, sharding.model_axis),
        sharding,
    )


def _reshape_heads(x, shape, sharding: NNXShardingConfig | None):
    """Reshape hidden features while assigning model shards to RWKV heads."""
    if sharding is None or not sharding.uses_explicit_axes:
        return jnp.reshape(x, shape)
    if len(shape) != 4:
        raise ValueError(f"RWKV head shape must have rank 4, got {shape}")
    return jax.lax.reshape(
        x,
        shape,
        out_sharding=sharding.named(
            sharding.data_axis,
            None,
            sharding.model_axis,
            None,
        ),
    )


def _flatten_heads(x, shape, sharding: NNXShardingConfig | None):
    """Flatten RWKV heads back into the model-sharded hidden dimension."""
    if sharding is None or not sharding.uses_explicit_axes:
        return jnp.reshape(x, shape)
    return jax.lax.reshape(
        x,
        shape,
        out_sharding=sharding.activation(len(shape), model_sharded=True),
    )


def _reshape_hidden(x, shape, sharding: NNXShardingConfig | None):
    """Reshape token axes without changing the hidden-axis contract."""
    if sharding is None or not sharding.uses_explicit_axes:
        return jnp.reshape(x, shape)
    return jax.lax.reshape(
        x,
        shape,
        out_sharding=sharding.activation(len(shape), model_sharded=True),
    )


def _group_norm(
    module: nnx.GroupNorm,
    x,
    sharding: NNXShardingConfig | None,
):
    """Run GroupNorm with FP32 statistics and restore the input dtype."""
    if sharding is None or not sharding.uses_explicit_axes:
        return module(x.astype(jnp.float32)).astype(x.dtype)
    if x.ndim != 2:
        raise ValueError(f"explicit RWKV GroupNorm expects rank 2, got {x.ndim}")
    grouped_sharding = sharding.named(
        sharding.data_axis,
        sharding.model_axis,
        None,
    )
    grouped = jax.lax.reshape(
        x,
        (x.shape[0], module.num_groups, module.group_size),
        out_sharding=grouped_sharding,
    )
    stats = grouped.astype(jnp.promote_types(grouped.dtype, jnp.float32))
    mean = jnp.mean(stats, axis=-1, keepdims=True)
    if module.use_fast_variance:
        mean_square = jnp.mean(jnp.abs(stats) ** 2, axis=-1, keepdims=True)
        variance = jnp.maximum(0.0, mean_square - jnp.abs(mean) ** 2)
    else:
        variance = jnp.mean(
            jnp.abs(stats - mean) ** 2,
            axis=-1,
            keepdims=True,
        )
    normalized = (stats - mean) * jax.lax.rsqrt(variance + module.epsilon)
    parameter_sharding = sharding.named(sharding.model_axis, None)
    if module.scale is not None:
        scale = jax.lax.reshape(
            _value(module.scale),
            (module.num_groups, module.group_size),
            out_sharding=parameter_sharding,
        )
        normalized *= scale[None, :, :]
    if module.bias is not None:
        bias = jax.lax.reshape(
            _value(module.bias),
            (module.num_groups, module.group_size),
            out_sharding=parameter_sharding,
        )
        normalized += bias[None, :, :]
    return jax.lax.reshape(
        normalized,
        x.shape,
        out_sharding=sharding.activation(x.ndim, model_sharded=True),
    ).astype(x.dtype)


class NNXRWKV7TimeMix(nnx.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        *,
        rngs: nnx.Rngs,
        sharding: NNXShardingConfig | None = None,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.sharding = sharding

        C = config.d_model
        H = config.n_heads
        N = config.head_size
        model = sharding.model_axis if sharding is not None else None
        hidden_axes = (None, None, model) if model else None
        mix_axes = (model,) if model else None
        row_axes = (model, None) if model else None
        column_axes = (None, model) if model else None
        compute_dtype = _get_model_dtype(config)
        param_dtype = _dtype_from_name(config.param_dtype)

        ratio_0_to_1 = layer_idx / max(1, config.n_layers - 1)
        ratio_1_to_almost0 = 1.0 - (layer_idx / max(1, config.n_layers))

        def mix_init(power):
            def init_fn(key, shape, dtype=jnp.float32):
                del key
                ddd = jnp.arange(shape[-1], dtype=jnp.float32) / shape[-1]
                return (1.0 - jnp.power(ddd, power * ratio_1_to_almost0)).astype(dtype)

            return init_fn

        self.x_r = _param(rngs, mix_init(0.2), (1, 1, C), axes=mix_axes, sharding=sharding, dtype=param_dtype)
        self.x_w = _param(rngs, mix_init(0.9), (1, 1, C), axes=mix_axes, sharding=sharding, dtype=param_dtype)
        self.x_k = _param(rngs, mix_init(0.7), (1, 1, C), axes=mix_axes, sharding=sharding, dtype=param_dtype)
        self.x_v = _param(rngs, mix_init(0.7), (1, 1, C), axes=mix_axes, sharding=sharding, dtype=param_dtype)
        self.x_a = _param(rngs, mix_init(0.9), (1, 1, C), axes=mix_axes, sharding=sharding, dtype=param_dtype)
        self.x_g = _param(rngs, mix_init(0.2), (1, 1, C), axes=mix_axes, sharding=sharding, dtype=param_dtype)

        self.receptance = _linear(
            C,
            C,
            use_bias=False,
            kernel_init=symmetric_uniform_init(0.5 / math.sqrt(C)),
            kernel_axes=row_axes,
            rngs=rngs,
            sharding=sharding,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )
        d_decay = max(32, int(round((2.5 * math.sqrt(C)) / 32) * 32))
        self.w1 = _param(rngs, initializers.zeros_init(), (C, d_decay), axes=row_axes, sharding=sharding, dtype=param_dtype)
        self.w2 = _param(rngs, initializers.orthogonal(0.1), (d_decay, C), axes=column_axes, sharding=sharding, dtype=param_dtype)

        def w0_init(key, shape, dtype=jnp.float32):
            del key, shape
            zigzag = ((jnp.arange(C) % N) - ((N - 1) / 2)) / ((N - 1) / 2)
            zigzag = zigzag * jnp.abs(zigzag)
            www = -6.0 + 6.0 * jnp.power(
                jnp.arange(C) / max(1, C - 1), 1 + ratio_0_to_1**0.3
            )
            return (www + 0.5 + zigzag * 2.5).reshape(1, 1, C).astype(dtype)

        self.w0 = _param(rngs, w0_init, (1, 1, C), axes=hidden_axes, sharding=sharding, dtype=param_dtype)
        self.key = _linear(
            C,
            C,
            use_bias=False,
            kernel_init=symmetric_uniform_init(0.05 / math.sqrt(C)),
            kernel_axes=row_axes,
            rngs=rngs,
            sharding=sharding,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )
        self.value = _linear(
            C,
            C,
            use_bias=False,
            kernel_init=symmetric_uniform_init(0.5 / math.sqrt(C)),
            kernel_axes=row_axes,
            rngs=rngs,
            sharding=sharding,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )

        if layer_idx > 0:
            d_mv = max(32, int(round((1.7 * math.sqrt(C)) / 32) * 32))
            self.v1 = _param(rngs, initializers.zeros_init(), (C, d_mv), axes=row_axes, sharding=sharding, dtype=param_dtype)
            self.v2 = _param(rngs, initializers.orthogonal(0.1), (d_mv, C), axes=column_axes, sharding=sharding, dtype=param_dtype)

            def v0_init(key, shape, dtype=jnp.float32):
                del key, shape
                linear = jnp.arange(C) / max(1, C - 1) - 0.5
                return (0.73 - linear * 0.4).reshape(1, 1, C).astype(dtype)

            self.v0 = _param(rngs, v0_init, (1, 1, C), axes=hidden_axes, sharding=sharding, dtype=param_dtype)
        else:
            self.v1 = nnx.data(None)
            self.v2 = nnx.data(None)
            self.v0 = nnx.data(None)

        d_aaa = max(32, int(round((2.5 * math.sqrt(C)) / 32) * 32))
        self.a1 = _param(rngs, initializers.zeros_init(), (C, d_aaa), axes=row_axes, sharding=sharding, dtype=param_dtype)
        self.a2 = _param(rngs, initializers.orthogonal(0.1), (d_aaa, C), axes=column_axes, sharding=sharding, dtype=param_dtype)

        def a0_init(key, shape, dtype=jnp.float32):
            del key, shape
            linear = jnp.arange(C) / max(1, C - 1) - 0.5
            zigzag = ((jnp.arange(C) % N) - ((N - 1) / 2)) / ((N - 1) / 2)
            zigzag = zigzag * jnp.abs(zigzag)
            return (-0.19 + zigzag * 0.3 + linear * 0.4).reshape(1, 1, C).astype(dtype)

        self.a0 = _param(rngs, a0_init, (1, 1, C), axes=hidden_axes, sharding=sharding, dtype=param_dtype)
        d_gate = max(32, int(round((5.0 * math.sqrt(C)) / 32) * 32))
        self.g1 = _param(rngs, initializers.zeros_init(), (C, d_gate), axes=row_axes, sharding=sharding, dtype=param_dtype)
        self.g2 = _param(rngs, initializers.orthogonal(0.1), (d_gate, C), axes=column_axes, sharding=sharding, dtype=param_dtype)

        def k_k_init(key, shape, dtype=jnp.float32):
            del key, shape
            linear = jnp.arange(C) / max(1, C - 1) - 0.5
            return (0.71 - linear * 0.1).reshape(1, 1, C).astype(dtype)

        self.k_k = _param(rngs, k_k_init, (1, 1, C), axes=hidden_axes, sharding=sharding, dtype=param_dtype)
        self.k_a = _param(
            rngs,
            initializers.constant(1.02),
            (1, 1, C),
            axes=hidden_axes,
            sharding=sharding,
            dtype=param_dtype,
        )
        vector_axes = (model,) if model else None
        self.ln_x = nnx.GroupNorm(
            C,
            num_groups=H,
            epsilon=64e-5,
            scale_init=_partitioned_init(
                initializers.constant(((1 + layer_idx) / config.n_layers) ** 0.7),
                vector_axes,
                sharding,
            ),
            bias_init=_partitioned_init(initializers.zeros_init(), vector_axes, sharding),
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.r_k = _param(
            rngs,
            initializers.constant(-0.04),
            (H, N),
            axes=(model, None) if model else None,
            sharding=sharding,
            dtype=param_dtype,
        )
        self.output = _linear(
            C,
            C,
            use_bias=False,
            kernel_init=initializers.zeros_init(),
            kernel_axes=column_axes,
            rngs=rngs,
            sharding=sharding,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )

    def __call__(self, x, v_first, state=None):
        B, T, C = x.shape
        H = self.config.n_heads
        N = self.config.head_size
        compute_dtype = _get_model_dtype(self.config)
        if C != H * N:
            raise ValueError(f"d_model={C} must equal n_heads*head_size={H*N}")
        if state is None:
            prev_x = jnp.zeros((B, C), dtype=x.dtype)
            initial_state = jnp.zeros((B, H, N, N), dtype=jnp.float32)
        else:
            prev_x = state.time_mix_x
            initial_state = state.wkv.astype(jnp.float32)
        initial_state = _constrain_wkv(initial_state, self.sharding)

        xx = _time_shift(x, prev_x) - x
        x_r = x + xx * _value(self.x_r)
        x_w = x + xx * _value(self.x_w)
        x_k = x + xx * _value(self.x_k)
        x_v = x + xx * _value(self.x_v)
        x_a = x + xx * _value(self.x_a)
        x_g = x + xx * _value(self.x_g)

        r = self.receptance(x_r)
        w_hidden = _dot_last(
            x_w,
            _value(self.w1),
            sharding=self.sharding,
            output_model_sharded=False,
            dtype=compute_dtype,
        )
        w_raw = _value(self.w0) + _dot_last(
            jnp.tanh(w_hidden),
            _value(self.w2),
            sharding=self.sharding,
            output_model_sharded=True,
            dtype=compute_dtype,
        )
        w_clamped = -jax.nn.softplus(-w_raw) - 0.5
        k = self.key(x_k)
        v = self.value(x_v)
        if self.layer_idx == 0:
            v_first = v
        else:
            v12 = _dot_last(
                _dot_last(
                    x_v,
                    _value(self.v1),
                    sharding=self.sharding,
                    output_model_sharded=False,
                    dtype=compute_dtype,
                ),
                _value(self.v2),
                sharding=self.sharding,
                output_model_sharded=True,
                dtype=compute_dtype,
            )
            v = v + (v_first - v) * jax.nn.sigmoid(_value(self.v0) + v12)

        a_hidden = _dot_last(
            x_a,
            _value(self.a1),
            sharding=self.sharding,
            output_model_sharded=False,
            dtype=compute_dtype,
        )
        a = jax.nn.sigmoid(
            _value(self.a0)
            + _dot_last(
                a_hidden,
                _value(self.a2),
                sharding=self.sharding,
                output_model_sharded=True,
                dtype=compute_dtype,
            )
        )
        g = _dot_last(
            jax.nn.sigmoid(
                _dot_last(
                    x_g,
                    _value(self.g1),
                    sharding=self.sharding,
                    output_model_sharded=False,
                    dtype=compute_dtype,
                )
            ),
            _value(self.g2),
            sharding=self.sharding,
            output_model_sharded=True,
            dtype=compute_dtype,
        )
        kk = k * _value(self.k_k)
        kk_h = _reshape_heads(kk, (B, T, H, N), self.sharding)
        kk_h /= jnp.sqrt(jnp.sum(kk_h * kk_h, axis=-1, keepdims=True) + 1e-12**2)
        kk = _flatten_heads(kk_h, (B, T, C), self.sharding)
        k = k * (1.0 + (a - 1.0) * _value(self.k_a))

        r_h = _reshape_heads(r, (B, T, H, N), self.sharding)
        w_h = _reshape_heads(w_clamped, (B, T, H, N), self.sharding)
        k_h = _reshape_heads(k, (B, T, H, N), self.sharding)
        v_h = _reshape_heads(v, (B, T, H, N), self.sharding)
        neg_kk_h = _reshape_heads(-kk, (B, T, H, N), self.sharding)
        kka_h = _reshape_heads(kk * a, (B, T, H, N), self.sharding)
        inputs = tuple(
            jnp.swapaxes(value, 0, 1)
            for value in (r_h, w_h, k_h, v_h, neg_kk_h, kka_h)
        )
        if self.sharding is None:
            y_h, final_state = wkv7(*inputs, initial_state)
        else:
            y_h, final_state = wkv7_sharded(
                *inputs,
                initial_state,
                mesh=self.sharding.mesh,
                data_axis=self.sharding.data_axis,
                model_axis=self.sharding.model_axis,
            )
        y = _flatten_heads(
            jnp.swapaxes(y_h, 0, 1), (B, T, C), self.sharding
        ).astype(compute_dtype)
        y = _reshape_hidden(
            _group_norm(
                self.ln_x,
                _reshape_hidden(y, (B * T, C), self.sharding),
                self.sharding,
            ),
            (B, T, C),
            self.sharding,
        )
        rk = r_h * k_h * _value(self.r_k)
        recurrent_bonus = _flatten_heads(
            jnp.sum(rk, axis=-1, keepdims=True) * v_h,
            (B, T, C),
            self.sharding,
        )
        y = y + recurrent_bonus
        y = self.output(y * g)
        y = _constrain_hidden(y, self.sharding)
        return y, v_first, x[:, -1, :].astype(jnp.float32), final_state


class NNXRWKV7ChannelMix(nnx.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        *,
        rngs: nnx.Rngs,
        sharding: NNXShardingConfig | None = None,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.sharding = sharding
        C = config.d_model
        model = sharding.model_axis if sharding is not None else None
        compute_dtype = _get_model_dtype(config)
        param_dtype = _dtype_from_name(config.param_dtype)
        ratio = 1.0 - (layer_idx / max(1, config.n_layers))

        def mix_init(key, shape, dtype=jnp.float32):
            del key, shape
            ddd = jnp.arange(C, dtype=jnp.float32) / C
            # The Linen reference initializer historically returns [C] even
            # though the requested shape is [1, 1, C]. Preserve that portable
            # parameter contract; broadcasting supplies the leading axes.
            return (1.0 - jnp.power(ddd, ratio**4)).astype(dtype)

        self.x_k = _param(
            rngs,
            mix_init,
            (1, 1, C),
            axes=(model,) if model else None,
            sharding=sharding,
            dtype=param_dtype,
        )
        d_ffn = _get_ffn_dim(config)
        self.key = _linear(
            C,
            d_ffn,
            use_bias=False,
            kernel_init=symmetric_uniform_init(0.5 / math.sqrt(C)),
            kernel_axes=(model, None) if model else None,
            rngs=rngs,
            sharding=sharding,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )
        self.value = _linear(
            d_ffn,
            C,
            use_bias=False,
            kernel_init=initializers.zeros_init(),
            kernel_axes=(None, model) if model else None,
            rngs=rngs,
            sharding=sharding,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )

    def __call__(self, x, prev_x=None):
        B, _, C = x.shape
        if prev_x is None:
            prev_x = jnp.zeros((B, C), dtype=x.dtype)
        xx = _time_shift(x, prev_x) - x
        k = jax.nn.relu(self.key(x + xx * _value(self.x_k))) ** 2
        value = _constrain_hidden(self.value(k), self.sharding)
        return value, x[:, -1, :].astype(jnp.float32)


class NNXRWKV7Block(nnx.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        *,
        rngs: nnx.Rngs,
        sharding: NNXShardingConfig | None = None,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.sharding = sharding
        self.ln0 = (
            _layer_norm(
                config.d_model,
                epsilon=1e-5,
                rngs=rngs,
                sharding=sharding,
                param_dtype=_dtype_from_name(config.param_dtype),
            )
            if layer_idx == 0
            else nnx.data(None)
        )
        self.ln1 = _layer_norm(
            config.d_model,
            epsilon=1e-5,
            rngs=rngs,
            sharding=sharding,
            param_dtype=_dtype_from_name(config.param_dtype),
        )
        self.ln2 = _layer_norm(
            config.d_model,
            epsilon=1e-5,
            rngs=rngs,
            sharding=sharding,
            param_dtype=_dtype_from_name(config.param_dtype),
        )
        self.att = NNXRWKV7TimeMix(config, layer_idx, rngs=rngs, sharding=sharding)
        self.ffn = NNXRWKV7ChannelMix(config, layer_idx, rngs=rngs, sharding=sharding)

    def __call__(self, x, v_first, rwkv_state=None):
        cfg = self.config
        B = x.shape[0]
        if rwkv_state is None:
            rwkv_state = LayerRWKVState(
                time_mix_x=jnp.zeros((B, cfg.d_model), dtype=jnp.float32),
                channel_mix_x=jnp.zeros((B, cfg.d_model), dtype=jnp.float32),
                wkv=jnp.zeros(
                    (B, cfg.n_heads, cfg.head_size, cfg.head_size),
                    dtype=jnp.float32,
                ),
            )
        if self.layer_idx == 0:
            x = _apply_norm_in_float32(
                self.ln0,
                x,
                output_dtype=_get_model_dtype(cfg),
            )
        x_attn, v_first, time_mix_x, wkv = self.att(
            _apply_norm_in_float32(
                self.ln1,
                x,
                output_dtype=_get_model_dtype(cfg),
            ),
            v_first,
            rwkv_state,
        )
        x = _constrain_hidden(x + x_attn, self.sharding)
        x_ffn, channel_mix_x = self.ffn(
            _apply_norm_in_float32(
                self.ln2,
                x,
                output_dtype=_get_model_dtype(cfg),
            ),
            rwkv_state.channel_mix_x,
        )
        x = _constrain_hidden(x + x_ffn, self.sharding)
        return x, v_first, LayerRWKVState(
            time_mix_x=time_mix_x,
            channel_mix_x=channel_mix_x,
            wkv=wkv,
        )


class NNXStateLevelScreening(nnx.Module):
    def __init__(
        self,
        config: ScreeningConfig,
        *,
        rngs: nnx.Rngs,
        sharding: NNXShardingConfig | None = None,
        compute_dtype=jnp.float32,
        param_dtype=jnp.float32,
    ):
        self.config = config
        self.semantics_version = resolve_semantics_version(config)
        v5_enabled = is_v5_semantics(config)
        if self.semantics_version == "screening-v5-retention":
            raise NotImplementedError(
                "screening-v5-retention is gated on v5-core evaluation"
            )
        self.sharding = sharding
        self.compute_dtype = compute_dtype
        # The v5 slot state is carried across sequence chunks. ROCm may choose
        # shape-dependent reduced-precision FP32 GEMM algorithms by default,
        # which makes the same token projection depend on the chunk length and
        # accumulates visible drift in the recurrent state. HIGH keeps the
        # state-forming projections chunk-stable without changing the v4 fast
        # path or forcing HIGHEST precision over the full model.
        screening_precision = (
            jax.lax.Precision.HIGH if v5_enabled else None
        )

        def screening_linear(*args, **kwargs):
            return _linear(
                *args,
                precision=screening_precision,
                **kwargs,
            )

        C = config.d_model
        model = sharding.model_axis if sharding is not None else None
        row = (model, None) if model else None
        column = (None, model) if model else None
        vector = (model,) if model else None
        slot = (None, model) if model else None
        self.q_proj_r = screening_linear(
            C,
            config.d_k,
            use_bias=False,
            kernel_axes=row,
            rngs=rngs,
            sharding=sharding,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )
        self.k_proj_r = screening_linear(
            config.d_slot,
            config.d_k,
            use_bias=False,
            kernel_axes=row,
            rngs=rngs,
            sharding=sharding,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )
        self.v_proj = screening_linear(
            config.d_slot,
            config.d_v,
            use_bias=False,
            kernel_axes=row,
            rngs=rngs,
            sharding=sharding,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )
        self.out_proj = screening_linear(
            config.d_v,
            C,
            use_bias=False,
            kernel_axes=column,
            rngs=rngs,
            sharding=sharding,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )
        gate_size = C if config.gate_space == "model" else config.d_v
        self.gate_proj = screening_linear(
            C,
            gate_size,
            kernel_axes=row,
            rngs=rngs,
            sharding=sharding,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )
        if config.candidate_rank is None:
            # Keep the portable concatenated kernel shape, but shard its d_slot
            # output. Sharding the concatenated input would make x/h/slot
            # slices cross device boundaries and add avoidable collectives.
            self.delta_proj = screening_linear(
                2 * C + config.d_slot,
                config.d_slot,
                kernel_axes=column,
                bias_axes=vector,
                rngs=rngs,
                sharding=sharding,
                dtype=compute_dtype,
                param_dtype=param_dtype,
            )
            self.delta_context_proj = nnx.data(None)
            self.delta_slot_proj = nnx.data(None)
            self.delta_out_proj = nnx.data(None)
        else:
            rank = config.candidate_rank
            self.delta_proj = nnx.data(None)
            self.delta_context_proj = screening_linear(
                2 * C,
                rank,
                use_bias=False,
                kernel_axes=row,
                rngs=rngs,
                sharding=sharding,
                dtype=compute_dtype,
                param_dtype=param_dtype,
            )
            self.delta_slot_proj = screening_linear(
                config.d_slot,
                rank,
                use_bias=False,
                kernel_axes=row,
                rngs=rngs,
                sharding=sharding,
                dtype=compute_dtype,
                param_dtype=param_dtype,
            )
            self.delta_out_proj = screening_linear(
                rank,
                config.d_slot,
                kernel_axes=column,
                bias_axes=vector,
                rngs=rngs,
                sharding=sharding,
                dtype=compute_dtype,
                param_dtype=param_dtype,
            )
        self.screen_ln = _layer_norm(
            C,
            dtype=jnp.float32,
            rngs=rngs,
            sharding=sharding,
            param_dtype=param_dtype,
        )
        tau_r_shape = () if config.n_read_tiles == 1 else (config.n_read_tiles,)
        if is_v5_semantics(config):
            self.tau_r_raw = nnx.data(None)
            self.tau_r_offset = _param(
                rngs,
                initializers.zeros_init(),
                tau_r_shape,
                sharding=sharding,
                dtype=param_dtype,
            )
        else:
            self.tau_r_raw = _param(
                rngs,
                lambda k, s, d=jnp.float32: jnp.full(
                    s, theta_from_tau(config.tau_init), dtype=d
                ),
                tau_r_shape,
                sharding=sharding,
                dtype=param_dtype,
            )
            self.tau_r_offset = nnx.data(None)
        lambda_init = math.log(math.expm1(config.lambda_screen_init))
        self.lambda_raw = _param(rngs, initializers.constant(lambda_init), (), sharding=sharding, dtype=param_dtype)
        self.slot_embed = _param(rngs, initializers.normal(0.02), (config.n_slots, config.d_slot), axes=slot, sharding=sharding, dtype=param_dtype)
        self.mu_by_bank_raw = _param(rngs, initializers.zeros_init(), (3,), sharding=sharding, dtype=param_dtype)
        if write_mode_uses_write_projection(config):
            self.q_proj_w = screening_linear(
                2 * C,
                config.d_k,
                use_bias=False,
                kernel_axes=row,
                rngs=rngs,
                sharding=sharding,
                dtype=compute_dtype,
                param_dtype=param_dtype,
            )
            self.k_proj_w = screening_linear(
                config.d_slot,
                config.d_k,
                use_bias=False,
                kernel_axes=row,
                rngs=rngs,
                sharding=sharding,
                dtype=compute_dtype,
                param_dtype=param_dtype,
            )
            if is_v5_semantics(config):
                self.tau_w_raw = nnx.data(None)
                self.tau_w_offset = _param(
                    rngs,
                    initializers.zeros_init(),
                    (),
                    sharding=sharding,
                    dtype=param_dtype,
                )
            else:
                self.tau_w_raw = _param(rngs, lambda k, s, d=jnp.float32: jnp.asarray(theta_from_tau(config.tau_init), d), (), sharding=sharding, dtype=param_dtype)
                self.tau_w_offset = nnx.data(None)
        else:
            self.q_proj_w = nnx.data(None)
            self.k_proj_w = nnx.data(None)
            self.tau_w_raw = nnx.data(None)
            self.tau_w_offset = nnx.data(None)
        if config.write_mode == "competitive_novel":
            route_input_size = 2 * C
            admission_bias = math.log(
                config.admission_init / (1.0 - config.admission_init)
            )
            self.admission_proj = screening_linear(
                route_input_size,
                1,
                kernel_axes=row,
                bias_init=initializers.constant(admission_bias),
                rngs=rngs,
                sharding=sharding,
                dtype=compute_dtype,
                param_dtype=param_dtype,
            )
            self.bank_route_proj = screening_linear(
                route_input_size,
                3,
                kernel_axes=row,
                rngs=rngs,
                sharding=sharding,
                dtype=compute_dtype,
                param_dtype=param_dtype,
            )
            if is_v5_semantics(config):
                self.admission_feature_weights = _param(
                    rngs,
                    initializers.zeros_init(),
                    (5,),
                    sharding=sharding,
                    dtype=param_dtype,
                )
            else:
                self.admission_feature_weights = nnx.data(None)
            if is_v5_semantics(config) and config.edit_mode != "tied":
                erase_bias = math.log(
                    config.erase_gate_init / (1.0 - config.erase_gate_init)
                )
                write_bias = math.log(
                    config.write_gate_init / (1.0 - config.write_gate_init)
                )

                def matched_edit_bias(key, shape, dtype=jnp.float32):
                    del key
                    return jnp.concatenate(
                        [
                            jnp.full((config.n_slots,), erase_bias, dtype=dtype),
                            jnp.full((config.n_slots,), write_bias, dtype=dtype),
                        ]
                    )

                def novel_edit_bias(key, shape, dtype=jnp.float32):
                    del key, shape
                    return jnp.asarray([erase_bias, write_bias], dtype=dtype)

                self.matched_edit_proj = screening_linear(
                    route_input_size,
                    2 * config.n_slots,
                    kernel_axes=row,
                    kernel_init=initializers.zeros_init(),
                    bias_init=matched_edit_bias,
                    rngs=rngs,
                    sharding=sharding,
                    dtype=compute_dtype,
                    param_dtype=param_dtype,
                )
                self.novel_edit_proj = screening_linear(
                    route_input_size,
                    2,
                    kernel_axes=row,
                    kernel_init=initializers.zeros_init(),
                    bias_init=novel_edit_bias,
                    rngs=rngs,
                    sharding=sharding,
                    dtype=compute_dtype,
                    param_dtype=param_dtype,
                )
            else:
                self.matched_edit_proj = nnx.data(None)
                self.novel_edit_proj = nnx.data(None)
        else:
            self.admission_proj = nnx.data(None)
            self.bank_route_proj = nnx.data(None)
            self.matched_edit_proj = nnx.data(None)
            self.novel_edit_proj = nnx.data(None)
            self.admission_feature_weights = nnx.data(None)

    def _compute_mu(self):
        cfg = self.config
        mu_max = jnp.array([cfg.mu_short_max, cfg.mu_mid_max, cfg.mu_long_max])
        legacy_mu = mu_max * jax.nn.sigmoid(_value(self.mu_by_bank_raw))
        half_lives = (
            cfg.short_half_life_tokens,
            cfg.mid_half_life_tokens,
            cfg.long_half_life_tokens,
        )
        per_bank = jnp.stack(
            [
                legacy_mu[index]
                if half_life is None
                else update_rate_from_half_life(half_life)
                for index, half_life in enumerate(half_lives)
            ]
        )
        return per_bank[jnp.asarray(cfg.bank_ids)]

    def __call__(
        self,
        x_seq,
        h_base_seq,
        state,
        *,
        phase="read_screening_only",
        deterministic=True,
        training_step=None,
    ):
        cfg = self.config
        phase = normalize_phase(phase)
        v5_enabled = is_v5_semantics(cfg)
        if v5_enabled and phase != "read_write":
            raise ValueError(
                "v5 semantics requires phase='read_write' so occupancy can be "
                "initialized through explicit novel allocation"
            )
        write_mode = resolve_write_mode(cfg, phase)
        slots = _constrain_slots(state.slots.astype(jnp.float32), self.sharding)
        ages = state.ages.astype(jnp.float32)
        usage_ema = state.usage_ema.astype(jnp.float32)
        if v5_enabled and state.occupancy is None:
            raise ValueError(
                "screening-v5 state requires explicit occupancy metadata; "
                "a v4 checkpoint cannot be upgraded implicitly"
            )
        occupancy = (
            jnp.zeros_like(ages)
            if state.occupancy is None
            else state.occupancy.astype(jnp.float32)
        )
        x_ln_seq = _apply_norm_in_float32(
            self.screen_ln,
            x_seq,
            output_dtype=self.compute_dtype,
        )
        q_r_raw = self.q_proj_r(x_ln_seq).astype(jnp.float32)
        q_r_tiled = q_r_raw.reshape(
            *q_r_raw.shape[:-1],
            cfg.n_read_tiles,
            cfg.d_k // cfg.n_read_tiles,
        )
        q_r_seq = unit_norm(q_r_tiled, eps=cfg.eps).reshape(q_r_raw.shape)
        gate_seq = apply_screening_gate(
            self.gate_proj(x_ln_seq).astype(jnp.float32),
            cfg.gate_activation,
        )
        if cfg.candidate_rank is None:
            delta_s_seq = _compute_slot_delta(
                x_ln_seq,
                h_base_seq.astype(jnp.float32),
                _value(self.slot_embed),
                _value(self.delta_proj.kernel),
                _value(self.delta_proj.bias),
                self.sharding,
                dtype=self.compute_dtype,
            )
        else:
            route_context = jnp.concatenate(
                [x_ln_seq, h_base_seq.astype(self.compute_dtype)], axis=-1
            )
            context_latent = self.delta_context_proj(route_context)
            slot_out_sharding = None
            if self.sharding is not None and self.sharding.uses_explicit_axes:
                # ``slot_embed`` has no leading batch/data dimension. Override
                # the parallel Linear default so the replicated slot/latent
                # axes are not mistaken for ``P(data, None)``.
                slot_out_sharding = self.sharding.named(None, None)
            slot_latent = self.delta_slot_proj(
                _value(self.slot_embed),
                out_sharding=slot_out_sharding,
            )
            latent = jax.nn.silu(
                context_latent[..., None, :] + slot_latent[None, None, :, :]
            )
            delta_s_seq = jnp.tanh(self.delta_out_proj(latent))

        initial_read_keys = self.k_proj_r(slots)
        initial_values = self.v_proj(slots)
        delta_read_keys = self.k_proj_r(delta_s_seq)
        delta_values = self.v_proj(delta_s_seq)

        tau_r = (
            _value(self.tau_r_offset).astype(jnp.float32)
            if v5_enabled
            else bounded_tau(_value(self.tau_r_raw)).astype(jnp.float32)
        )
        learned_lambda_screen = jax.nn.softplus(_value(self.lambda_raw)).astype(
            jnp.float32
        )
        lambda_screen_floor = jnp.zeros((), dtype=jnp.float32)
        if (
            v5_enabled
            and training_step is not None
            and cfg.lambda_screen_warmup_floor > 0.0
            and cfg.lambda_screen_warmup_steps > 0
        ):
            lambda_screen_floor = (
                cfg.lambda_screen_warmup_floor
                * jnp.clip(
                    1.0
                    - jnp.asarray(training_step, dtype=jnp.float32)
                    / float(cfg.lambda_screen_warmup_steps),
                    0.0,
                    1.0,
                )
            )
        lambda_screen = jnp.maximum(
            learned_lambda_screen,
            lambda_screen_floor,
        )
        mu = self._compute_mu().astype(jnp.float32)
        uses_write_score = write_mode in (
            "legacy_threshold",
            "competitive_novel",
        )
        if uses_write_score:
            q_w_in = jnp.concatenate(
                [x_ln_seq, h_base_seq.astype(jnp.float32)], axis=-1
            )
            q_w_seq = unit_norm(
                self.q_proj_w(q_w_in).astype(jnp.float32), eps=cfg.eps
            )
            tau_w = (
                _value(self.tau_w_offset).astype(jnp.float32)
                if v5_enabled
                else bounded_tau(_value(self.tau_w_raw)).astype(jnp.float32)
            )
            slot_embed = _value(self.slot_embed)
            if v5_enabled:
                # Explicit occupancy removes the v4 need to perturb zero keys
                # with slot identity. Keeping the identity term here would
                # break P((1-e)s + w*d) consistency when write_mass !=
                # erase_mass.
                initial_write_keys = self.k_proj_w(slots)
                delta_write_keys = self.k_proj_w(delta_s_seq)
            else:
                initial_write_keys = self.k_proj_w(
                    slots + slot_embed[None, :, :]
                )
                delta_write_keys = self.k_proj_w(
                    delta_s_seq + slot_embed[None, None, :, :]
                )
        else:
            q_w_seq = jnp.zeros_like(q_r_seq)
            tau_w = jnp.zeros((), dtype=jnp.float32)
            initial_write_keys = jnp.zeros_like(initial_read_keys)
            delta_write_keys = jnp.zeros_like(delta_read_keys)

        if write_mode == "competitive_novel":
            route_input = jnp.concatenate(
                [x_ln_seq, h_base_seq.astype(self.compute_dtype)], axis=-1
            )
            admission_projection = self.admission_proj(route_input).astype(
                jnp.float32
            )[..., 0]
            admission_seq = (
                admission_projection
                if v5_enabled
                else jax.nn.sigmoid(admission_projection)
            )
            bank_logits_seq = self.bank_route_proj(route_input).astype(
                jnp.float32
            )
            if v5_enabled and cfg.edit_mode != "tied":
                matched_edit_logits = self.matched_edit_proj(
                    route_input
                ).astype(jnp.float32)
                matched_erase_logits, matched_write_logits = jnp.split(
                    matched_edit_logits,
                    2,
                    axis=-1,
                )
                novel_edit_logits = self.novel_edit_proj(route_input).astype(
                    jnp.float32
                )
                novel_erase_logits = novel_edit_logits[..., 0]
                novel_write_logits = novel_edit_logits[..., 1]
            else:
                matched_erase_logits = jnp.zeros(
                    (*q_r_seq.shape[:2], cfg.n_slots),
                    dtype=jnp.float32,
                )
                matched_write_logits = jnp.zeros_like(
                    matched_erase_logits
                )
                novel_erase_logits = jnp.zeros(
                    q_r_seq.shape[:2],
                    dtype=jnp.float32,
                )
                novel_write_logits = jnp.zeros_like(novel_erase_logits)
        else:
            admission_seq = jnp.zeros(q_r_seq.shape[:2], dtype=jnp.float32)
            bank_logits_seq = jnp.zeros(
                (*q_r_seq.shape[:2], 3), dtype=jnp.float32
            )
            matched_erase_logits = jnp.zeros(
                (*q_r_seq.shape[:2], cfg.n_slots),
                dtype=jnp.float32,
            )
            matched_write_logits = jnp.zeros_like(matched_erase_logits)
            novel_erase_logits = jnp.zeros(
                q_r_seq.shape[:2],
                dtype=jnp.float32,
            )
            novel_write_logits = jnp.zeros_like(novel_erase_logits)

        projected_inputs = tuple(
            jnp.swapaxes(value, 0, 1)
            for value in (
                q_r_seq,
                q_w_seq,
                delta_s_seq,
                delta_read_keys,
                delta_values,
                delta_write_keys,
            )
        )
        if v5_enabled:
            from .screening_v5 import (
                ScreeningV5RecurrenceConfig,
                screening_v5_recurrence_reference,
            )

            # Keep the complete state-forming recurrence boundary in FP32.
            # Casting BF16 projection results inside individual scan steps is
            # forward-equivalent, but leaves their reverse-scan cotangents on
            # a mixed-precision path. Learned, weak routes can make that path
            # non-finite at production context lengths on ROCm. Projection
            # GEMMs and the module output remain in the configured compute
            # dtype; only recurrent state/cotangent transport is promoted.
            v5_projected_inputs = tuple(
                value.astype(jnp.float32) for value in projected_inputs
            )

            v5_config = ScreeningV5RecurrenceConfig(
                use_value_unit_norm=cfg.use_value_unit_norm,
                usage_ema_decay=cfg.usage_ema_decay,
                tanh_norm_cap=cfg.tanh_norm_cap,
                eps=cfg.eps,
                bank_ids=cfg.bank_ids,
                route_power=cfg.route_power,
                novelty_temperature=cfg.novelty_temperature,
                admission_threshold=(
                    cfg.admission_threshold
                    if cfg.admission_threshold is not None
                    else 0.5
                ),
                allocation_temperature=cfg.allocation_temperature,
                bank_route_temperature=cfg.bank_route_temperature,
                allocation_age_weight=cfg.allocation_age_weight,
                allocation_usage_weight=cfg.allocation_usage_weight,
                allocation_redundancy_weight=(
                    cfg.allocation_redundancy_weight
                ),
                n_read_tiles=cfg.n_read_tiles,
                tau_min=cfg.tau_min,
                tau_max=cfg.tau_max,
                target_false_read_rate=cfg.target_false_read_rate,
                target_false_write_rate=cfg.target_false_write_rate,
                target_false_match_rate=cfg.target_false_match_rate,
                threshold_warmup_by_load=cfg.threshold_warmup_by_load,
                threshold_warmup_tau=cfg.threshold_warmup_tau,
                eta_ambiguity=cfg.eta_ambiguity,
                edit_mode=cfg.edit_mode,
                write_accounting_floor=cfg.write_accounting_floor,
            )
            recurrence_outputs = screening_v5_recurrence_reference(
                v5_projected_inputs[0],
                v5_projected_inputs[1],
                jnp.swapaxes(admission_seq, 0, 1),
                _value(self.admission_feature_weights).astype(jnp.float32),
                jnp.swapaxes(bank_logits_seq, 0, 1),
                jnp.swapaxes(matched_erase_logits, 0, 1),
                jnp.swapaxes(matched_write_logits, 0, 1),
                jnp.swapaxes(novel_erase_logits, 0, 1),
                jnp.swapaxes(novel_write_logits, 0, 1),
                *v5_projected_inputs[2:],
                slots,
                initial_read_keys.astype(jnp.float32),
                initial_values.astype(jnp.float32),
                initial_write_keys.astype(jnp.float32),
                ages,
                usage_ema,
                occupancy,
                mu,
                tau_r,
                tau_w,
                v5_config,
            )
            (
                u_time,
                final_slots,
                final_ages,
                final_usage,
                final_occupancy,
                step_statistics,
                update_squared,
            ) = recurrence_outputs
        else:
            recurrence_config = ScreeningRecurrenceConfig(
                write_enabled=uses_write_score,
                use_value_unit_norm=cfg.use_value_unit_norm,
                use_leaky_warmup=cfg.use_leaky_warmup,
                leaky_alpha=cfg.leaky_alpha,
                leaky_gamma=cfg.leaky_gamma,
                use_age_mask=cfg.use_age_mask,
                age_ref=cfg.age_ref,
                age_sigma=cfg.age_sigma,
                write_rel_floor=cfg.write_rel_floor,
                usage_ema_decay=cfg.usage_ema_decay,
                tanh_norm_cap=cfg.tanh_norm_cap,
                eps=cfg.eps,
                write_mode=write_mode,
                bank_ids=cfg.bank_ids,
                route_power=cfg.route_power,
                novelty_threshold=cfg.novelty_threshold,
                novelty_temperature=cfg.novelty_temperature,
                allocation_temperature=cfg.allocation_temperature,
                bank_route_temperature=cfg.bank_route_temperature,
                allocation_age_weight=cfg.allocation_age_weight,
                allocation_usage_weight=cfg.allocation_usage_weight,
                hard_admission=(write_mode == "competitive_novel"),
                admission_threshold=(
                    cfg.admission_threshold
                    if cfg.admission_threshold is not None
                    else 0.5
                ),
                n_read_tiles=cfg.n_read_tiles,
                checkpoint_interval=cfg.checkpoint_interval,
            )
            recurrence_inputs = (
                projected_inputs[0],
                projected_inputs[1],
                jnp.swapaxes(admission_seq, 0, 1),
                jnp.swapaxes(bank_logits_seq, 0, 1),
                *projected_inputs[2:],
            )
            state_inputs = (
                slots,
                initial_read_keys,
                initial_values,
                initial_write_keys,
                ages,
                usage_ema,
                mu,
                tau_r,
                tau_w,
            )
            if self.sharding is None:
                recurrence_outputs = screening_recurrence(
                    *recurrence_inputs,
                    *state_inputs,
                    recurrence_config,
                )
            else:
                recurrence_outputs = screening_recurrence_sharded(
                    *recurrence_inputs,
                    *state_inputs,
                    recurrence_config,
                    mesh=self.sharding.mesh,
                    data_axis=self.sharding.data_axis,
                    model_axis=self.sharding.model_axis,
                )
            (
                u_time,
                final_slots,
                final_ages,
                final_usage,
                step_statistics,
                update_squared,
            ) = recurrence_outputs
            final_occupancy = state.occupancy

        u_seq = jnp.swapaxes(u_time, 0, 1)
        effective_lambda = lambda_screen / math.sqrt(cfg.n_read_tiles)
        if cfg.gate_space == "value":
            read_out_seq = self.out_proj(
                u_seq * gate_seq.astype(u_seq.dtype)
            )
            memory_branch = read_out_seq.astype(h_base_seq.dtype)
        else:
            read_out_seq = self.out_proj(u_seq)
            memory_branch = (
                gate_seq * read_out_seq.astype(jnp.float32)
            ).astype(h_base_seq.dtype)
        h_seq = (
            h_base_seq + effective_lambda * memory_branch
        ).astype(h_base_seq.dtype)
        screening_residual = (
            effective_lambda * memory_branch.astype(jnp.float32)
        )
        screening_residual_rms = jnp.sqrt(
            jnp.mean(screening_residual**2)
        )
        base_residual_rms = jnp.sqrt(
            jnp.mean(h_base_seq.astype(jnp.float32) ** 2)
        )

        final_slots_normalized = unit_norm(final_slots, eps=cfg.eps)
        if self.sharding is not None and self.sharding.uses_explicit_axes:
            slot_similarity = jax.lax.dot_general(
                final_slots_normalized,
                final_slots_normalized,
                (((2,), (2,)), ((0,), (0,))),
                out_sharding=self.sharding.named(
                    self.sharding.data_axis,
                    None,
                    None,
                ),
            )
        else:
            slot_similarity = jnp.einsum(
                "bms,bns->bmn",
                final_slots_normalized,
                final_slots_normalized,
            )
        slot_count = final_slots.shape[1]
        off_diagonal = 1.0 - jnp.eye(slot_count, dtype=jnp.float32)
        if v5_enabled:
            occupancy_pair = (
                final_occupancy[:, :, None] * final_occupancy[:, None, :]
            )
            redundancy_mask = off_diagonal[None, :, :] * occupancy_pair
            redundancy_denominator = jnp.maximum(
                jnp.sum(redundancy_mask),
                1.0,
            )
        else:
            redundancy_mask = off_diagonal[None, :, :]
            redundancy_denominator = (
                final_slots.shape[0]
                * max(slot_count * (slot_count - 1), 1)
            )

        stats = {
            "rel_read_mean": jnp.mean(step_statistics[..., READ_MEAN]),
            "rel_read_max": jnp.mean(
                jnp.max(step_statistics[..., READ_MAX], axis=1)
            ),
            "active_slots_mean": jnp.mean(
                step_statistics[..., ACTIVE_SLOTS]
            ),
            "z_norm_mean": jnp.mean(step_statistics[..., Z_NORM]),
            "u_norm_mean": jnp.mean(step_statistics[..., U_NORM]),
            "lambda_screen": effective_lambda,
            "rel_write_mean": jnp.mean(
                step_statistics[..., WRITE_MEAN]
            ),
            "rel_write_effective_mean": jnp.mean(
                step_statistics[..., WRITE_EFFECTIVE_MEAN]
            ),
            "slot_update_norm_mean": jnp.mean(jnp.sqrt(update_squared)),
            "slot_usage_ema_mean": jnp.mean(
                step_statistics[..., USAGE_MEAN]
            ),
            "matched_route_mass": jnp.mean(
                step_statistics[..., MATCHED_ROUTE_MASS]
            ),
            "novel_route_mass": jnp.mean(
                step_statistics[..., NOVEL_ROUTE_MASS]
            ),
            "route_entropy": jnp.mean(
                step_statistics[..., ROUTE_ENTROPY]
            ),
            "route_top1_concentration": jnp.mean(
                step_statistics[..., ROUTE_TOP1]
            ),
            "admission_mean": jnp.mean(
                step_statistics[..., ADMISSION_MEAN]
            ),
            "admission_low_rate": jnp.mean(
                step_statistics[..., ADMISSION_LOW_RATE]
            ),
            "admission_high_rate": jnp.mean(
                step_statistics[..., ADMISSION_HIGH_RATE]
            ),
            "novel_token_rate": jnp.mean(
                step_statistics[..., NOVEL_RATE]
            ),
            "rejected_write_rate": jnp.mean(
                step_statistics[..., REJECTED_RATE]
            ),
            "short_bank_write_mass": jnp.mean(
                step_statistics[..., BANK_SHORT_WRITE_MASS]
            ),
            "mid_bank_write_mass": jnp.mean(
                step_statistics[..., BANK_MID_WRITE_MASS]
            ),
            "long_bank_write_mass": jnp.mean(
                step_statistics[..., BANK_LONG_WRITE_MASS]
            ),
            "eviction_age_mean": jnp.mean(
                step_statistics[..., EVICTION_AGE_MEAN]
            ),
            "eviction_usage_mean": jnp.mean(
                step_statistics[..., EVICTION_USAGE_MEAN]
            ),
            "slot_cosine_redundancy": jnp.sum(
                jnp.abs(slot_similarity) * redundancy_mask
            )
            / redundancy_denominator,
        }
        if v5_enabled:
            from .screening_v5 import (
                ACCEPTED_NOVEL_RATE,
                EMPTY_ALLOCATION_RATE,
                MATCHED_ERASE_MASS,
                MATCHED_WRITE_MASS,
                NOVEL_ERASE_MASS,
                NOVEL_WRITE_MASS,
                OCCUPIED_EVICTION_RATE,
                READ_ENERGY_MEAN,
                TAU_NOVEL_MEAN,
                TAU_READ_MEAN,
                TAU_WRITE_MEAN,
                WRITE_BUDGET_RATE,
                WRITE_SATURATION_RATE,
            )

            stats.update(
                {
                    "tau_r": jnp.mean(
                        step_statistics[..., TAU_READ_MEAN]
                    ),
                    "tau_r_min": jnp.min(
                        step_statistics[..., TAU_READ_MEAN]
                    ),
                    "tau_r_max": jnp.max(
                        step_statistics[..., TAU_READ_MEAN]
                    ),
                    "tau_w": jnp.mean(
                        step_statistics[..., TAU_WRITE_MEAN]
                    ),
                    "tau_novel": jnp.mean(
                        step_statistics[..., TAU_NOVEL_MEAN]
                    ),
                    "slot_utilization": jnp.mean(final_occupancy),
                    "dead_slot_rate": jnp.mean(final_occupancy <= 0.5),
                    "matched_erase_mass": jnp.mean(
                        step_statistics[..., MATCHED_ERASE_MASS]
                    ),
                    "matched_write_mass": jnp.mean(
                        step_statistics[..., MATCHED_WRITE_MASS]
                    ),
                    "novel_erase_mass": jnp.mean(
                        step_statistics[..., NOVEL_ERASE_MASS]
                    ),
                    "novel_write_mass": jnp.mean(
                        step_statistics[..., NOVEL_WRITE_MASS]
                    ),
                    "accepted_novel_rate": jnp.mean(
                        step_statistics[..., ACCEPTED_NOVEL_RATE]
                    ),
                    "empty_allocation_rate": jnp.mean(
                        step_statistics[..., EMPTY_ALLOCATION_RATE]
                    ),
                    "occupied_eviction_rate": jnp.mean(
                        step_statistics[..., OCCUPIED_EVICTION_RATE]
                    ),
                    "read_energy_mean": jnp.mean(
                        step_statistics[..., READ_ENERGY_MEAN]
                    ),
                    "write_saturation_rate": jnp.mean(
                        step_statistics[..., WRITE_SATURATION_RATE]
                    ),
                    "write_budget_rate": jnp.mean(
                        step_statistics[..., WRITE_BUDGET_RATE]
                    ),
                    "screening_residual_rms": screening_residual_rms,
                    "base_residual_rms": base_residual_rms,
                    "screening_base_rms_ratio": screening_residual_rms
                    / (base_residual_rms + cfg.eps),
                    "lambda_screen_learned": learned_lambda_screen
                    / math.sqrt(cfg.n_read_tiles),
                    "lambda_screen_floor": lambda_screen_floor
                    / math.sqrt(cfg.n_read_tiles),
                }
            )
        else:
            stats.update(
                {
                    "tau_r": jnp.mean(tau_r),
                    "tau_r_min": jnp.min(tau_r),
                    "tau_r_max": jnp.max(tau_r),
                    "tau_w": tau_w,
                    "slot_utilization": jnp.mean(final_usage > 1e-3),
                    "dead_slot_rate": jnp.mean(final_usage <= 1e-3),
                }
            )
        new_state = LayerScreenState(
            slots=_constrain_slots(final_slots, self.sharding).astype(
                state.slots.dtype
            ),
            ages=final_ages,
            usage_ema=final_usage.astype(state.usage_ema.dtype),
            occupancy=(
                final_occupancy.astype(jnp.float32)
                if v5_enabled
                else state.occupancy
            ),
        )
        return h_seq, new_state, stats


class NNXScreenedRWKVLayer(nnx.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        *,
        rngs: nnx.Rngs,
        sharding: NNXShardingConfig | None = None,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self._block_name = f"rwkv_block_{layer_idx}"
        setattr(
            self,
            self._block_name,
            NNXRWKV7Block(config, layer_idx, rngs=rngs, sharding=sharding),
        )
        self._screening_name = f"screening_{layer_idx}"
        self._has_screening = (
            config.use_screening and layer_idx in config.screening.screened_layers
        )
        if self._has_screening:
            setattr(
                self,
                self._screening_name,
                NNXStateLevelScreening(
                    config.screening,
                    rngs=rngs,
                    sharding=sharding,
                    compute_dtype=_get_model_dtype(config),
                    param_dtype=_dtype_from_name(config.param_dtype),
                ),
            )

    def __call__(
        self,
        x,
        v_first,
        rwkv_state,
        screen_state,
        *,
        phase,
        deterministic,
        training_step=None,
    ):
        block = getattr(self, self._block_name)
        h_base, v_first, new_rwkv_state = block(x, v_first, rwkv_state)
        if self._has_screening:
            screening = getattr(self, self._screening_name)
            h, new_screen, stats = screening(
                x,
                h_base,
                screen_state,
                phase=phase,
                deterministic=deterministic,
                training_step=training_step,
            )
        else:
            h, new_screen, stats = h_base, screen_state, {}
        return h, v_first, new_rwkv_state, new_screen, stats


def _call_screened_layer(
    layer,
    x,
    v_first,
    rwkv_state,
    screen_state,
    phase,
    deterministic,
    training_step,
):
    return layer(
        x,
        v_first,
        rwkv_state,
        screen_state,
        phase=phase,
        deterministic=deterministic,
        training_step=training_step,
    )


_remat_screened_layer = nnx.remat(
    _call_screened_layer,
    static_argnums=(5, 6),
)


class NNXScreenedRWKVModel(nnx.Module):
    """Complete NNX RWKV7M model with explicit parameter/activation sharding."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nnx.Rngs,
        sharding: NNXShardingConfig | None = None,
    ):
        self.config = config
        self.sharding = sharding
        model = sharding.model_axis if sharding is not None else None
        param_dtype = _dtype_from_name(config.param_dtype)
        self.token_embedding = nnx.Embed(
            config.vocab_size,
            config.d_model,
            dtype=_get_model_dtype(config),
            param_dtype=param_dtype,
            embedding_init=_partitioned_init(
                symmetric_uniform_init(1e-4),
                (None, model) if model else None,
                sharding,
            ),
            rngs=rngs,
        )
        for layer_idx in range(config.n_layers):
            setattr(
                self,
                f"layer_{layer_idx}",
                NNXScreenedRWKVLayer(
                    config,
                    layer_idx,
                    rngs=rngs,
                    sharding=sharding,
                ),
            )
        self.final_ln = _layer_norm(
            config.d_model,
            epsilon=1e-5,
            rngs=rngs,
            sharding=sharding,
            param_dtype=param_dtype,
        )
        head_gain = (
            0.5 * math.sqrt(config.vocab_size / config.d_model)
            if config.vocab_size > config.d_model
            else 0.5
        )
        if config.lm_head_init == "orthogonal":
            head_init = initializers.orthogonal(head_gain)
        else:
            head_init = initializers.variance_scaling(
                head_gain * head_gain, "fan_in", "truncated_normal"
            )
        head_axes = None
        if model is not None:
            head_axes = (None, model) if config.vocab_parallel else (model, None)
        self.lm_head = _linear(
            config.d_model,
            config.vocab_size,
            use_bias=False,
            kernel_init=head_init,
            kernel_axes=head_axes,
            rngs=rngs,
            sharding=sharding,
            dtype=_get_model_dtype(config),
            param_dtype=param_dtype,
        )

    def compute_recurrent_hidden(
        self,
        input_ids,
        rwkv_state,
        screen_state,
        *,
        phase="read_screening_only",
        deterministic=True,
        training_step=None,
    ):
        cfg = self.config
        phase = normalize_phase(phase)
        batch_size = input_ids.shape[0]
        embedding_out_sharding = None
        if self.sharding is not None and self.sharding.uses_explicit_axes:
            embedding_out_sharding = self.sharding.activation(
                input_ids.ndim + 1,
                model_sharded=True,
            )
        x = _constrain_hidden(
            self.token_embedding(
                input_ids,
                out_sharding=embedding_out_sharding,
            ).astype(_get_model_dtype(cfg)),
            self.sharding,
        )
        screened_idx = {
            layer_id: index
            for index, layer_id in enumerate(cfg.screening.screened_layers)
        }
        v_first = jnp.zeros_like(x)
        new_rwkv_layers = list(rwkv_state)
        new_screen_layers = list(screen_state.layers)
        all_stats = []
        for layer_idx in range(cfg.n_layers):
            if layer_idx in screened_idx:
                scr_state = screen_state.layers[screened_idx[layer_idx]]
            else:
                scr_state = LayerScreenState(
                    slots=jnp.zeros((batch_size, 1, 1), dtype=jnp.float32),
                    ages=jnp.zeros((batch_size, 1), dtype=jnp.float32),
                    usage_ema=jnp.zeros((batch_size, 1), dtype=jnp.float32),
                )
            layer = getattr(self, f"layer_{layer_idx}")
            layer_call = (
                _remat_screened_layer if cfg.remat_blocks else _call_screened_layer
            )
            x, v_first, new_rwkv, new_screen, stats = layer_call(
                layer,
                x,
                v_first,
                rwkv_state[layer_idx],
                scr_state,
                phase,
                deterministic,
                training_step,
            )
            new_rwkv_layers[layer_idx] = new_rwkv
            if layer_idx in screened_idx:
                new_screen_layers[screened_idx[layer_idx]] = new_screen
            all_stats.append(stats)
        keys = {key for stats in all_stats for key in stats}
        agg_stats = {
            key: jnp.mean(
                jnp.asarray([stats[key] for stats in all_stats if key in stats])
            )
            for key in keys
        }
        return (
            x,
            tuple(new_rwkv_layers),
            ModelScreenState(layers=tuple(new_screen_layers)),
            agg_stats,
        )

    def compute_logits(self, hidden):
        """Apply the final norm and LM head to recurrent hidden activations."""
        if hidden.shape[-1] != self.config.d_model:
            raise ValueError(
                "hidden final dimension must equal config.d_model, got "
                f"{hidden.shape[-1]} and {self.config.d_model}"
            )
        normalized = _apply_norm_in_float32(
            self.final_ln,
            hidden,
            output_dtype=_get_model_dtype(self.config),
        )
        return self.lm_head(normalized)

    def compute_training_loss(
        self,
        hidden,
        targets,
        mask=None,
        *,
        target_sharding=None,
    ):
        """Return reducible CE/L2 components without exposing training logits."""
        tile_size = self.config.training_vocab_tile_size
        backend = (
            resolve_training_loss_backend()
            if tile_size is not None
            else "reference"
        )
        vocab_size = self.config.vocab_size
        use_tiled_head = (
            tile_size is not None
            and backend != "reference"
            and self.sharding is None
            and (vocab_size <= tile_size or vocab_size % tile_size == 0)
        )
        if use_tiled_head:
            normalized = _apply_norm_in_float32(
                self.final_ln,
                hidden,
                output_dtype=_get_model_dtype(self.config),
            )
            loss_mask = (
                jnp.ones(targets.shape, dtype=jnp.float32)
                if mask is None
                else mask
            )
            ce_total, ce_count, l2_total, l2_count = (
                tiled_training_loss_components(
                    normalized,
                    _value(self.lm_head.kernel),
                    targets,
                    loss_mask,
                    int(tile_size),
                    backend,
                )
            )
            return {
                "ce_total": ce_total,
                "ce_count": ce_count,
                "l2_total": l2_total,
                "l2_count": l2_count,
            }
        logits = self.compute_logits(hidden)
        ce_total, ce_count = cross_entropy_components(
            logits,
            targets,
            mask,
            target_sharding=target_sharding,
        )
        l2_total, l2_count = l2wrap_components(logits)
        return {
            "ce_total": ce_total,
            "ce_count": ce_count,
            "l2_total": l2_total,
            "l2_count": l2_count,
        }

    def __call__(
        self,
        input_ids,
        rwkv_state,
        screen_state,
        *,
        phase="read_screening_only",
        deterministic=True,
        training_step=None,
    ):
        hidden, new_rwkv, new_screen, stats = self.compute_recurrent_hidden(
            input_ids,
            rwkv_state,
            screen_state,
            phase=phase,
            deterministic=deterministic,
            training_step=training_step,
        )
        return (
            self.compute_logits(hidden),
            new_rwkv,
            new_screen,
            stats,
        )

    def apply(self, variables, *args, **kwargs):
        """Small compatibility adapter for former runtime call sites.

        New code should call the NNX module directly. The adapter accepts an
        ``nnx.State`` params entry so existing inference tooling can migrate
        without reintroducing Linen's variables/apply ownership model.
        """

        params = variables.get("params") if variables is not None else None
        if params is None:
            return self(*args, **kwargs)
        if not isinstance(params, nnx.State):
            raise TypeError(
                "NNX apply expects variables['params'] to be nnx.State; "
                "load portable dictionaries with load_linen_params_into_nnx()"
            )
        graphdef, _ = nnx.split(self, nnx.Param)
        return nnx.merge(graphdef, params)(*args, **kwargs)


def initialize_nnx_model(
    rng_key,
    config: ModelConfig,
    *,
    sharding: NNXShardingConfig | None = None,
) -> NNXScreenedRWKVModel:
    """Initialize NNX parameters directly in their target NamedShardings."""

    if sharding is None:
        return NNXScreenedRWKVModel(config, rngs=nnx.Rngs(params=rng_key))

    @jax.jit
    def initialize(key):
        return NNXScreenedRWKVModel(
            config,
            rngs=nnx.Rngs(params=key),
            sharding=sharding,
        )

    with jax.set_mesh(sharding.mesh):
        return initialize(rng_key)


__all__ = [
    "NNXShardingConfig",
    "NNXRWKV7TimeMix",
    "NNXRWKV7ChannelMix",
    "NNXRWKV7Block",
    "NNXStateLevelScreening",
    "NNXScreenedRWKVLayer",
    "NNXScreenedRWKVModel",
    "initialize_nnx_model",
]
