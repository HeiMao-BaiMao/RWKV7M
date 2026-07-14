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

from .losses import cross_entropy_components, l2wrap_components
from .rwkv_core import _get_ffn_dim, _time_shift, symmetric_uniform_init
from .screened_rwkv import ModelConfig, _get_model_dtype
from .screening import (
    ScreeningConfig,
    bounded_tau,
    compute_slot_delta,
    normalize_phase,
    tanh_norm,
    theta_from_tau,
    trim_square,
    unit_norm,
    update_rate_from_half_life,
)
from .state import LayerRWKVState, LayerScreenState, ModelScreenState
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
    return jnp.tanh(
        x_term[:, None, :]
        + h_term[:, None, :]
        + slot_term[None, :, :]
        + bias
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
        self.sharding = sharding
        self.compute_dtype = compute_dtype
        C = config.d_model
        model = sharding.model_axis if sharding is not None else None
        row = (model, None) if model else None
        column = (None, model) if model else None
        vector = (model,) if model else None
        slot = (None, model) if model else None
        self.q_proj_r = _linear(
            C,
            config.d_k,
            use_bias=False,
            kernel_axes=row,
            rngs=rngs,
            sharding=sharding,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )
        self.k_proj_r = _linear(
            config.d_slot,
            config.d_k,
            use_bias=False,
            kernel_axes=row,
            rngs=rngs,
            sharding=sharding,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )
        self.v_proj = _linear(
            config.d_slot,
            config.d_v,
            use_bias=False,
            kernel_axes=row,
            rngs=rngs,
            sharding=sharding,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )
        self.out_proj = _linear(
            config.d_v,
            C,
            use_bias=False,
            kernel_axes=column,
            rngs=rngs,
            sharding=sharding,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )
        self.gate_proj = _linear(
            C,
            C,
            kernel_axes=row,
            rngs=rngs,
            sharding=sharding,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )
        # Keep the portable concatenated kernel shape, but shard its d_slot
        # output. Sharding the concatenated input would make x/h/slot slices
        # cross device boundaries and introduce avoidable reshard collectives.
        self.delta_proj = _linear(
            2 * C + config.d_slot,
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
        self.tau_r_raw = _param(rngs, lambda k, s, d=jnp.float32: jnp.asarray(theta_from_tau(config.tau_init), d), (), sharding=sharding, dtype=param_dtype)
        lambda_init = math.log(math.expm1(config.lambda_screen_init))
        self.lambda_raw = _param(rngs, initializers.constant(lambda_init), (), sharding=sharding, dtype=param_dtype)
        self.slot_embed = _param(rngs, initializers.normal(0.02), (config.n_slots, config.d_slot), axes=slot, sharding=sharding, dtype=param_dtype)
        self.mu_by_bank_raw = _param(rngs, initializers.zeros_init(), (3,), sharding=sharding, dtype=param_dtype)
        if config.use_write_screening:
            self.q_proj_w = _linear(
                2 * C,
                config.d_k,
                use_bias=False,
                kernel_axes=row,
                rngs=rngs,
                sharding=sharding,
                dtype=compute_dtype,
                param_dtype=param_dtype,
            )
            self.k_proj_w = _linear(
                config.d_slot,
                config.d_k,
                use_bias=False,
                kernel_axes=row,
                rngs=rngs,
                sharding=sharding,
                dtype=compute_dtype,
                param_dtype=param_dtype,
            )
            self.tau_w_raw = _param(rngs, lambda k, s, d=jnp.float32: jnp.asarray(theta_from_tau(config.tau_init), d), (), sharding=sharding, dtype=param_dtype)
        else:
            self.q_proj_w = nnx.data(None)
            self.k_proj_w = nnx.data(None)
            self.tau_w_raw = nnx.data(None)

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
    ):
        del deterministic
        cfg = self.config
        phase = normalize_phase(phase)
        _, T, _ = x_seq.shape
        slots = _constrain_slots(state.slots.astype(jnp.float32), self.sharding)
        ages = state.ages
        usage_ema = state.usage_ema.astype(jnp.float32)
        x_ln_seq = _apply_norm_in_float32(
            self.screen_ln,
            x_seq,
            output_dtype=self.compute_dtype,
        )
        q_r_seq = unit_norm(
            self.q_proj_r(x_ln_seq).astype(jnp.float32), eps=cfg.eps
        )
        tau_r = bounded_tau(_value(self.tau_r_raw))
        lambda_screen = jax.nn.softplus(_value(self.lambda_raw))
        mu = self._compute_mu()
        write_enabled = phase == "read_write" and cfg.use_write_screening
        if write_enabled:
            q_w_in = jnp.concatenate(
                [x_ln_seq, h_base_seq.astype(jnp.float32)], axis=-1
            )
            q_w_seq = unit_norm(
                self.q_proj_w(q_w_in).astype(jnp.float32), eps=cfg.eps
            )
            tau_w = bounded_tau(_value(self.tau_w_raw))
        else:
            q_w_seq = None
            tau_w = jnp.zeros(())

        def step(carry, t):
            slots_t, ages_t, usage_t = carry
            k_r = unit_norm(
                self.k_proj_r(slots_t).astype(jnp.float32), eps=cfg.eps
            )
            values = self.v_proj(slots_t).astype(jnp.float32)
            if cfg.use_value_unit_norm:
                values = unit_norm(values, eps=cfg.eps)
            sim_r = jnp.einsum("bk,bmk->bm", q_r_seq[:, t, :], k_r)
            if cfg.use_leaky_warmup:
                hard = trim_square(sim_r, tau_r)
                soft = jax.nn.sigmoid(cfg.leaky_gamma * (sim_r - tau_r))
                rel_r = (1.0 - cfg.leaky_alpha) * hard + cfg.leaky_alpha * soft
            else:
                rel_r = trim_square(sim_r, tau_r, eps=cfg.eps)
            if cfg.use_age_mask:
                age_scores = (cfg.age_ref - ages_t) / (cfg.age_sigma + cfg.eps)
                rel_r *= jax.nn.sigmoid(age_scores)
            z = jnp.einsum("bm,bmv->bv", rel_r, values)
            u = tanh_norm(z, cap=cfg.tanh_norm_cap, eps=cfg.eps)
            gate = jax.nn.sigmoid(
                self.gate_proj(x_ln_seq[:, t, :]).astype(jnp.float32)
            )
            read_out = self.out_proj(u)
            h_t = (
                h_base_seq[:, t, :]
                + lambda_screen * gate * read_out.astype(h_base_seq.dtype)
            ).astype(h_base_seq.dtype)
            delta_s_t = _compute_slot_delta(
                x_ln_seq[:, t, :],
                h_base_seq[:, t, :].astype(jnp.float32),
                _value(self.slot_embed),
                _value(self.delta_proj.kernel),
                _value(self.delta_proj.bias),
                self.sharding,
                dtype=self.compute_dtype,
            ).astype(jnp.float32)
            if not write_enabled:
                strength = mu[None, :, None]
                new_slots = slots_t + strength * (delta_s_t - slots_t)
                new_ages = ages_t
                rel_w = jnp.zeros_like(rel_r)
            else:
                slots_for_key = slots_t + _value(self.slot_embed)[None, :, :]
                k_w = unit_norm(
                    self.k_proj_w(slots_for_key).astype(jnp.float32),
                    eps=cfg.eps,
                )
                sim_w = jnp.einsum("bk,bmk->bm", q_w_seq[:, t, :], k_w)
                rel_w = trim_square(sim_w, tau_w, eps=cfg.eps)
                strength = mu[None, :, None] * jnp.maximum(
                    rel_w, cfg.write_rel_floor
                )[:, :, None]
                new_slots = slots_t + strength * (delta_s_t - slots_t)
                new_ages = jnp.where(rel_w > 1e-3, 0.0, ages_t + 1.0)
            new_slots = _constrain_slots(new_slots, self.sharding)
            update_delta = new_slots - slots_t
            activity = jnp.maximum(rel_r, rel_w if write_enabled else rel_r)
            new_usage = cfg.usage_ema_decay * usage_t + (
                1.0 - cfg.usage_ema_decay
            ) * activity
            stats_t = {
                "rel_read_mean": jnp.mean(rel_r),
                "rel_read_max": jnp.max(rel_r),
                "active_slots_mean": jnp.mean(jnp.sum(rel_r > 1e-3, axis=-1)),
                "z_norm_mean": jnp.mean(jnp.linalg.norm(z, axis=-1)),
                "u_norm_mean": jnp.mean(jnp.linalg.norm(u, axis=-1)),
                "tau_r": tau_r,
                "lambda_screen": lambda_screen,
                "rel_write_mean": jnp.mean(rel_w),
                "rel_write_effective_mean": (
                    jnp.mean(jnp.maximum(rel_w, cfg.write_rel_floor))
                    if write_enabled
                    else jnp.mean(rel_w)
                ),
                "slot_update_norm_mean": jnp.mean(
                    jnp.linalg.norm(update_delta, axis=-1)
                ),
                "slot_usage_ema_mean": jnp.mean(new_usage),
                "tau_w": tau_w,
            }
            return (new_slots, new_ages, new_usage), (h_t, stats_t)

        (final_slots, final_ages, final_usage), (h_seq, stats_seq) = jax.lax.scan(
            step, (slots, ages, usage_ema), jnp.arange(T)
        )
        new_state = LayerScreenState(
            slots=final_slots.astype(state.slots.dtype),
            ages=final_ages,
            usage_ema=final_usage.astype(state.usage_ema.dtype),
        )
        return (
            jnp.swapaxes(h_seq, 0, 1),
            new_state,
            {key: jnp.mean(value) for key, value in stats_seq.items()},
        )


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
        self, x, v_first, rwkv_state, screen_state, *, phase, deterministic
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
):
    return layer(
        x,
        v_first,
        rwkv_state,
        screen_state,
        phase=phase,
        deterministic=deterministic,
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
    ):
        hidden, new_rwkv, new_screen, stats = self.compute_recurrent_hidden(
            input_ids,
            rwkv_state,
            screen_state,
            phase=phase,
            deterministic=deterministic,
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
