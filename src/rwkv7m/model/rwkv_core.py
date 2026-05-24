"""RWKV-7 core implementation in JAX/Flax.

Reference implementation for RWKV-7 style recurrent blocks.
"""

import math
import jax
import jax.numpy as jnp
from flax import linen as nn
from dataclasses import dataclass

from .state import LayerRWKVState


@dataclass
class RWKV7Config:
    d_model: int = 768
    d_ffn: int = -1
    n_layers: int = 12
    n_heads: int = 12
    head_size: int = 64
    vocab_size: int = 50257
    max_seq_len: int = 4096
    dtype: str = "bfloat16"


def _get_dtype(cfg: RWKV7Config):
    if cfg.dtype == "float32":
        return jnp.float32
    elif cfg.dtype == "bfloat16":
        return jnp.bfloat16
    return jnp.float32


def _time_shift(x, prev_x=None):
    """x: [B, T, C] -> shift by 1 in time, using prev_x for chunked decode."""
    if prev_x is None:
        prev_x = jnp.zeros((x.shape[0], x.shape[-1]), dtype=x.dtype)
    first = prev_x.astype(x.dtype)[:, None, :]
    return jnp.concatenate([first, x[:, :-1, :]], axis=1)


def wkv_step(state, r_t, w_t, k_t, v_t, neg_kk_t, kka_t):
    """Single-step WKV recurrence (pure function).

    Args:
        state: [B, H, N, N]
        r_t, w_t, k_t, v_t, neg_kk_t, kka_t: [B, H, N]
    Returns:
        new_state: [B, H, N, N]
        y_t: [B, H, N]
    """
    w_decay = jnp.exp(-jnp.exp(w_t))  # [B, H, N]
    w_decay = w_decay[..., None, :]   # [B, H, 1, N]

    sa = jnp.einsum("bhij,bhj->bhi", state, neg_kk_t)  # [B, H, N]

    state = (
        state * w_decay
        + jnp.einsum("bhi,bhj->bhij", sa, kka_t)
        + jnp.einsum("bhi,bhj->bhij", v_t, k_t)
    )

    y = jnp.einsum("bhij,bhj->bhi", state, r_t)  # [B, H, N]
    return state, y


class RWKV7TimeMix(nn.Module):
    config: RWKV7Config
    layer_idx: int

    @nn.compact
    def __call__(self, x, v_first, state=None):
        B, T, _ = x.shape
        C = self.config.d_model
        H = self.config.n_heads
        N = self.config.head_size
        assert x.shape[-1] == H * N, f"d_model={x.shape[-1]} must equal n_heads*head_size={H*N}"
        if state is None:
            prev_x = jnp.zeros((B, C), dtype=x.dtype)
            initial_state = jnp.zeros((B, H, N, N), dtype=jnp.float32)
        else:
            prev_x = state.time_mix_x
            initial_state = state.wkv.astype(jnp.float32)

        # --- Layer-dependent init helpers ---
        ratio_0_to_1 = self.layer_idx / max(1, self.config.n_layers - 1)
        ratio_1_to_almost0 = 1.0 - (self.layer_idx / max(1, self.config.n_layers))

        def mix_init(power):
            def init_fn(key, shape, dtype=jnp.float32):
                ddd = jnp.arange(shape[-1], dtype=jnp.float32) / shape[-1]
                return 1.0 - jnp.power(ddd, power * ratio_1_to_almost0)
            return init_fn

        # --- Time-shift ---
        xx = _time_shift(x, prev_x) - x  # [B, T, C]

        x_r = x + xx * self.param("x_r", mix_init(0.2), (1, 1, C))
        x_w = x + xx * self.param("x_w", mix_init(0.9), (1, 1, C))
        x_k = x + xx * self.param("x_k", mix_init(0.7), (1, 1, C))
        x_v = x + xx * self.param("x_v", mix_init(0.7), (1, 1, C))
        x_a = x + xx * self.param("x_a", mix_init(0.9), (1, 1, C))
        x_g = x + xx * self.param("x_g", mix_init(0.2), (1, 1, C))

        # --- Projections ---
        r = nn.Dense(C, use_bias=False, name="receptance",
                     kernel_init=nn.initializers.uniform(scale=1.0 / math.sqrt(C)))(x_r)

        # Decay (LoRA-style)
        d_decay = max(32, int(round((2.5 * math.sqrt(C)) / 32) * 32))
        w1 = self.param("w1", nn.initializers.zeros, (C, d_decay))
        w2 = self.param("w2", nn.initializers.orthogonal(scale=0.1), (d_decay, C))

        def w0_init(key, shape, dtype=jnp.float32):
            linear = jnp.arange(C) / max(1, C - 1) - 0.5
            zigzag = ((jnp.arange(C) % N) - ((N - 1) / 2)) / ((N - 1) / 2)
            zigzag = zigzag * jnp.abs(zigzag)
            www = -6.0 + 6.0 * jnp.power(jnp.arange(C) / max(1, C - 1), 1 + 1 * ratio_0_to_1 ** 0.3)
            return (www + 0.5 + zigzag * 2.5).reshape(1, 1, C)

        w0 = self.param("w0", w0_init, (1, 1, C))
        w_raw = w0 + jnp.tanh(x_w @ w1) @ w2  # [B, T, C]
        w_clamped = -jax.nn.softplus(-w_raw) - 0.5  # soft-clamp to (-inf, -0.5)

        k = nn.Dense(C, use_bias=False, name="key",
                     kernel_init=nn.initializers.uniform(scale=0.1 / math.sqrt(C)))(x_k)
        v = nn.Dense(C, use_bias=False, name="value",
                     kernel_init=nn.initializers.uniform(scale=1.0 / math.sqrt(C)))(x_v)

        # --- Value residual (cross-layer, layer > 0) ---
        if self.layer_idx == 0:
            v_first = v
        else:
            d_mv = max(32, int(round((1.7 * math.sqrt(C)) / 32) * 32))
            v1 = self.param("v1", nn.initializers.zeros, (C, d_mv))
            v2 = self.param("v2", nn.initializers.orthogonal(scale=0.1), (d_mv, C))

            def v0_init(key, shape, dtype=jnp.float32):
                linear = jnp.arange(C) / max(1, C - 1) - 0.5
                return (0.73 - linear * 0.4).reshape(1, 1, C)

            v0 = self.param("v0", v0_init, (1, 1, C))
            v12 = jnp.tanh(x_v @ v1) @ v2
            v = v + (v_first - v) * jax.nn.sigmoid(v0 + v12)

        # --- In-context learning rate gate ---
        d_aaa = max(32, int(round((2.5 * math.sqrt(C)) / 32) * 32))
        a1 = self.param("a1", nn.initializers.zeros, (C, d_aaa))
        a2 = self.param("a2", nn.initializers.orthogonal(scale=0.1), (d_aaa, C))

        def a0_init(key, shape, dtype=jnp.float32):
            linear = jnp.arange(C) / max(1, C - 1) - 0.5
            zigzag = ((jnp.arange(C) % N) - ((N - 1) / 2)) / ((N - 1) / 2)
            zigzag = zigzag * jnp.abs(zigzag)
            return (-0.19 + zigzag * 0.3 + linear * 0.4).reshape(1, 1, C)

        a0 = self.param("a0", a0_init, (1, 1, C))
        a12 = jnp.tanh(x_a @ a1) @ a2
        a = jax.nn.sigmoid(a0 + a12)

        # --- Output gate ---
        d_gate = max(32, int(round((5.0 * math.sqrt(C)) / 32) * 32))
        g1 = self.param("g1", nn.initializers.zeros, (C, d_gate))
        g2 = self.param("g2", nn.initializers.orthogonal(scale=0.1), (d_gate, C))
        g = jax.nn.sigmoid(x_g @ g1) @ g2

        # --- Key preprocessing ---
        def k_k_init(key, shape, dtype=jnp.float32):
            linear = jnp.arange(C) / max(1, C - 1) - 0.5
            return (0.71 - linear * 0.1).reshape(1, 1, C)

        k_k = self.param("k_k", k_k_init, (1, 1, C))
        k_a = self.param("k_a", nn.initializers.constant(1.02), (1, 1, C))

        kk = k * k_k  # [B, T, C]
        kk_h = kk.reshape(B, T, H, N)
        kk_norm = jnp.sqrt(jnp.sum(kk_h * kk_h, axis=-1, keepdims=True) + 1e-12 ** 2)
        kk_h = kk_h / kk_norm
        kk = kk_h.reshape(B, T, C)

        k = k * (1.0 + (a - 1.0) * k_a)
        neg_kk = -kk
        kka = kk * a

        # --- Head reshape ---
        r_h = r.reshape(B, T, H, N)
        w_h = w_clamped.reshape(B, T, H, N)
        k_h = k.reshape(B, T, H, N)
        v_h = v.reshape(B, T, H, N)
        neg_kk_h = neg_kk.reshape(B, T, H, N)
        kka_h = kka.reshape(B, T, H, N)

        # --- WKV recurrence via lax.scan ---
        inputs = (
            jnp.swapaxes(r_h, 0, 1),
            jnp.swapaxes(w_h, 0, 1),
            jnp.swapaxes(k_h, 0, 1),
            jnp.swapaxes(v_h, 0, 1),
            jnp.swapaxes(neg_kk_h, 0, 1),
            jnp.swapaxes(kka_h, 0, 1),
        )

        final_state, y_h = jax.lax.scan(
            lambda state, ins: wkv_step(state, *ins),
            initial_state,
            inputs,
        )

        y = jnp.swapaxes(y_h, 0, 1).reshape(B, T, C)

        # --- Post-processing ---
        # GroupNorm (note: eps=64e-5 = 0.00064)
        y = nn.GroupNorm(num_groups=H, epsilon=64e-5, name="ln_x")(
            y.reshape(B * T, C)
        ).reshape(B, T, C)

        # RKVR residual
        r_k = self.param("r_k", nn.initializers.constant(-0.04), (H, N))
        rk = r_h * k_h * r_k  # [B, T, H, N]
        rk_sum = jnp.sum(rk, axis=-1, keepdims=True)  # [B, T, H, 1]
        rkv = rk_sum * v_h  # [B, T, H, N]
        rkv = rkv.reshape(B, T, C)
        y = y + rkv

        # Gate + output
        y = y * g
        y = nn.Dense(C, use_bias=False, name="output",
                     kernel_init=nn.initializers.zeros)(y)

        return y, v_first, x[:, -1, :].astype(jnp.float32), final_state


class RWKV7ChannelMix(nn.Module):
    config: RWKV7Config
    layer_idx: int

    @nn.compact
    def __call__(self, x, prev_x=None):
        B, T, _ = x.shape
        C = self.config.d_model
        if prev_x is None:
            prev_x = jnp.zeros((B, C), dtype=x.dtype)

        ratio_1_to_almost0 = 1.0 - (self.layer_idx / max(1, self.config.n_layers))

        def mix_init(key, shape, dtype=jnp.float32):
            ddd = jnp.arange(C, dtype=jnp.float32) / C
            return 1.0 - jnp.power(ddd, ratio_1_to_almost0 ** 4)

        # Time-shift
        xx = _time_shift(x, prev_x) - x
        x_k = x + xx * self.param("x_k", mix_init, (1, 1, C))

        # FFN
        k = nn.Dense(C * 4, use_bias=False, name="key",
                     kernel_init=nn.initializers.uniform(scale=1.0 / math.sqrt(C)))(x_k)
        k = jax.nn.relu(k) ** 2  # Squared ReLU
        v = nn.Dense(C, use_bias=False, name="value",
                     kernel_init=nn.initializers.zeros)(k)

        return v, x[:, -1, :].astype(jnp.float32)


class RWKV7Block(nn.Module):
    config: RWKV7Config
    layer_idx: int

    @nn.compact
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
            x = nn.LayerNorm(epsilon=1e-5, name="ln0")(x)

        # Time mixing
        x_ln = nn.LayerNorm(epsilon=1e-5, name="ln1")(x)
        x_attn, v_first, time_mix_x, wkv = RWKV7TimeMix(
            config=cfg,
            layer_idx=self.layer_idx,
            name="att",
        )(x_ln, v_first, rwkv_state)
        x = x + x_attn

        # Channel mixing
        x_ln = nn.LayerNorm(epsilon=1e-5, name="ln2")(x)
        x_ffn, channel_mix_x = RWKV7ChannelMix(
            config=cfg,
            layer_idx=self.layer_idx,
            name="ffn",
        )(x_ln, rwkv_state.channel_mix_x)
        x = x + x_ffn

        new_state = LayerRWKVState(
            time_mix_x=time_mix_x,
            channel_mix_x=channel_mix_x,
            wkv=wkv,
        )
        return x, v_first, new_state
