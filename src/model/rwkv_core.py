import jax
import jax.numpy as jnp
from flax import linen as nn


class RWKVCoreInterface:
    def __call__(self, x_t, rwkv_state, *, deterministic):
        raise NotImplementedError


class PlaceholderRWKVCore(nn.Module):
    d_model: int
    d_ffn: int = -1

    def setup(self):
        d_ffn = self.d_ffn if self.d_ffn > 0 else 4 * self.d_model
        self.ln = nn.LayerNorm(dtype=jnp.float32, name="core_ln")
        self.ffn1 = nn.Dense(d_ffn, name="core_ffn1")
        self.ffn2 = nn.Dense(self.d_model, name="core_ffn2")

    def __call__(self, x_t, rwkv_state, *, deterministic):
        h = x_t.astype(jnp.float32)
        h_norm = self.ln(h)
        h_ffn = self.ffn2(jax.nn.gelu(self.ffn1(h_norm)))
        h_base = x_t + h_ffn.astype(x_t.dtype)
        return h_base, rwkv_state
