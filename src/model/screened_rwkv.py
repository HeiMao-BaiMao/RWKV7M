import jax
import jax.numpy as jnp
from flax import linen as nn
from dataclasses import dataclass, field
from typing import Any

from .screening import ScreeningConfig, StateLevelScreening
from .state import ModelScreenState, LayerScreenState, init_screen_state, tuple_set
from .rwkv_core import PlaceholderRWKVCore


@dataclass
class ModelConfig:
    d_model: int = 512
    d_ffn: int = -1
    n_layers: int = 8
    n_heads: int = 8
    head_size: int = 64
    vocab_size: int = 50257
    max_seq_len: int = 2048
    dtype: str = "bfloat16"

    # Screening config
    use_screening: bool = True
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)


def _get_model_dtype(cfg: ModelConfig):
    if cfg.dtype == "float32":
        return jnp.float32
    elif cfg.dtype == "bfloat16":
        return jnp.bfloat16
    else:
        return jnp.float32


class ScreenedRWKVLayer(nn.Module):
    config: ModelConfig
    layer_idx: int

    def setup(self):
        cfg = self.config
        self.rwkv_core = PlaceholderRWKVCore(
            d_model=cfg.d_model,
            d_ffn=cfg.d_ffn,
            name=f"rwkv_core_{self.layer_idx}",
        )
        if cfg.use_screening and self.layer_idx in cfg.screening.screened_layers:
            self.screening = StateLevelScreening(
                config=cfg.screening,
                name=f"screening_{self.layer_idx}",
            )
            self._has_screening = True
        else:
            self._has_screening = False

    def __call__(self, x_t, rwkv_state_l, screen_state_l, *, phase, deterministic):
        h_base, new_rwkv_state_l = self.rwkv_core(
            x_t, rwkv_state_l, deterministic=deterministic
        )
        if self._has_screening:
            h, new_screen_state_l, stats = self.screening(
                x_t,
                h_base,
                screen_state_l,
                phase=phase,
                deterministic=deterministic,
            )
        else:
            h = h_base
            new_screen_state_l = screen_state_l
            stats = {}
        return h, new_rwkv_state_l, new_screen_state_l, stats


class ScreenedRWKVModel(nn.Module):
    config: ModelConfig

    def setup(self):
        cfg = self.config
        self.token_embedding = nn.Embed(
            cfg.vocab_size,
            cfg.d_model,
            name="token_embedding",
        )
        self.layers = [
            ScreenedRWKVLayer(
                config=cfg,
                layer_idx=i,
                name=f"layer_{i}",
            )
            for i in range(cfg.n_layers)
        ]
        self.final_ln = nn.LayerNorm(dtype=jnp.float32, name="final_ln")
        self.lm_head = nn.Dense(
            cfg.vocab_size,
            use_bias=False,
            name="lm_head",
        )

    def __call__(
        self,
        input_ids,
        rwkv_state,
        screen_state,
        *,
        phase="read_only",
        deterministic=True,
    ):
        cfg = self.config
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        x = self.token_embedding(input_ids)  # [B, T, d]
        x = x.astype(_get_model_dtype(cfg))

        screened_idx = {layer_id: i for i, layer_id in enumerate(cfg.screening.screened_layers)}

        logits_list = []
        all_stats = []
        rwkv_state_cur = rwkv_state
        screen_state_cur = screen_state

        for t in range(seq_len):
            token_x = x[:, t, :]
            layer_stats = []
            h = token_x
            new_rwkv_layers = list(rwkv_state_cur)
            new_screen_layers = list(screen_state_cur.layers)

            for l_idx in range(cfg.n_layers):
                rwkv_s = rwkv_state_cur[l_idx]
                if l_idx in screened_idx:
                    scr_s = screen_state_cur.layers[screened_idx[l_idx]]
                else:
                    scr_s = LayerScreenState(
                        slots=jnp.zeros((batch_size, 1, 1), dtype=jnp.float32),
                        ages=jnp.zeros((batch_size, 1), dtype=jnp.float32),
                        usage_ema=jnp.zeros((batch_size, 1), dtype=jnp.float32),
                    )

                h, new_rwkv_s, new_scr_s, stats = self.layers[l_idx](
                    h, rwkv_s, scr_s,
                    phase=phase, deterministic=deterministic,
                )
                new_rwkv_layers[l_idx] = new_rwkv_s
                if l_idx in screened_idx:
                    new_screen_layers[screened_idx[l_idx]] = new_scr_s
                layer_stats.append(stats)

            h_final = self.final_ln(h.astype(jnp.float32))
            logits_t = self.lm_head(h_final)
            logits_list.append(logits_t)
            rwkv_state_cur = tuple(new_rwkv_layers)
            screen_state_cur = ModelScreenState(layers=tuple(new_screen_layers))
            all_stats.append(layer_stats)

        logits = jnp.stack(logits_list, axis=1)  # [B, T, V]

        # Aggregate stats: average over layers and time
        agg_stats = {}
        if len(all_stats) > 0:
            # Find first non-empty dict across all layers and time steps to get keys
            keys = set()
            for t_step in all_stats:
                for l_stats in t_step:
                    if l_stats:
                        keys.update(l_stats.keys())
            for key in keys:
                vals = []
                for t_step in all_stats:
                    for l_stats in t_step:
                        if l_stats and key in l_stats:
                            vals.append(l_stats[key])
                if vals:
                    agg_stats[key] = jnp.mean(jnp.array(vals))

        return logits, rwkv_state_cur, screen_state_cur, agg_stats


def init_rwkv_state(batch_size, config: ModelConfig):
    """Initialize RWKV state as a tuple of placeholders (None per layer)."""
    return tuple(
        jnp.zeros((batch_size, 1), dtype=jnp.float32)
        for _ in range(config.n_layers)
    )


def cross_entropy_loss(logits, targets, mask=None):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    if mask is not None:
        nll = nll * mask
        return jnp.sum(nll) / jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.mean(nll)


def create_model_variables(rng, config: ModelConfig, batch_size: int):
    """Create initialized model variables."""
    model = ScreenedRWKVModel(config=config)
    rwkv_state = init_rwkv_state(batch_size, config)
    screen_state = init_screen_state(batch_size, config.screening)

    dummy_ids = jnp.zeros((batch_size, config.max_seq_len), dtype=jnp.int32)

    variables = model.init(
        rng,
        dummy_ids,
        rwkv_state,
        screen_state,
        phase="read_only",
        deterministic=True,
    )
    return variables, model
