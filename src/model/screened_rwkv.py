import jax
import jax.numpy as jnp
from flax import linen as nn
from dataclasses import dataclass, field

from .screening import ScreeningConfig, StateLevelScreening
from .state import ModelScreenState, LayerScreenState, init_screen_state
from .rwkv_core import RWKV7Block


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
        self.rwkv_block = RWKV7Block(
            config=cfg,
            layer_idx=self.layer_idx,
            name=f"rwkv_block_{self.layer_idx}",
        )
        if cfg.use_screening and self.layer_idx in cfg.screening.screened_layers:
            self.screening = StateLevelScreening(
                config=cfg.screening,
                name=f"screening_{self.layer_idx}",
            )
            self._has_screening = True
        else:
            self._has_screening = False

    def __call__(self, x, v_first, screen_state, *, phase, deterministic):
        # RWKV7 core processes full sequence via internal lax.scan
        h_base, v_first = self.rwkv_block(x, v_first)

        if self._has_screening:
            # Apply screening per-token (lightweight compared to RWKV core)
            B, T, C = h_base.shape
            h_list = []
            stats_list = []
            new_screen = screen_state

            for t in range(T):
                x_t = x[:, t, :]
                h_base_t = h_base[:, t, :]
                h_t, new_screen, stats_t = self.screening(
                    x_t,
                    h_base_t,
                    new_screen,
                    phase=phase,
                    deterministic=deterministic,
                )
                h_list.append(h_t)
                stats_list.append(stats_t)

            h = jnp.stack(h_list, axis=1)
            # Aggregate stats over time
            if stats_list:
                stats = {
                    k: jnp.mean(jnp.array([s[k] for s in stats_list]))
                    for k in stats_list[0].keys()
                }
            else:
                stats = {}
        else:
            h = h_base
            new_screen = screen_state
            stats = {}

        return h, v_first, new_screen, stats


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
        self.final_ln = nn.LayerNorm(epsilon=1e-5, name="final_ln")
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

        x = self.token_embedding(input_ids)  # [B, T, d]
        x = x.astype(_get_model_dtype(cfg))

        screened_idx = {layer_id: i for i, layer_id in enumerate(cfg.screening.screened_layers)}

        # v_first is the value projection from layer 0, threaded through all layers
        v_first = jnp.empty_like(x)

        all_stats = []
        new_screen_layers = list(screen_state.layers)

        for l_idx in range(cfg.n_layers):
            if l_idx in screened_idx:
                scr_s = screen_state.layers[screened_idx[l_idx]]
            else:
                # Dummy state for non-screened layers
                scr_s = LayerScreenState(
                    slots=jnp.zeros((batch_size, 1, 1), dtype=jnp.float32),
                    ages=jnp.zeros((batch_size, 1), dtype=jnp.float32),
                    usage_ema=jnp.zeros((batch_size, 1), dtype=jnp.float32),
                )

            x, v_first, new_scr_s, stats = self.layers[l_idx](
                x,
                v_first,
                scr_s,
                phase=phase,
                deterministic=deterministic,
            )

            if l_idx in screened_idx:
                new_screen_layers[screened_idx[l_idx]] = new_scr_s
            all_stats.append(stats)

        x = self.final_ln(x.astype(jnp.float32))
        logits = self.lm_head(x)

        new_screen_state = ModelScreenState(layers=tuple(new_screen_layers))

        # Aggregate stats across layers
        agg_stats = {}
        keys = set()
        for s in all_stats:
            if s:
                keys.update(s.keys())
        for key in keys:
            vals = [s[key] for s in all_stats if s and key in s]
            if vals:
                agg_stats[key] = jnp.mean(jnp.array(vals))

        # rwkv_state is kept as dummy for backward compatibility
        return logits, rwkv_state, new_screen_state, agg_stats


def init_rwkv_state(batch_size, config: ModelConfig):
    """Dummy state for backward compatibility. RWKV-7 core manages recurrence internally."""
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

    # Use a short sequence for init to save memory
    dummy_ids = jnp.zeros((batch_size, 4), dtype=jnp.int32)

    variables = model.init(
        rng,
        dummy_ids,
        rwkv_state,
        screen_state,
        phase="read_only",
        deterministic=True,
    )
    return variables, model
