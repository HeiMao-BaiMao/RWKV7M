import jax
import jax.numpy as jnp
from flax import linen as nn
from dataclasses import dataclass, field

from .screening import ScreeningConfig, StateLevelScreening, normalize_phase
from .state import (
    ModelScreenState,
    LayerScreenState,
    init_screen_state,
    init_rwkv_state as init_model_rwkv_state,
)
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

    def __post_init__(self):
        if self.d_model != self.n_heads * self.head_size:
            raise ValueError("d_model must equal n_heads * head_size")
        self.screening.screened_layers = tuple(self.screening.screened_layers)
        self.screening.bank_ids = tuple(self.screening.bank_ids)
        if not self.use_screening or not self.screening.screened_layers:
            return
        if self.screening.d_model != self.d_model:
            raise ValueError("screening.d_model must equal model.d_model")
        if len(self.screening.bank_ids) != self.screening.n_slots:
            raise ValueError("screening.bank_ids must have length screening.n_slots")
        invalid_banks = [
            bank_id for bank_id in self.screening.bank_ids if bank_id not in (0, 1, 2)
        ]
        if invalid_banks:
            raise ValueError("screening.bank_ids values must be only 0, 1, or 2")
        invalid_layers = [
            layer_id
            for layer_id in self.screening.screened_layers
            if layer_id < 0 or layer_id >= self.n_layers
        ]
        if invalid_layers:
            raise ValueError("screening.screened_layers must be within [0, n_layers)")


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

    def __call__(self, x, v_first, rwkv_state, screen_state, *, phase, deterministic):
        # RWKV7 core processes full sequence via internal lax.scan
        h_base, v_first, new_rwkv_state = self.rwkv_block(x, v_first, rwkv_state)

        if self._has_screening:
            h, new_screen, stats = self.screening(
                x,
                h_base,
                screen_state,
                phase=phase,
                deterministic=deterministic,
            )
        else:
            h = h_base
            new_screen = screen_state
            stats = {}

        return h, v_first, new_rwkv_state, new_screen, stats


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
        phase="read_screening_only",
        deterministic=True,
    ):
        cfg = self.config
        batch_size = input_ids.shape[0]
        phase = normalize_phase(phase)

        x = self.token_embedding(input_ids)  # [B, T, d]
        x = x.astype(_get_model_dtype(cfg))

        screened_idx = {layer_id: i for i, layer_id in enumerate(cfg.screening.screened_layers)}

        # v_first is the value projection from layer 0, threaded through all layers
        v_first = jnp.zeros_like(x)

        all_stats = []
        new_rwkv_layers = list(rwkv_state)
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

            x, v_first, new_rwkv_s, new_scr_s, stats = self.layers[l_idx](
                x,
                v_first,
                rwkv_state[l_idx],
                scr_s,
                phase=phase,
                deterministic=deterministic,
            )

            new_rwkv_layers[l_idx] = new_rwkv_s
            if l_idx in screened_idx:
                new_screen_layers[screened_idx[l_idx]] = new_scr_s
            all_stats.append(stats)

        x = self.final_ln(x.astype(jnp.float32))
        logits = self.lm_head(x)

        new_screen_state = ModelScreenState(layers=tuple(new_screen_layers))
        new_rwkv_state = tuple(new_rwkv_layers)

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

        return logits, new_rwkv_state, new_screen_state, agg_stats


def init_rwkv_state(batch_size, config: ModelConfig):
    return init_model_rwkv_state(batch_size, config)


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
        phase="read_screening_only",
        deterministic=True,
    )
    return variables, model
