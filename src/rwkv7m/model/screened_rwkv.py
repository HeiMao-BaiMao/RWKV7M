import math

import jax
import jax.numpy as jnp
from flax import linen as nn
from dataclasses import dataclass, field

from .losses import cross_entropy_components, cross_entropy_loss
from .screening import ScreeningConfig, StateLevelScreening, normalize_phase
from .state import (
    ModelScreenState,
    LayerScreenState,
    init_screen_state,
    init_rwkv_state as init_model_rwkv_state,
)
from .rwkv_core import RWKV7Block, symmetric_uniform_init


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
    # Storage/update policy shared by small and large NNX models. Parameters
    # default to FP32 for reference parity; large configurations can select
    # BF16 storage while keeping gradients and Adam moments in FP32.
    param_dtype: str = "float32"
    param_update_dtype: str = "float32"
    optimizer_state_dtype: str = "float32"
    gradient_accum_dtype: str = "float32"
    # The global orthogonal LM-head initializer is retained for reference
    # parity on small models. Large sharded models should use
    # ``variance_scaled`` to avoid a global QR decomposition during init.
    lm_head_init: str = "orthogonal"
    # Large-model execution controls. These do not change the architecture;
    # they select memory-aware implementations of the same NNX graph.
    vocab_parallel: bool = False
    remat_blocks: bool = False
    sequence_chunk_size: int | None = None
    # None inherits sequence_chunk_size in the training path. This preserves
    # existing preset memory behavior while allowing independent head tuning.
    head_chunk_size: int | None = None

    # Screening config
    use_screening: bool = True
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)

    # Training defaults used by the reference train-state builder.
    lr_init: float = 1e-3
    lr_final: float = 1e-5
    warmup_steps: int = 10
    lr_schedule: str = "optax_cosine"
    max_grad_norm: float = 1.0
    weight_decay: float = 0.001
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    def __post_init__(self):
        if self.d_model != self.n_heads * self.head_size:
            raise ValueError("d_model must equal n_heads * head_size")
        supported_dtypes = {"float32", "bfloat16"}
        for name in (
            "dtype",
            "param_dtype",
            "param_update_dtype",
            "optimizer_state_dtype",
            "gradient_accum_dtype",
        ):
            value = getattr(self, name)
            if value not in supported_dtypes:
                raise ValueError(
                    f"{name} must be one of {sorted(supported_dtypes)}, got {value!r}"
                )
        if self.sequence_chunk_size is not None and self.sequence_chunk_size <= 0:
            raise ValueError("sequence_chunk_size must be positive when set")
        if self.head_chunk_size is not None and self.head_chunk_size <= 0:
            raise ValueError("head_chunk_size must be positive when set")
        if self.lm_head_init not in ("orthogonal", "variance_scaled"):
            raise ValueError(
                "lm_head_init must be 'orthogonal' or 'variance_scaled'"
            )
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
        # Upstream RWKV-LM tiny embedding init (relies on ln0 to renormalize).
        self.token_embedding = nn.Embed(
            cfg.vocab_size,
            cfg.d_model,
            embedding_init=symmetric_uniform_init(1e-4),
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
        if cfg.vocab_size > cfg.d_model:
            head_gain = 0.5 * math.sqrt(cfg.vocab_size / cfg.d_model)
        else:
            head_gain = 0.5
        if cfg.lm_head_init == "orthogonal":
            head_init = nn.initializers.orthogonal(scale=head_gain)
        else:
            head_init = nn.initializers.variance_scaling(
                scale=head_gain * head_gain,
                mode="fan_in",
                distribution="truncated_normal",
            )
        self.lm_head = nn.Dense(
            cfg.vocab_size,
            use_bias=False,
            kernel_init=head_init,
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
