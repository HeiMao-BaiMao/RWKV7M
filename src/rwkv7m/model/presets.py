"""Named RWKV7M model-size presets built on the shared NNX config contract."""

from __future__ import annotations

from .screened_rwkv import ModelConfig
from .screening import ScreeningConfig


MODEL_PRESET_NAMES = ("0.185b", "0.3b", "1b", "3b", "7b")
MODEL_PRESET_PARAMETER_COUNTS = {
    "0.185b": 184_985_222,
    "0.3b": 297_738_764,
    "1b": 985_479_192,
    "3b": 2_943_319_064,
    "7b": 6_994_788_376,
}
MODEL_PRESET_RECOMMENDED_MODEL_AXIS_SIZES = {
    "0.185b": 1,
    "0.3b": 1,
    "1b": 2,
    "3b": 4,
    "7b": 8,
}

_ALIASES = {
    "0.19b": "0.185b",
    "185m": "0.185b",
    "300m": "0.3b",
    "1.0b": "1b",
    "3.0b": "3b",
    "7.0b": "7b",
}
_BANK_IDS = (0,) * 8 + (1,) * 4 + (2,) * 4


def canonical_model_preset_name(name: str) -> str:
    normalized = str(name).strip().lower()
    normalized = _ALIASES.get(normalized, normalized)
    if normalized not in MODEL_PRESET_NAMES:
        available = ", ".join(MODEL_PRESET_NAMES)
        raise ValueError(f"unknown model preset {name!r}; choose one of: {available}")
    return normalized


def _preset_config(
    *,
    d_model: int,
    d_ffn: int,
    n_layers: int,
    max_seq_len: int,
    screened_layers: tuple[int, ...],
    d_slot: int,
    d_k: int,
    d_v: int,
    vocab_parallel: bool,
    sequence_chunk_size: int,
) -> ModelConfig:
    return ModelConfig(
        d_model=d_model,
        d_ffn=d_ffn,
        n_layers=n_layers,
        n_heads=d_model // 64,
        head_size=64,
        vocab_size=65_536,
        max_seq_len=max_seq_len,
        dtype="bfloat16",
        param_dtype="bfloat16",
        param_update_dtype="float32",
        optimizer_state_dtype="float32",
        gradient_accum_dtype="float32",
        lm_head_init="variance_scaled",
        vocab_parallel=vocab_parallel,
        remat_blocks=True,
        sequence_chunk_size=sequence_chunk_size,
        use_screening=True,
        screening=ScreeningConfig(
            d_model=d_model,
            d_slot=d_slot,
            d_k=d_k,
            d_v=d_v,
            n_slots=16,
            screened_layers=screened_layers,
            bank_ids=_BANK_IDS,
            use_write_screening=True,
        ),
    )


def model_preset(name: str) -> ModelConfig:
    """Return a fresh config for a named approximate parameter scale."""

    name = canonical_model_preset_name(name)
    if name == "0.185b":
        return _preset_config(
            d_model=768,
            d_ffn=2688,
            n_layers=12,
            max_seq_len=512,
            screened_layers=(6,),
            d_slot=128,
            d_k=64,
            d_v=64,
            vocab_parallel=False,
            sequence_chunk_size=128,
        )
    if name == "0.3b":
        return _preset_config(
            d_model=1024,
            d_ffn=3584,
            n_layers=13,
            max_seq_len=1024,
            screened_layers=(5, 12),
            d_slot=256,
            d_k=64,
            d_v=128,
            vocab_parallel=False,
            sequence_chunk_size=128,
        )
    if name == "1b":
        return _preset_config(
            d_model=1536,
            d_ffn=5376,
            n_layers=28,
            max_seq_len=2048,
            screened_layers=(6, 13, 20, 27),
            d_slot=384,
            d_k=96,
            d_v=192,
            vocab_parallel=True,
            sequence_chunk_size=256,
        )
    if name == "3b":
        return _preset_config(
            d_model=2560,
            d_ffn=8960,
            n_layers=34,
            max_seq_len=4096,
            screened_layers=(7, 15, 23, 33),
            d_slot=640,
            d_k=128,
            d_v=320,
            vocab_parallel=True,
            sequence_chunk_size=256,
        )
    return _preset_config(
        d_model=4096,
        d_ffn=15232,
        n_layers=32,
        max_seq_len=4096,
        screened_layers=(7, 15, 23, 31),
        d_slot=1024,
        d_k=128,
        d_v=512,
        vocab_parallel=True,
        sequence_chunk_size=256,
    )


__all__ = [
    "MODEL_PRESET_NAMES",
    "MODEL_PRESET_PARAMETER_COUNTS",
    "MODEL_PRESET_RECOMMENDED_MODEL_AXIS_SIZES",
    "canonical_model_preset_name",
    "model_preset",
]
