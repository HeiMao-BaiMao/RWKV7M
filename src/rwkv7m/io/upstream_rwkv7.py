"""Conversion helpers for numerical parity with official RWKV-LM-V7 x070 weights."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import flax
import numpy as np

from ..model.screened_rwkv import ModelConfig
from ..model.screening import ScreeningConfig


@dataclass(frozen=True)
class UpstreamRWKV7Spec:
    n_layers: int
    d_model: int
    d_ffn: int
    vocab_size: int
    head_size: int
    n_heads: int

    def model_config(self, *, dtype="bfloat16"):
        return ModelConfig(
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            head_size=self.head_size,
            vocab_size=self.vocab_size,
            max_seq_len=4096,
            dtype=dtype,
            use_screening=False,
            screening=ScreeningConfig(),
        )


@dataclass(frozen=True)
class UpstreamRWKV7ConversionReport:
    source_tensors: int
    converted_tensors: int
    consumed_names: tuple[str, ...]
    ignored_names: tuple[str, ...]
    unexpected_names: tuple[str, ...]

    @property
    def complete(self):
        return (
            not self.unexpected_names
            and self.source_tensors == self.converted_tensors + len(self.ignored_names)
        )

    def to_dict(self):
        payload = asdict(self)
        payload["complete"] = self.complete
        return payload


def _arrays(state_dict):
    result = {}
    for raw_name, value in state_dict.items():
        name = str(raw_name)
        if name.startswith("_forward_module."):
            name = name.removeprefix("_forward_module.")
        if name in result:
            raise ValueError(f"duplicate upstream tensor after prefix normalization: {name}")
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        result[name] = np.asarray(value)
    return result


def infer_upstream_rwkv7_spec(state_dict, *, head_size=64):
    arrays = _arrays(state_dict)
    try:
        embedding = arrays["emb.weight"]
        ffn_key = arrays["blocks.0.ffn.key.weight"]
    except KeyError as exc:
        raise ValueError(f"missing required upstream tensor: {exc.args[0]}") from exc
    if embedding.ndim != 2 or ffn_key.ndim != 2:
        raise ValueError("upstream embedding and FFN key tensors must be matrices")
    block_ids = set()
    for name in arrays:
        if name.startswith("blocks."):
            try:
                block_ids.add(int(name.split(".", 2)[1]))
            except (IndexError, ValueError):
                pass
    if not block_ids or block_ids != set(range(max(block_ids) + 1)):
        raise ValueError(f"upstream block indices must be contiguous from zero: {sorted(block_ids)}")
    vocab_size, d_model = embedding.shape
    if d_model % int(head_size) != 0:
        raise ValueError(f"d_model={d_model} must be divisible by head_size={head_size}")
    if ffn_key.shape[1] != d_model:
        raise ValueError("upstream FFN key input width does not match embedding width")
    return UpstreamRWKV7Spec(
        n_layers=len(block_ids),
        d_model=int(d_model),
        d_ffn=int(ffn_key.shape[0]),
        vocab_size=int(vocab_size),
        head_size=int(head_size),
        n_heads=int(d_model // head_size),
    )


def _mapping(spec):
    C, F, H, N = spec.d_model, spec.d_ffn, spec.n_heads, spec.head_size
    d_decay = max(32, int(round((2.5 * math.sqrt(C)) / 32) * 32))
    d_value = max(32, int(round((1.7 * math.sqrt(C)) / 32) * 32))
    d_gate = max(32, int(round((5.0 * math.sqrt(C)) / 32) * 32))
    entries = []

    def add(source, path, shape, transpose=False):
        entries.append((source, path, tuple(shape), transpose))

    add("emb.weight", "token_embedding.embedding", (spec.vocab_size, C))
    for layer in range(spec.n_layers):
        source = f"blocks.{layer}"
        target = f"layer_{layer}.rwkv_block_{layer}"
        if layer == 0:
            add(f"{source}.ln0.weight", f"{target}.ln0.scale", (C,))
            add(f"{source}.ln0.bias", f"{target}.ln0.bias", (C,))
        for norm in ("ln1", "ln2"):
            add(f"{source}.{norm}.weight", f"{target}.{norm}.scale", (C,))
            add(f"{source}.{norm}.bias", f"{target}.{norm}.bias", (C,))
        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            add(f"{source}.att.{name}", f"{target}.att.{name}", (C,))
        for name, shape in (
            ("w0", (1, 1, C)), ("w1", (C, d_decay)), ("w2", (d_decay, C)),
            ("a0", (1, 1, C)), ("a1", (C, d_decay)), ("a2", (d_decay, C)),
            ("g1", (C, d_gate)), ("g2", (d_gate, C)),
            ("k_k", (1, 1, C)), ("k_a", (1, 1, C)), ("r_k", (H, N)),
        ):
            add(f"{source}.att.{name}", f"{target}.att.{name}", shape)
        for name, shape in (
            ("v0", (1, 1, C)), ("v1", (C, d_value)), ("v2", (d_value, C)),
        ):
            if layer > 0:
                add(f"{source}.att.{name}", f"{target}.att.{name}", shape)
            else:
                # Official x070 allocates these on layer 0, but forward replaces
                # v_first directly and never reads them. The local core omits
                # the dead parameters; validate and report them explicitly.
                add(f"{source}.att.{name}", None, shape)
        for name in ("receptance", "key", "value", "output"):
            add(f"{source}.att.{name}.weight", f"{target}.att.{name}.kernel", (C, C), True)
        add(f"{source}.att.ln_x.weight", f"{target}.att.ln_x.scale", (C,))
        add(f"{source}.att.ln_x.bias", f"{target}.att.ln_x.bias", (C,))
        add(f"{source}.ffn.x_k", f"{target}.ffn.x_k", (C,))
        add(f"{source}.ffn.key.weight", f"{target}.ffn.key.kernel", (C, F), True)
        add(f"{source}.ffn.value.weight", f"{target}.ffn.value.kernel", (F, C), True)
    add("ln_out.weight", "final_ln.scale", (C,))
    add("ln_out.bias", "final_ln.bias", (C,))
    add("head.weight", "lm_head.kernel", (C, spec.vocab_size), True)
    return entries


def convert_upstream_rwkv7_state_dict(state_dict, *, head_size=64, strict=True):
    arrays = _arrays(state_dict)
    spec = infer_upstream_rwkv7_spec(arrays, head_size=head_size)
    flat = {}
    consumed = []
    ignored = []
    for source, target, expected_shape, transpose in _mapping(spec):
        if source not in arrays:
            raise ValueError(f"missing required upstream tensor: {source}")
        value = arrays[source]
        if transpose:
            if value.ndim != 2:
                raise ValueError(f"{source} must be a matrix before transpose, got {value.shape}")
            value = value.T
        if value.size != int(np.prod(expected_shape)):
            raise ValueError(
                f"shape mismatch for {source}: source={arrays[source].shape}, "
                f"converted={value.shape}, expected={expected_shape}"
            )
        value = value.reshape(expected_shape)
        if target is None:
            ignored.append(source)
        else:
            flat[target] = value
            consumed.append(source)
    accounted = set(consumed) | set(ignored)
    unexpected = tuple(sorted(set(arrays) - accounted))
    if strict and unexpected:
        raise ValueError(f"unexpected upstream tensors: {', '.join(unexpected)}")
    params = flax.traverse_util.unflatten_dict(flat, sep=".")
    report = UpstreamRWKV7ConversionReport(
        source_tensors=len(arrays),
        converted_tensors=len(consumed),
        consumed_names=tuple(sorted(consumed)),
        ignored_names=tuple(sorted(ignored)),
        unexpected_names=unexpected,
    )
    return params, spec, report


def load_upstream_rwkv7_reference_archive(path):
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {"metadata_json", "input_ids", "reference_logits"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"reference archive is missing: {', '.join(sorted(missing))}")
        metadata = json.loads(str(archive["metadata_json"].item()))
        weights = {
            key.removeprefix("weight::"): np.asarray(archive[key])
            for key in archive.files if key.startswith("weight::")
        }
        gradients = {
            key.removeprefix("gradient::"): np.asarray(archive[key])
            for key in archive.files if key.startswith("gradient::")
        }
        updated_weights = {
            key.removeprefix("updated_weight::"): np.asarray(archive[key])
            for key in archive.files if key.startswith("updated_weight::")
        }
        layers = {
            int(key.removeprefix("layer::")): np.asarray(archive[key])
            for key in archive.files if key.startswith("layer::")
        }
        targets = np.asarray(archive["target_ids"]) if "target_ids" in archive else None
        return {
            "metadata": metadata,
            "weights": weights,
            "gradients": gradients,
            "updated_weights": updated_weights,
            "input_ids": np.asarray(archive["input_ids"]),
            "target_ids": targets,
            "reference_logits": np.asarray(archive["reference_logits"]),
            "reference_loss": (
                float(archive["reference_loss"])
                if "reference_loss" in archive else None
            ),
            "reference_layers": layers,
        }
