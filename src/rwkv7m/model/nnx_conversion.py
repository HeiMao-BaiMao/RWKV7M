"""Explicit parameter boundary between the Linen reference and NNX runtime."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
from flax import nnx, traverse_util
from flax.core import freeze


def _linen_flat(params: Mapping) -> dict[str, object]:
    return traverse_util.flatten_dict(params, sep="/")


def _nnx_flat(model: nnx.Module) -> dict[str, nnx.Variable]:
    return {
        "/".join(str(part) for part in path): variable
        for path, variable in nnx.to_flat_state(nnx.state(model, nnx.Param))
    }


def _check_parameter_contract(linen_flat, nnx_flat):
    linen_keys = set(linen_flat)
    nnx_keys = set(nnx_flat)
    if linen_keys != nnx_keys:
        missing = sorted(linen_keys - nnx_keys)
        extra = sorted(nnx_keys - linen_keys)
        raise ValueError(
            "Linen/NNX parameter paths differ: "
            f"missing_in_nnx={missing}, extra_in_nnx={extra}"
        )
    mismatches = []
    for path in sorted(linen_keys):
        source = jnp.asarray(linen_flat[path])
        target = jnp.asarray(nnx_flat[path][...])
        if source.shape != target.shape:
            mismatches.append(f"{path}: Linen {source.shape}, NNX {target.shape}")
    if mismatches:
        raise ValueError("Linen/NNX parameter shapes differ: " + "; ".join(mismatches))


def load_linen_params_into_nnx(model: nnx.Module, params: Mapping) -> nnx.Module:
    """Load a complete Linen params tree into an isomorphic NNX model.

    This function mutates ``model`` in the normal NNX style. It refuses partial
    loads so parity tests and conversion tools cannot silently skip tensors.
    Existing NNX variable metadata, including sharding, is retained.
    """

    linen_flat = _linen_flat(params)
    nnx_flat = _nnx_flat(model)
    _check_parameter_contract(linen_flat, nnx_flat)
    for path, variable in nnx_flat.items():
        source = jnp.asarray(linen_flat[path], dtype=variable[...].dtype)
        variable[...] = source
    return model


def nnx_params_to_linen(model_or_state) -> Mapping:
    """Return NNX parameters as the established Linen-style portable tree."""

    if isinstance(model_or_state, nnx.Module):
        state = nnx.state(model_or_state, nnx.Param)
    elif isinstance(model_or_state, nnx.State):
        state = model_or_state
    else:
        raise TypeError("expected an NNX Module or nnx.State")
    flat = {
        "/".join(str(part) for part in path): variable[...]
        for path, variable in nnx.to_flat_state(state)
    }
    return freeze(traverse_util.unflatten_dict(flat, sep="/"))


def assert_nnx_linen_parameter_contract(model: nnx.Module, params: Mapping) -> None:
    """Validate path and shape parity without changing either object."""

    _check_parameter_contract(_linen_flat(params), _nnx_flat(model))


__all__ = [
    "assert_nnx_linen_parameter_contract",
    "load_linen_params_into_nnx",
    "nnx_params_to_linen",
]
