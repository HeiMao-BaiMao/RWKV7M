from collections.abc import Mapping
from dataclasses import asdict
import json
from pathlib import Path

from ..model import ModelConfig, ScreeningConfig


def model_config_to_dict(config: ModelConfig):
    data = asdict(config)
    data["screening"]["screened_layers"] = list(data["screening"]["screened_layers"])
    data["screening"]["bank_ids"] = list(data["screening"]["bank_ids"])
    return data


def model_config_from_dict(data) -> ModelConfig:
    data = dict(data)
    screening_data = dict(data.get("screening", {}))
    screening_data["screened_layers"] = tuple(screening_data.get("screened_layers", ()))
    screening_data["bank_ids"] = tuple(screening_data.get("bank_ids", ()))
    data["screening"] = ScreeningConfig(**screening_data)
    return ModelConfig(**data)


def load_model_config(source) -> ModelConfig:
    """Resolve the shared small/large model configuration contract.

    Public APIs accept an existing ``ModelConfig``, a JSON-compatible mapping,
    or a path to the same JSON artifact used by the scale planner and trainer.
    """

    if isinstance(source, ModelConfig):
        return source
    if isinstance(source, Mapping):
        return model_config_from_dict(source)
    if isinstance(source, (str, Path)):
        path = Path(source)
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError("model config file must contain a JSON object")
        return model_config_from_dict(data)
    raise TypeError(
        "model config must be ModelConfig, a mapping, or a JSON file path; "
        f"got {type(source)!r}"
    )
