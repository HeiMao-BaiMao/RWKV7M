from dataclasses import asdict

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
