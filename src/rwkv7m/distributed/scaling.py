from dataclasses import asdict, dataclass
import math

import jax
from flax import nnx

from ..model.screened_rwkv import ModelConfig
from ..model.nnx_model import NNXScreenedRWKVModel


_DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "float64": 8,
}


@dataclass(frozen=True)
class DTypePolicy:
    param_storage_dtype: str
    param_update_dtype: str = "float32"
    compute_dtype: str = "bfloat16"
    optimizer_state_dtype: str = "float32"
    gradient_accum_dtype: str = "float32"
    recurrent_state_dtype: str = "float32"
    master_param_dtype: str | None = None

    @classmethod
    def stability(cls):
        return cls(param_storage_dtype="float32")

    @classmethod
    def memory(cls):
        return cls(param_storage_dtype="bfloat16")

    @classmethod
    def from_config(cls, config: ModelConfig):
        return cls(
            param_storage_dtype=config.param_dtype,
            param_update_dtype=config.param_update_dtype,
            compute_dtype=config.dtype,
            optimizer_state_dtype=config.optimizer_state_dtype,
            gradient_accum_dtype=config.gradient_accum_dtype,
        )

    def __post_init__(self):
        for value in asdict(self).values():
            if value is not None and value not in _DTYPE_BYTES:
                raise ValueError(f"unsupported dtype in policy: {value}")


@dataclass(frozen=True)
class ParameterSummary:
    total: int
    core: int
    screening: int


@dataclass(frozen=True)
class TrainingMemoryEstimate:
    parameter_summary: ParameterSummary
    dtype_policy: DTypePolicy
    model_axis_size: int
    parameter_storage_bytes: int
    master_parameter_bytes: int
    optimizer_state_bytes: int
    gradient_bytes: int
    update_workspace_bytes: int
    global_estimated_bytes: int
    per_device_estimated_bytes: int

    def to_dict(self):
        data = asdict(self)
        data["limitations"] = [
            "activation memory is not included",
            "RWKV and screening runtime state memory is not included",
            "compiler temporaries and collective buffers are not included",
        ]
        return data


def abstract_parameter_summary(config: ModelConfig) -> ParameterSummary:
    """Count the production NNX model without materializing its arrays."""

    model = nnx.eval_shape(
        lambda: NNXScreenedRWKVModel(
            config,
            rngs=nnx.Rngs(params=jax.random.key(0)),
        )
    )
    core = 0
    screening = 0
    for path, variable in nnx.to_flat_state(nnx.state(model, nnx.Param)):
        value = variable.get_value()
        count = math.prod(value.shape)
        if any(str(part).startswith("screening_") for part in path):
            screening += count
        else:
            core += count
    return ParameterSummary(total=core + screening, core=core, screening=screening)


def estimate_training_memory(
    config: ModelConfig,
    dtype_policy: DTypePolicy,
    *,
    model_axis_size: int,
    optimizer_slots: int = 2,
) -> TrainingMemoryEstimate:
    """Estimate parameter-related training memory before activation memory.

    The estimate is deliberately conservative about the optimizer update: an
    update-sized workspace is included in addition to gradients and Adam
    moments. Data-parallel axes replicate these bytes; only model-axis
    sharding reduces the per-device value.
    """
    if model_axis_size <= 0:
        raise ValueError("model_axis_size must be positive")
    if optimizer_slots < 0:
        raise ValueError("optimizer_slots must be non-negative")

    summary = abstract_parameter_summary(config)
    count = summary.total
    parameter_storage_bytes = count * _DTYPE_BYTES[dtype_policy.param_storage_dtype]
    master_parameter_bytes = (
        0
        if dtype_policy.master_param_dtype is None
        else count * _DTYPE_BYTES[dtype_policy.master_param_dtype]
    )
    optimizer_state_bytes = (
        count
        * optimizer_slots
        * _DTYPE_BYTES[dtype_policy.optimizer_state_dtype]
    )
    gradient_bytes = count * _DTYPE_BYTES[dtype_policy.gradient_accum_dtype]
    update_workspace_bytes = count * _DTYPE_BYTES[dtype_policy.param_update_dtype]
    global_estimated_bytes = (
        parameter_storage_bytes
        + master_parameter_bytes
        + optimizer_state_bytes
        + gradient_bytes
        + update_workspace_bytes
    )
    per_device_estimated_bytes = math.ceil(global_estimated_bytes / model_axis_size)
    return TrainingMemoryEstimate(
        parameter_summary=summary,
        dtype_policy=dtype_policy,
        model_axis_size=model_axis_size,
        parameter_storage_bytes=parameter_storage_bytes,
        master_parameter_bytes=master_parameter_bytes,
        optimizer_state_bytes=optimizer_state_bytes,
        gradient_bytes=gradient_bytes,
        update_workspace_bytes=update_workspace_bytes,
        global_estimated_bytes=global_estimated_bytes,
        per_device_estimated_bytes=per_device_estimated_bytes,
    )
