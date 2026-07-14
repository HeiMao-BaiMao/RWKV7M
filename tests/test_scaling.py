import json

from rwkv7m.cli.plan_scale import build_report, parse_args
from rwkv7m.distributed import (
    DTypePolicy,
    abstract_parameter_summary,
    estimate_training_memory,
    runtime_version_manifest,
)
from rwkv7m.model import ModelConfig, ScreeningConfig


def seven_b_config():
    return ModelConfig(
        d_model=4096,
        d_ffn=15232,
        n_layers=32,
        n_heads=64,
        head_size=64,
        vocab_size=65536,
        max_seq_len=4096,
        dtype="bfloat16",
        use_screening=True,
        screening=ScreeningConfig(
            d_model=4096,
            d_slot=1024,
            d_k=128,
            d_v=512,
            n_slots=16,
            screened_layers=(7, 15, 23, 31),
            bank_ids=(0,) * 8 + (1,) * 4 + (2,) * 4,
            use_write_screening=True,
        ),
    )


def test_abstract_parameter_summary_matches_current_seven_b_shapes():
    summary = abstract_parameter_summary(seven_b_config())
    assert summary.core == 6_871_986_176
    assert summary.screening == 122_802_200
    assert summary.total == 6_994_788_376


def test_memory_profile_uses_less_parameter_storage_than_stability_profile():
    config = ModelConfig(
        d_model=32,
        d_ffn=64,
        n_layers=2,
        n_heads=2,
        head_size=16,
        vocab_size=64,
        use_screening=False,
    )
    stability = estimate_training_memory(
        config,
        DTypePolicy.stability(),
        model_axis_size=2,
    )
    memory = estimate_training_memory(
        config,
        DTypePolicy.memory(),
        model_axis_size=2,
    )
    assert memory.parameter_summary == stability.parameter_summary
    assert memory.parameter_storage_bytes * 2 == stability.parameter_storage_bytes
    assert memory.per_device_estimated_bytes < stability.per_device_estimated_bytes


def test_runtime_version_manifest_records_fixed_packages():
    manifest = runtime_version_manifest()
    assert manifest["packages"]["jax"]
    assert manifest["packages"]["jaxlib"]
    assert manifest["packages"]["flax"]
    assert manifest["packages"]["optax"]
    assert manifest["packages"]["orbax-checkpoint"]
    assert manifest["device_count"] >= 1


def test_plan_scale_builds_json_serializable_report(tmp_path):
    config_path = tmp_path / "model.json"
    config_path.write_text(
        json.dumps(
            {
                "d_model": 32,
                "d_ffn": 64,
                "n_layers": 2,
                "n_heads": 2,
                "head_size": 16,
                "vocab_size": 64,
                "use_screening": False,
            }
        ),
        encoding="utf-8",
    )
    args = parse_args(
        [
            "--model-config",
            str(config_path),
            "--model-axis-size",
            "1",
            "--dtype-profile",
            "memory",
        ]
    )
    report = build_report(args)
    assert report["estimate"]["dtype_policy"]["param_storage_dtype"] == "bfloat16"
    json.dumps(report)
