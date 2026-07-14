from pathlib import Path

from rwkv7m import (
    MODEL_PRESET_NAMES,
    MODEL_PRESET_PARAMETER_COUNTS,
    MODEL_PRESET_RECOMMENDED_MODEL_AXIS_SIZES,
    load_model_config,
    model_config_to_dict,
    model_preset,
)
from rwkv7m.cli.bench_binidx import build_config as build_bench_config
from rwkv7m.cli.bench_binidx import parse_args as parse_bench_args
from rwkv7m.cli.eval_binidx import build_config as build_eval_config
from rwkv7m.cli.eval_binidx import parse_args as parse_eval_args
from rwkv7m.cli.plan_scale import parse_args as parse_scale_args
from rwkv7m.cli.train_binidx import build_config as build_train_config
from rwkv7m.cli.train_binidx_distributed import parse_args as parse_distributed_args
from rwkv7m.distributed import abstract_parameter_summary


PRESET_FILES = {
    "0.185b": Path("configs/rwkv7m-0.185b.json.example"),
    "0.3b": Path("configs/rwkv7m-0.3b.json.example"),
    "1b": Path("configs/rwkv7m-1b.json.example"),
    "3b": Path("configs/rwkv7m-3b.json.example"),
    "7b": Path("configs/rwkv7m-7b-tpu.json.example"),
}


def test_named_presets_match_tracked_json_examples():
    assert MODEL_PRESET_NAMES == ("0.185b", "0.3b", "1b", "3b", "7b")
    assert set(MODEL_PRESET_RECOMMENDED_MODEL_AXIS_SIZES) == set(
        MODEL_PRESET_NAMES
    )
    for name, path in PRESET_FILES.items():
        assert model_config_to_dict(model_preset(name)) == model_config_to_dict(
            load_model_config(path)
        )

    assert model_preset("0.19B") == model_preset("0.185b")
    assert model_preset("300m") == model_preset("0.3b")
    assert model_preset("1b") is not model_preset("1b")


def test_named_preset_parameter_counts_match_the_production_nnx_tree():
    actual = {
        name: abstract_parameter_summary(model_preset(name)).total
        for name in MODEL_PRESET_NAMES
    }
    assert actual == MODEL_PRESET_PARAMETER_COUNTS


def test_training_evaluation_planner_and_benchmark_accept_model_presets():
    train_args = parse_distributed_args(
        [
            "--data-file",
            "unused",
            "--model-preset",
            "1b",
            "--ctx-len",
            "2048",
            "--global-batch-size",
            "2",
            "--steps",
            "0",
        ]
    )
    assert build_train_config(train_args) == model_preset("1b")

    eval_args = parse_eval_args(
        [
            "--data-file",
            "unused",
            "--model-preset",
            "0.185b",
            "--ctx-len",
            "512",
        ]
    )
    assert build_eval_config(eval_args) == model_preset("0.185b")

    scale_args = parse_scale_args(
        ["--model-preset", "3b", "--model-axis-size", "4"]
    )
    assert scale_args.model_preset == "3b"

    bench_args = parse_bench_args(
        [
            "--data-file",
            "unused",
            "--model-preset",
            "0.3b",
            "--ctx-len",
            "1024",
        ]
    )
    baseline = build_bench_config(bench_args, "baseline")
    read_write = build_bench_config(bench_args, "read_write")
    assert baseline.use_screening is False
    assert read_write == model_preset("0.3b")
