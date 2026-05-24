import json

from rwkv7m.cli.bench_binidx import parse_args as parse_bench_args
from rwkv7m.cli.eval_binidx import parse_args as parse_eval_args
from rwkv7m.cli.train_binidx import parse_args as parse_train_args


def write_config(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_train_config_file_supplies_required_values_and_cli_overrides(tmp_path):
    config = write_config(
        tmp_path,
        {
            "data_file": "data/minipile",
            "ctx_len": 512,
            "batch_size": 2,
            "steps": 10,
            "vocab_size": 65536,
        },
    )
    args = parse_train_args(["--config", str(config), "--steps", "3"])
    assert args.data_file == "data/minipile"
    assert args.ctx_len == 512
    assert args.batch_size == 2
    assert args.steps == 3


def test_eval_and_bench_config_file_parse(tmp_path):
    config = write_config(tmp_path, {"data_file": "data/minipile", "ctx_len": 128})
    eval_args = parse_eval_args(["--config", str(config), "--steps", "2"])
    bench_args = parse_bench_args(["--config", str(config), "--steps", "2"])
    assert eval_args.data_file == "data/minipile"
    assert bench_args.data_file == "data/minipile"
    assert eval_args.ctx_len == 128
    assert bench_args.ctx_len == 128
