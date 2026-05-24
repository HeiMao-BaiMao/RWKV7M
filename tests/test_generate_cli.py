import jax

from rwkv7m import create_model_variables, save_model_safetensors, tiny_config
from rwkv7m.cli.generate import parse_args


def test_generate_cli_parser_accepts_config_and_checkpoint(tmp_path):
    config = tmp_path / "generate.json"
    config.write_text(
        '{"checkpoint": "out/model.safetensors", "prompt": "a", "max_new_tokens": 1}',
        encoding="utf-8",
    )
    args = parse_args(["--config", str(config), "--prompt", "b"])
    assert args.checkpoint == "out/model.safetensors"
    assert args.prompt == "b"
    assert args.max_new_tokens == 1


def test_generate_checkpoint_can_be_written_for_cli(tmp_path):
    model_config = tiny_config(vocab_size=512, d_model=32, n_layers=2, n_heads=2, head_size=16)
    variables, _ = create_model_variables(jax.random.PRNGKey(0), model_config, batch_size=1)
    path = tmp_path / "model.safetensors"
    save_model_safetensors(path, variables["params"], model_config)
    assert path.exists()
