import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict

from rwkv7m import (
    create_model_variables,
    load_model_safetensors,
    model_config_from_dict,
    model_config_to_dict,
    save_model_safetensors,
    tiny_config,
)


def test_model_config_roundtrip():
    config = tiny_config(vocab_size=32, d_model=32, n_layers=2, n_heads=2, head_size=16)
    config.head_chunk_size = 64
    restored = model_config_from_dict(model_config_to_dict(config))
    assert restored == config


def test_safetensors_roundtrip_params_and_config(tmp_path):
    config = tiny_config(vocab_size=32, d_model=32, n_layers=2, n_heads=2, head_size=16)
    variables, _ = create_model_variables(jax.random.PRNGKey(0), config, batch_size=1)

    path = tmp_path / "model.safetensors"
    save_model_safetensors(path, variables["params"], config, metadata={"step": 7})
    params, restored_config, metadata = load_model_safetensors(path)

    assert restored_config == config
    assert metadata["format"] == "rwkv7m"
    assert metadata["architecture"] == "rwkv7m"
    assert metadata["dtype"] == config.dtype
    assert metadata["d_model"] == str(config.d_model)
    assert metadata["n_layers"] == str(config.n_layers)
    assert metadata["screening_semantics_version"] == "screening-v4-legacy"
    assert metadata["tokenizer_format"] == "rwkv_vocab"
    assert len(metadata["tokenizer_vocab_sha256"]) == 64
    assert metadata["step"] == "7"

    original_flat = flatten_dict(variables["params"], sep="/")
    restored_flat = flatten_dict(params, sep="/")
    assert original_flat.keys() == restored_flat.keys()
    for key in original_flat:
        assert original_flat[key].shape == restored_flat[key].shape
        assert original_flat[key].dtype == restored_flat[key].dtype
        assert jnp.allclose(original_flat[key], restored_flat[key])
