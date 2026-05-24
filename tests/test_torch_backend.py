import pytest

from rwkv7m.backends.torch import flax_key_to_torch_key, is_torch_available, require_torch


def test_torch_backend_imports_without_torch_dependency():
    assert flax_key_to_torch_key("layer_0/rwkv_block_0/time_mix/key/kernel") == (
        "layer_0.rwkv_block_0.time_mix.key.kernel"
    )


def test_require_torch_reports_clear_error_when_missing():
    if is_torch_available():
        assert require_torch().__name__ == "torch"
    else:
        with pytest.raises(ImportError, match="PyTorch is not installed"):
            require_torch()
