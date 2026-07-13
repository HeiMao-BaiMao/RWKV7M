import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_upstream_rwkv7_single_gpu.py"
SPEC = importlib.util.spec_from_file_location("upstream_single_gpu", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_upstream_learning_rate_matches_warmup_and_cosine_endpoints():
    kwargs = dict(lr_init=1e-3, lr_final=1e-5, warmup_steps=10, total_steps=100)
    assert MODULE.upstream_learning_rate(0, **kwargs) == pytest.approx(1e-5)
    assert MODULE.upstream_learning_rate(10, **kwargs) == pytest.approx(1e-3)
    assert MODULE.upstream_learning_rate(100, **kwargs) == pytest.approx(1e-5)


def test_upstream_learning_rate_rejects_exit_during_warmup():
    with pytest.raises(ValueError, match="greater than warmup"):
        MODULE.upstream_learning_rate(
            0, lr_init=1e-3, lr_final=1e-5, warmup_steps=10, total_steps=10
        )


@pytest.mark.parametrize(
    ("name", "ndim", "weight_decay", "expected"),
    [
        ("blocks.0.att.w0", 2, 0.1, "2x"),
        ("blocks.0.ffn.key.weight", 2, 0.1, "decay"),
        ("blocks.0.ln1.weight", 1, 0.1, "1x"),
        ("blocks.0.ffn.key.weight", 2, 0.0, "1x"),
        ("_forward_module.blocks.0.att.w0", 2, 0.1, "2x"),
    ],
)
def test_optimizer_group_matches_official_rules(name, ndim, weight_decay, expected):
    assert MODULE.optimizer_group(name, ndim, weight_decay) == expected


def test_epoch_and_index_preserves_official_40320_sample_boundary():
    assert MODULE.epoch_and_index(0) == (0, 0)
    assert MODULE.epoch_and_index(40_319) == (0, 40_319)
    assert MODULE.epoch_and_index(40_320) == (1, 0)
