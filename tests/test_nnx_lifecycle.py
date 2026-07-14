import json
from pathlib import Path
import subprocess
import sys


def _run_probe(checkpoint_dir, mode):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rwkv7m.cli.verify_nnx_lifecycle",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--mode",
            mode,
            "--width",
            "8",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_nnx_lifecycle_restores_in_a_new_process_and_keeps_sharding(tmp_path):
    checkpoint_dir = Path(tmp_path) / "nnx-checkpoint"
    created = _run_probe(checkpoint_dir, "create")
    restored = _run_probe(checkpoint_dir, "restore")

    assert created["optimizer_step"] == 1
    assert restored["optimizer_step"] == 2
    assert restored["loss"] < created["loss"]

    expected_axes = {
        "0/in_proj/kernel": ([None, "model"], "P(None, 'model')"),
        "0/out_proj/kernel": (["model", None], "P('model', None)"),
        "1/opt_state/0/mu/in_proj/kernel": ([None, "model"], "P(None, 'model')"),
        "1/opt_state/0/nu/out_proj/kernel": (["model", None], "P('model', None)"),
    }
    for path, (logical_axes, sharding) in expected_axes.items():
        assert created["sharding_after"][path]["logical_axes"] == logical_axes
        assert restored["sharding_before"][path]["logical_axes"] == logical_axes
        assert restored["sharding_after"][path]["logical_axes"] == logical_axes
        assert restored["sharding_before"][path]["sharding"] == sharding
