import json
import os
from pathlib import Path
import subprocess
import sys

from rwkv7m.distributed.collectives import audit_collectives


def test_collective_audit_counts_compiled_hlo_operations():
    hlo = """
      %all-reduce.1 = f32[8] all-reduce(%x), replica_groups={{0,1}}
      %all-gather.2 = f32[16] all-gather(%y), dimensions={0}
      %all-to-all.3 = f32[16] all-to-all(%z), dimensions={0}
    """
    audit = audit_collectives(hlo)
    assert audit.counts["all-reduce"] == 1
    assert audit.counts["all-gather"] == 1
    assert audit.counts["all-to-all"] == 1
    assert audit.total == 3


def test_full_nnx_model_parallel_path_on_two_cpu_devices():
    env = os.environ.copy()
    env["JAX_NUM_CPU_DEVICES"] = "2"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rwkv7m.cli.audit_nnx_model_parallel",
            "--model-axis-size",
            "2",
            "--screening",
            "--write-screening",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    report = json.loads(result.stdout)
    assert report["mesh"]["shape"] == {"data": 1, "model": 2}
    assert report["mesh"]["axis_types"] == [
        "AxisType.Explicit",
        "AxisType.Explicit",
    ]
    assert report["contract"] == {
        "has_column_parallel_kernel": True,
        "has_row_parallel_kernel": True,
        "model_parallel_exercised": True,
    }
    assert report["forward"]["finite"] is True
    assert report["forward"]["collectives"]["counts"]["all-reduce"] > 0
    assert report["forward"]["collectives"]["total"] > 0
    assert report["train_step"]["finite"] is True
    assert report["train_step"]["optimizer_step"] == 1
    assert report["states"]["wkv"]["sharding"] == (
        "P('data', 'model', None, None)"
    )
    assert report["states"]["screening_slots"]["sharding"] == (
        "P('data', None, 'model')"
    )
    assert report["parameters"]["token_embedding/embedding"]["sharding"] == (
        "P(None, 'model')"
    )
