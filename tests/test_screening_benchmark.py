from argparse import Namespace
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_screening_accelerator import _parity_gate, parse_args


def _args(**overrides):
    values = {
        "require_parity": True,
        "output_max_abs": 5e-3,
        "output_max_relative_l2": 5e-3,
        "gradient_max_abs": 1e-2,
        "gradient_max_relative_l2": 5e-3,
        "loss_atol": 1e-4,
    }
    values.update(overrides)
    return Namespace(**values)


def test_screening_benchmark_requires_parity_by_default():
    args = parse_args([])
    assert args.require_parity is True
    assert parse_args(["--no-require-parity"]).require_parity is False


def test_screening_benchmark_rejects_v5_accelerator_label():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--semantics-version",
                "screening-v5-core",
                "--write-mode",
                "competitive_novel",
            ]
        )


def test_screening_benchmark_parity_gate_is_fail_closed():
    passing = {"u": {"max_abs": 1e-4, "relative_l2": 2e-4}}
    gate = _parity_gate(passing, passing, 1e-6, _args())
    assert gate["passed"] is True
    assert gate["failures"] == []

    failing = {"u": {"max_abs": float("nan"), "relative_l2": 2e-4}}
    gate = _parity_gate(failing, passing, 1e-6, _args())
    assert gate["passed"] is False
    assert gate["failures"][0]["group"] == "output"

    gate = _parity_gate(passing, passing, 1e-3, _args())
    assert gate["passed"] is False
    assert gate["failures"][0]["group"] == "loss"
