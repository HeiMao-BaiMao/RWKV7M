"""Validate two compute records before calculating ratios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_common import COMPUTE_BENCHMARK_SCHEMA_VERSION


def _require_equal(local, official, path):
    left = local
    right = official
    for key in path:
        left = left[key]
        right = right[key]
    if left != right:
        dotted = ".".join(map(str, path))
        raise SystemExit(f"incomparable records: {dotted} differs ({left!r} != {right!r})")


def _load(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != COMPUTE_BENCHMARK_SCHEMA_VERSION:
        raise SystemExit(f"unsupported compute benchmark schema: {path}")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    local = _load(args.local)
    official = _load(args.official)
    _require_equal(local, official, ("benchmark_kind",))
    for field in ("warmup", "iterations", "python_gc_disabled"):
        _require_equal(local, official, ("method", field))
    _require_equal(local, official, ("devices", 0, "device_kind"))

    kind = local["benchmark_kind"]
    if kind == "train_compute_only":
        for field in (
            "batch",
            "tokens",
            "dtype",
            "n_layers",
            "d_model",
            "d_ffn",
            "n_heads",
            "head_size",
            "vocab_size",
        ):
            _require_equal(local, official, ("shape", field))
        _require_equal(
            local,
            official,
            ("fixed_batch", "content_sha256"),
        )
        phase_map = {
            "forward": ("forward", "forward"),
            "backward": ("backward", "backward"),
            "optimizer": ("optimizer", "optimizer"),
            "full_step": ("full_step", "full_step"),
        }
    elif kind == "wkv_compute_only":
        for field in ("time", "batch", "heads", "head_size", "dtype"):
            _require_equal(local, official, ("shape", field))
        _require_equal(
            local,
            official,
            ("inputs", "sha256_float32_before_bf16_cast"),
        )
        if local["inputs"]["initial_state"] != "zero":
            raise SystemExit("official WKV comparison requires local zero initial state")
        phase_map = {
            "forward": ("pallas_training_forward", "forward"),
            "backward": ("pallas_backward", "backward"),
            "forward_backward": (
                "pallas_forward_backward",
                "forward_backward",
            ),
        }
    else:
        raise SystemExit(f"unsupported benchmark kind: {kind!r}")

    ratios = {}
    for phase, (local_name, official_name) in phase_map.items():
        local_ms = local["timings"][local_name]["median_ms"]
        official_ms = official["timings"][official_name]["median_ms"]
        ratios[phase] = {
            "local_median_ms": local_ms,
            "official_median_ms": official_ms,
            "local_throughput_over_official": official_ms / local_ms,
        }
    comparison = {
        "schema_version": COMPUTE_BENCHMARK_SCHEMA_VERSION,
        "comparison_kind": kind,
        "local_record": str(args.local.resolve()),
        "official_record": str(args.official.resolve()),
        "validated_equal_fields": {
            "method": ["warmup", "iterations", "python_gc_disabled"],
            "device": "first visible device kind",
            "input": (
                "fixed batch logical-content SHA-256"
                if kind == "train_compute_only"
                else "pre-BF16 tensor SHA-256 and zero initial state"
            ),
        },
        "ratios": ratios,
        "limitations": [
            "framework model wrappers and compiler stacks remain different",
            "full-model optimizer implementations remain different",
            "phase windows are diagnostic; full-step is the throughput value",
        ],
    }
    rendered = json.dumps(comparison, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
