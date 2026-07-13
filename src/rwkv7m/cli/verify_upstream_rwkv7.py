import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..io.upstream_rwkv7 import (
    convert_upstream_rwkv7_state_dict,
    load_upstream_rwkv7_reference_archive,
)
from ..model.screened_rwkv import ScreenedRWKVModel, init_rwkv_state
from ..model.state import init_screen_state


def _error_metrics(reference, actual, *, atol, rtol):
    reference = np.asarray(reference, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    if reference.shape != actual.shape:
        return {
            "passed": False,
            "reference_shape": list(reference.shape),
            "local_shape": list(actual.shape),
            "max_abs": None,
            "mean_abs": None,
            "rmse": None,
        }
    delta = actual - reference
    abs_delta = np.abs(delta)
    return {
        "passed": bool(np.allclose(actual, reference, atol=atol, rtol=rtol)),
        "reference_shape": list(reference.shape),
        "local_shape": list(actual.shape),
        "max_abs": float(np.max(abs_delta)) if abs_delta.size else 0.0,
        "mean_abs": float(np.mean(abs_delta)) if abs_delta.size else 0.0,
        "rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
    }


def _cross_entropy(logits, targets):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return float(-np.mean(np.take_along_axis(log_probs, targets[..., None], axis=-1)))


def verify_archive(path, *, atol=0.08, rtol=0.08, dtype=None, strict=True):
    archive = load_upstream_rwkv7_reference_archive(path)
    metadata = archive["metadata"]
    head_size = int(metadata.get("head_size", 64))
    params, spec, coverage = convert_upstream_rwkv7_state_dict(
        archive["weights"], head_size=head_size, strict=strict
    )
    local_dtype = dtype or metadata.get("local_dtype", "bfloat16")
    config = spec.model_config(dtype=local_dtype)
    config.max_seq_len = int(archive["input_ids"].shape[1])
    model = ScreenedRWKVModel(config)
    input_ids = jnp.asarray(archive["input_ids"], dtype=jnp.int32)
    logits, _, _, _ = model.apply(
        {"params": params},
        input_ids,
        init_rwkv_state(input_ids.shape[0], config),
        init_screen_state(input_ids.shape[0], config.screening),
        phase="read_screening_only",
        deterministic=True,
    )
    local_logits = np.asarray(logits, dtype=np.float32)
    logits_report = _error_metrics(
        archive["reference_logits"], local_logits, atol=atol, rtol=rtol
    )
    targets = archive["target_ids"]
    loss_report = None
    if targets is not None:
        reference_loss = _cross_entropy(archive["reference_logits"], targets)
        local_loss = _cross_entropy(local_logits, targets)
        loss_report = {
            "reference": reference_loss,
            "local": local_loss,
            "abs_diff": abs(reference_loss - local_loss),
        }
    coverage_passed = coverage.complete if strict else True
    passed = bool(logits_report["passed"] and coverage_passed)
    return {
        "passed": passed,
        "scope": "zero-initial-state sequence forward parity; not training-dynamics equivalence",
        "archive": str(Path(path)),
        "upstream_commit": metadata.get("upstream_commit"),
        "checkpoint": metadata.get("checkpoint"),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "dtype": local_dtype,
        "atol": float(atol),
        "rtol": float(rtol),
        "model": {
            "n_layers": spec.n_layers,
            "d_model": spec.d_model,
            "d_ffn": spec.d_ffn,
            "vocab_size": spec.vocab_size,
            "head_size": spec.head_size,
            "n_heads": spec.n_heads,
            "tokens": int(input_ids.shape[1]),
            "batch_size": int(input_ids.shape[0]),
        },
        "parameter_coverage": coverage.to_dict(),
        "logits": logits_report,
        "loss": loss_report,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify local JAX RWKV-7 core logits against an official CUDA reference archive."
    )
    parser.add_argument("reference_archive")
    parser.add_argument("--atol", type=float, default=0.08)
    parser.add_argument("--rtol", type=float, default=0.08)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--allow-unexpected-weights", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        report = verify_archive(
            args.reference_archive,
            atol=args.atol,
            rtol=args.rtol,
            dtype=args.dtype,
            strict=not args.allow_unexpected_weights,
        )
    except (OSError, ValueError, KeyError) as exc:
        report = {"passed": False, "error": str(exc)}
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report.get("passed"):
        logits = report["logits"]
        print(
            f"RWKV-7 core parity PASS max_abs={logits['max_abs']:.6g} "
            f"rmse={logits['rmse']:.6g} tensors={report['parameter_coverage']['converted_tensors']}"
        )
        return 0
    print(f"RWKV-7 core parity FAIL: {report.get('error', report.get('logits'))}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
