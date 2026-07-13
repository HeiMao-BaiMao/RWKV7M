import argparse
import json
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..io.upstream_rwkv7 import (
    convert_upstream_rwkv7_state_dict,
    load_upstream_rwkv7_reference_archive,
)
from ..model.screened_rwkv import (
    ScreenedRWKVModel,
    cross_entropy_loss,
    init_rwkv_state,
)
from ..model.state import init_screen_state
from ..train.train_step import l2wrap_loss
from ..train.train_state import create_optimizer


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


def _gradient_metrics(
    reference,
    actual,
    *,
    atol,
    rtol,
    min_cosine_similarity=0.0,
    max_relative_l2=float("inf"),
):
    reference_flat = flax.traverse_util.flatten_dict(reference, sep=".")
    actual_flat = flax.traverse_util.flatten_dict(actual, sep=".")
    if set(reference_flat) != set(actual_flat):
        return {
            "passed": False,
            "missing_local": sorted(set(reference_flat) - set(actual_flat)),
            "unexpected_local": sorted(set(actual_flat) - set(reference_flat)),
        }
    per_tensor = {}
    reference_parts = []
    actual_parts = []
    for name in sorted(reference_flat):
        ref = np.asarray(reference_flat[name], dtype=np.float32)
        local = np.asarray(actual_flat[name], dtype=np.float32)
        metrics = _error_metrics(ref, local, atol=atol, rtol=rtol)
        ref_norm = float(np.linalg.norm(ref.ravel().astype(np.float64)))
        delta_norm = float(np.linalg.norm((local - ref).ravel().astype(np.float64)))
        metrics["reference_l2"] = ref_norm
        metrics["relative_l2"] = delta_norm / max(ref_norm, 1e-12)
        per_tensor[name] = metrics
        reference_parts.append(ref.ravel())
        actual_parts.append(local.ravel())
    reference_vector = np.concatenate(reference_parts).astype(np.float64)
    actual_vector = np.concatenate(actual_parts).astype(np.float64)
    delta = actual_vector - reference_vector
    ref_norm = float(np.linalg.norm(reference_vector))
    local_norm = float(np.linalg.norm(actual_vector))
    denominator = ref_norm * local_norm
    cosine = float(np.dot(reference_vector, actual_vector) / denominator) if denominator else 1.0
    failed_names = [name for name, metrics in per_tensor.items() if not metrics["passed"]]
    relative_l2 = float(np.linalg.norm(delta) / max(ref_norm, 1e-12))
    return {
        "passed": bool(
            not failed_names
            and cosine >= min_cosine_similarity
            and relative_l2 <= max_relative_l2
        ),
        "tensors": len(per_tensor),
        "failed_tensors": failed_names,
        "global_cosine_similarity": cosine,
        "global_relative_l2": relative_l2,
        "global_max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "min_cosine_similarity": float(min_cosine_similarity),
        "max_relative_l2": float(max_relative_l2),
        "per_tensor": per_tensor,
    }


def verify_archive(
    path,
    *,
    atol=0.08,
    rtol=0.08,
    gradient_atol=0.01,
    gradient_rtol=0.1,
    gradient_min_cosine=0.99,
    gradient_max_relative_l2=0.02,
    optimizer_atol=0.01,
    optimizer_rtol=0.1,
    optimizer_min_cosine=0.99,
    optimizer_max_relative_l2=0.1,
    dtype=None,
    strict=True,
):
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
    rwkv_state = init_rwkv_state(input_ids.shape[0], config)
    screen_state = init_screen_state(input_ids.shape[0], config.screening)

    def forward(local_params):
        return model.apply(
            {"params": local_params},
            input_ids,
            rwkv_state,
            screen_state,
            phase="read_screening_only",
            deterministic=True,
        )[0]

    logits = forward(params)
    local_logits = np.asarray(logits, dtype=np.float32)
    logits_report = _error_metrics(
        archive["reference_logits"], local_logits, atol=atol, rtol=rtol
    )
    targets = archive["target_ids"]
    loss_report = None
    gradient_report = None
    optimizer_report = None
    if targets is not None:
        reference_loss = _cross_entropy(archive["reference_logits"], targets)
        local_loss = _cross_entropy(local_logits, targets)
        loss_report = {
            "reference": reference_loss,
            "local": local_loss,
            "abs_diff": abs(reference_loss - local_loss),
        }
        if archive["gradients"]:
            reference_gradients, _, gradient_coverage = convert_upstream_rwkv7_state_dict(
                archive["gradients"], head_size=head_size, strict=strict
            )

            def total_loss(local_params):
                local_logits = forward(local_params)
                return cross_entropy_loss(local_logits, jnp.asarray(targets)) + l2wrap_loss(
                    local_logits
                )

            local_total_loss, local_gradients = jax.value_and_grad(total_loss)(params)
            gradient_report = _gradient_metrics(
                reference_gradients,
                local_gradients,
                atol=gradient_atol,
                rtol=gradient_rtol,
                min_cosine_similarity=gradient_min_cosine,
                max_relative_l2=gradient_max_relative_l2,
            )
            gradient_report["parameter_coverage"] = gradient_coverage.to_dict()
            gradient_report["reference_loss"] = archive["reference_loss"]
            gradient_report["local_loss"] = float(local_total_loss)
            if archive["updated_weights"]:
                reference_updated, _, updated_coverage = convert_upstream_rwkv7_state_dict(
                    archive["updated_weights"], head_size=head_size, strict=strict
                )
                optimizer_config = metadata.get("optimizer_step")
                if not optimizer_config:
                    raise ValueError("updated weights require optimizer_step metadata")
                optimizer = create_optimizer(
                    {
                        "lr_schedule": "rwkv",
                        "lr_init": optimizer_config["lr"],
                        "lr_final": optimizer_config["lr"],
                        "warmup_steps": 0,
                        "weight_decay": optimizer_config["weight_decay"],
                        "max_grad_norm": optimizer_config["grad_clip"],
                        "adam_beta1": optimizer_config["betas"][0],
                        "adam_beta2": optimizer_config["betas"][1],
                        "adam_eps": optimizer_config["eps"],
                    },
                    total_steps=1,
                )
                updates, _ = optimizer.update(
                    local_gradients, optimizer.init(params), params
                )
                local_updated = optax.apply_updates(params, updates)
                local_updated_bf16 = jax.tree.map(
                    lambda value: jnp.asarray(value, dtype=jnp.bfloat16).astype(jnp.float32),
                    local_updated,
                )
                reference_update_delta = jax.tree.map(
                    lambda updated, initial: np.asarray(updated) - np.asarray(initial),
                    reference_updated,
                    params,
                )
                local_update_delta = jax.tree.map(
                    lambda updated, initial: np.asarray(updated) - np.asarray(initial),
                    local_updated_bf16,
                    params,
                )
                optimizer_report = _gradient_metrics(
                    reference_update_delta,
                    local_update_delta,
                    atol=optimizer_atol,
                    rtol=optimizer_rtol,
                    min_cosine_similarity=optimizer_min_cosine,
                    max_relative_l2=optimizer_max_relative_l2,
                )
                optimizer_report["parameter_coverage"] = updated_coverage.to_dict()
                optimizer_report["comparison_dtype"] = "bfloat16"
                optimizer_report["metric_target"] = "updated_weight - initial_weight"
    coverage_passed = coverage.complete if strict else True
    gradients_passed = gradient_report is None or gradient_report["passed"]
    optimizer_passed = optimizer_report is None or optimizer_report["passed"]
    passed = bool(
        logits_report["passed"]
        and coverage_passed
        and gradients_passed
        and optimizer_passed
    )
    if optimizer_report is not None:
        scope = (
            "zero-initial-state sequence forward, backward, and one-step optimizer parity; "
            "not multi-step training-dynamics equivalence"
        )
    elif gradient_report is not None:
        scope = (
            "zero-initial-state sequence forward and backward parity; "
            "not optimizer or training-dynamics equivalence"
        )
    else:
        scope = "zero-initial-state sequence forward parity; not training-dynamics equivalence"
    return {
        "passed": passed,
        "scope": scope,
        "archive": str(Path(path)),
        "upstream_commit": metadata.get("upstream_commit"),
        "checkpoint": metadata.get("checkpoint"),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "dtype": local_dtype,
        "atol": float(atol),
        "rtol": float(rtol),
        "gradient_atol": float(gradient_atol),
        "gradient_rtol": float(gradient_rtol),
        "gradient_min_cosine": float(gradient_min_cosine),
        "gradient_max_relative_l2": float(gradient_max_relative_l2),
        "optimizer_atol": float(optimizer_atol),
        "optimizer_rtol": float(optimizer_rtol),
        "optimizer_min_cosine": float(optimizer_min_cosine),
        "optimizer_max_relative_l2": float(optimizer_max_relative_l2),
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
        "gradients": gradient_report,
        "optimizer_step": optimizer_report,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify local JAX RWKV-7 core logits against an official CUDA reference archive."
    )
    parser.add_argument("reference_archive")
    parser.add_argument("--atol", type=float, default=0.08)
    parser.add_argument("--rtol", type=float, default=0.08)
    parser.add_argument("--gradient-atol", type=float, default=0.01)
    parser.add_argument("--gradient-rtol", type=float, default=0.1)
    parser.add_argument("--gradient-min-cosine", type=float, default=0.99)
    parser.add_argument("--gradient-max-relative-l2", type=float, default=0.02)
    parser.add_argument("--optimizer-atol", type=float, default=0.01)
    parser.add_argument("--optimizer-rtol", type=float, default=0.1)
    parser.add_argument("--optimizer-min-cosine", type=float, default=0.99)
    parser.add_argument("--optimizer-max-relative-l2", type=float, default=0.1)
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
            gradient_atol=args.gradient_atol,
            gradient_rtol=args.gradient_rtol,
            gradient_min_cosine=args.gradient_min_cosine,
            gradient_max_relative_l2=args.gradient_max_relative_l2,
            optimizer_atol=args.optimizer_atol,
            optimizer_rtol=args.optimizer_rtol,
            optimizer_min_cosine=args.optimizer_min_cosine,
            optimizer_max_relative_l2=args.optimizer_max_relative_l2,
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
        gradient_summary = ""
        if report.get("gradients") is not None:
            gradient_summary = (
                f" grad_cos={report['gradients']['global_cosine_similarity']:.6g}"
                f" grad_rel_l2={report['gradients']['global_relative_l2']:.6g}"
            )
        optimizer_summary = ""
        if report.get("optimizer_step") is not None:
            optimizer_summary = (
                f" update_cos={report['optimizer_step']['global_cosine_similarity']:.6g}"
                f" update_rel_l2={report['optimizer_step']['global_relative_l2']:.6g}"
            )
        print(
            f"RWKV-7 core parity PASS max_abs={logits['max_abs']:.6g} "
            f"rmse={logits['rmse']:.6g} tensors={report['parameter_coverage']['converted_tensors']}"
            f"{gradient_summary}"
            f"{optimizer_summary}"
        )
        return 0
    print(f"RWKV-7 core parity FAIL: {report.get('error', report.get('logits'))}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
