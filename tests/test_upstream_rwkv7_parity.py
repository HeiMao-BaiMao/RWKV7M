import json

import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from rwkv7m.cli.verify_upstream_rwkv7 import main as verify_main, verify_archive
from rwkv7m.io.upstream_rwkv7 import (
    _mapping,
    convert_upstream_rwkv7_state_dict,
    infer_upstream_rwkv7_spec,
)
from rwkv7m.model.screened_rwkv import (
    ModelConfig,
    cross_entropy_loss,
    create_model_variables,
    init_rwkv_state,
)
from rwkv7m.model.screening import ScreeningConfig
from rwkv7m.model.state import init_screen_state
from rwkv7m.model.rwkv_core import wkv_step
from rwkv7m.train.train_step import l2wrap_loss
from rwkv7m.train.train_state import create_optimizer


def tiny_config():
    return ModelConfig(
        d_model=32,
        d_ffn=64,
        n_layers=2,
        n_heads=1,
        head_size=32,
        vocab_size=48,
        max_seq_len=4,
        dtype="float32",
        use_screening=False,
        screening=ScreeningConfig(),
    )


def upstream_from_local(params, config):
    flat = flax.traverse_util.flatten_dict(params, sep=".")
    spec = infer_upstream_rwkv7_spec(
        {
            "emb.weight": np.asarray(flat["token_embedding.embedding"]),
            "blocks.0.ffn.key.weight": np.zeros((config.d_ffn, config.d_model)),
            "blocks.0.marker": np.zeros(()),
            "blocks.1.marker": np.zeros(()),
        },
        head_size=config.head_size,
    )
    result = {}
    for source, target, shape, transpose in _mapping(spec):
        if target is None:
            result[source] = np.zeros(shape, dtype=np.float32)
            continue
        value = np.asarray(flat[target])
        if transpose:
            value = value.T
        if source.rsplit(".", 1)[-1].startswith("x_") and ".att." in source:
            value = value.reshape(1, 1, config.d_model)
        result[source] = value
    return result


def build_reference_archive(
    tmp_path,
    *,
    perturb=0.0,
    include_gradients=False,
    gradient_perturb=0.0,
    include_optimizer_step=False,
):
    config = tiny_config()
    variables, model = create_model_variables(jax.random.PRNGKey(7), config, batch_size=1)
    input_ids = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
    logits, _, _, _ = model.apply(
        variables,
        input_ids,
        init_rwkv_state(1, config),
        init_screen_state(1, config.screening),
        phase="read_screening_only",
        deterministic=True,
    )
    weights = upstream_from_local(variables["params"], config)
    metadata = {
        "head_size": 32,
        "local_dtype": "float32",
        "upstream_commit": "test",
        "upstream_dirty": True,
        "upstream_diff_sha256": "abc123",
    }
    payload = {
        "input_ids": np.asarray(input_ids),
        "target_ids": np.asarray([[2, 3, 4, 5]], dtype=np.int32),
        "reference_logits": np.asarray(logits) + perturb,
    }
    payload.update({f"weight::{name}": value for name, value in weights.items()})
    if include_gradients:
        targets = jnp.asarray(payload["target_ids"])

        def total_loss(params):
            local_logits, _, _, _ = model.apply(
                {"params": params},
                input_ids,
                init_rwkv_state(1, config),
                init_screen_state(1, config.screening),
                phase="read_screening_only",
                deterministic=True,
            )
            return cross_entropy_loss(local_logits, targets) + l2wrap_loss(local_logits)

        loss, gradients = jax.value_and_grad(total_loss)(variables["params"])
        upstream_gradients = upstream_from_local(gradients, config)
        upstream_gradients["head.weight"] = (
            upstream_gradients["head.weight"] + gradient_perturb
        )
        payload["reference_loss"] = np.asarray(loss, dtype=np.float32)
        payload.update(
            {f"gradient::{name}": value for name, value in upstream_gradients.items()}
        )
        if include_optimizer_step:
            optimizer_config = {
                "name": "test",
                "lr": 1e-3,
                "w0_lr_scale": 2.0,
                "weight_decay": 0.001,
                "grad_clip": 1.0,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            }
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
                gradients, optimizer.init(variables["params"]), variables["params"]
            )
            updated = optax.apply_updates(variables["params"], updates)
            updated_bf16 = jax.tree.map(
                lambda value: jnp.asarray(value, dtype=jnp.bfloat16).astype(jnp.float32),
                updated,
            )
            upstream_updated = upstream_from_local(updated_bf16, config)
            payload.update(
                {f"updated_weight::{name}": value for name, value in upstream_updated.items()}
            )
            metadata["optimizer_step"] = optimizer_config
    payload["metadata_json"] = np.asarray(json.dumps(metadata))
    path = tmp_path / "reference.npz"
    np.savez(path, **payload)
    return path, weights, variables["params"]


def test_complete_mapping_transposes_and_reshapes(tmp_path):
    _, weights, original_params = build_reference_archive(tmp_path)
    converted, spec, report = convert_upstream_rwkv7_state_dict(weights, head_size=32)
    assert spec.n_layers == 2
    assert spec.d_model == 32
    assert spec.d_ffn == 64
    assert report.complete
    assert report.ignored_names == (
        "blocks.0.att.v0",
        "blocks.0.att.v1",
        "blocks.0.att.v2",
    )
    original = flax.traverse_util.flatten_dict(original_params, sep=".")
    actual = flax.traverse_util.flatten_dict(converted, sep=".")
    assert set(actual) == set(original)
    for name in original:
        np.testing.assert_array_equal(actual[name], original[name])
    assert weights["blocks.0.att.x_r"].shape == (1, 1, 32)
    assert actual["layer_0.rwkv_block_0.att.x_r"].shape == (32,)


def test_mapping_rejects_missing_and_mismatched_tensors(tmp_path):
    _, weights, _ = build_reference_archive(tmp_path)
    missing = dict(weights)
    missing.pop("head.weight")
    with pytest.raises(ValueError, match="missing required upstream tensor: head.weight"):
        convert_upstream_rwkv7_state_dict(missing, head_size=32)
    mismatched = dict(weights)
    mismatched["blocks.0.att.w1"] = np.zeros((1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch for blocks.0.att.w1"):
        convert_upstream_rwkv7_state_dict(mismatched, head_size=32)


def test_verify_cli_passes_matching_archive(tmp_path):
    path, _, _ = build_reference_archive(tmp_path)
    report_path = tmp_path / "report.json"
    assert verify_main([str(path), "--atol", "1e-6", "--rtol", "1e-6", "--json-out", str(report_path)]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["parameter_coverage"]["complete"] is True
    assert report["logits"]["max_abs"] == pytest.approx(0.0, abs=1e-6)
    assert report["upstream_dirty"] is True
    assert report["upstream_diff_sha256"] == "abc123"


def test_verify_cli_fails_perturbed_reference(tmp_path):
    path, _, _ = build_reference_archive(tmp_path, perturb=1.0)
    assert verify_main([str(path), "--atol", "1e-6", "--rtol", "1e-6"]) == 1


def test_verify_cli_checks_matching_gradients(tmp_path):
    path, _, _ = build_reference_archive(tmp_path, include_gradients=True)
    report = verify_archive(
        path,
        atol=1e-6,
        rtol=1e-6,
        gradient_atol=1e-6,
        gradient_rtol=1e-6,
    )
    assert report["passed"] is True
    assert report["gradients"]["passed"] is True
    assert report["gradients"]["global_cosine_similarity"] == pytest.approx(1.0)
    assert report["gradients"]["global_relative_l2"] == pytest.approx(0.0, abs=1e-7)


def test_verify_cli_fails_perturbed_gradients(tmp_path):
    path, _, _ = build_reference_archive(
        tmp_path, include_gradients=True, gradient_perturb=1.0
    )
    report = verify_archive(
        path,
        gradient_atol=1e-6,
        gradient_rtol=1e-6,
    )
    assert report["passed"] is False
    assert report["gradients"]["passed"] is False
    assert "lm_head.kernel" in report["gradients"]["failed_tensors"]


def test_verify_cli_checks_matching_optimizer_step(tmp_path):
    path, _, _ = build_reference_archive(
        tmp_path, include_gradients=True, include_optimizer_step=True
    )
    report = verify_archive(
        path,
        gradient_atol=1e-6,
        gradient_rtol=1e-6,
        optimizer_atol=1e-6,
        optimizer_rtol=1e-6,
    )
    assert report["passed"] is True
    assert report["optimizer_step"]["passed"] is True
    assert report["optimizer_step"]["global_relative_l2"] == pytest.approx(
        0.0, abs=1e-7
    )


def test_wkv_step_matches_official_cuda_row_update():
    rng = np.random.default_rng(19)
    n = 4
    state = rng.normal(size=(1, 1, n, n)).astype(np.float32)
    r = rng.normal(size=(1, 1, n)).astype(np.float32)
    raw_w = rng.normal(size=(1, 1, n)).astype(np.float32)
    k = rng.normal(size=(1, 1, n)).astype(np.float32)
    v = rng.normal(size=(1, 1, n)).astype(np.float32)
    a = rng.normal(size=(1, 1, n)).astype(np.float32)
    b = rng.normal(size=(1, 1, n)).astype(np.float32)

    # Official rwkv7_clampw.cu: every row i computes sa=S[i]@a,
    # then S[i,j]=S[i,j]*decay[j]+sa*b[j]+k[j]*v[i], y[i]=S[i]@r.
    decay = np.exp(-np.exp(raw_w))
    sa = np.einsum("bhij,bhj->bhi", state, a)
    expected_state = (
        state * decay[..., None, :]
        + np.einsum("bhi,bhj->bhij", sa, b)
        + np.einsum("bhi,bhj->bhij", v, k)
    )
    expected_y = np.einsum("bhij,bhj->bhi", expected_state, r)

    actual_state, actual_y = wkv_step(
        jnp.asarray(state),
        jnp.asarray(r),
        jnp.asarray(raw_w),
        jnp.asarray(k),
        jnp.asarray(v),
        jnp.asarray(a),
        jnp.asarray(b),
    )
    np.testing.assert_allclose(actual_state, expected_state, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(actual_y, expected_y, atol=1e-6, rtol=1e-6)
