import jax
import jax.numpy as jnp
import pytest

from rwkv7m.kernels import (
    WKVFFIBackend,
    register_wkv_ffi_backend,
    resolve_wkv_backend,
    unregister_wkv_ffi_backend,
)
from rwkv7m.kernels.wkv_pallas_gpu import (
    WKVGPUConfig,
    wkv7_pallas_gpu_forward_with_aux,
)
from rwkv7m.kernels.wkv_pallas_tpu import (
    WKVTPUConfig,
    wkv7_pallas_tpu_forward_with_aux,
)
from rwkv7m.model.wkv import wkv7, wkv7_reference


def _inputs(dtype=jnp.float32, *, time=5):
    keys = jax.random.split(jax.random.key(17), 7)
    vectors = tuple(
        (jax.random.normal(key, (time, 1, 2, 4)) * 0.1).astype(dtype)
        for key in keys[:6]
    )
    state = (
        jax.random.normal(keys[6], (1, 2, 4, 4)) * 0.1
    ).astype(jnp.float32)
    return (*vectors, state)


@pytest.mark.parametrize(
    "backend", ["pallas_gpu_triton", "pallas_tpu"]
)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_pallas_interpret_forward_matches_reference(backend, dtype):
    inputs = _inputs(dtype)
    expected = wkv7_reference(*inputs)

    actual = jax.jit(
        lambda *values: wkv7(*values, backend, True)
    )(*inputs)

    assert actual[0].dtype == dtype
    assert actual[1].dtype == jnp.float32
    assert jnp.allclose(actual[0], expected[0], rtol=1e-3, atol=1e-3)
    assert jnp.allclose(actual[1], expected[1], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "backend", ["pallas_gpu_triton", "pallas_tpu"]
)
def test_pallas_custom_vjp_matches_reference_for_all_inputs(backend):
    inputs = _inputs()

    def loss(fn, *values):
        y, final_state = fn(*values)
        return jnp.sum(jnp.square(y)) + 0.03 * jnp.sum(
            jnp.square(final_state)
        )

    argnums = tuple(range(len(inputs)))
    expected = jax.grad(
        lambda *values: loss(wkv7_reference, *values),
        argnums=argnums,
    )(*inputs)
    actual = jax.jit(
        jax.grad(
            lambda *values: loss(
                lambda *args: wkv7(*args, backend, True), *values
            ),
            argnums=argnums,
        )
    )(*inputs)

    assert all(
        jnp.allclose(left, right, rtol=1e-5, atol=1e-6)
        for left, right in zip(actual, expected, strict=True)
    )


@pytest.mark.parametrize(
    "backend", ["pallas_gpu_triton", "pallas_tpu"]
)
def test_pallas_custom_vjp_remains_finite_when_decay_rounds_to_zero(
    backend,
):
    inputs = list(_inputs(time=3))
    inputs[1] = inputs[1].at[1].set(5.0)

    def loss(fn, *values):
        y, final_state = fn(*values)
        return jnp.sum(jnp.square(y)) + 0.03 * jnp.sum(
            jnp.square(final_state)
        )

    argnums = tuple(range(len(inputs)))
    expected = jax.grad(
        lambda *values: loss(wkv7_reference, *values),
        argnums=argnums,
    )(*inputs)
    actual = jax.jit(
        jax.grad(
            lambda *values: loss(
                lambda *args: wkv7(*args, backend, True), *values
            ),
            argnums=argnums,
        )
    )(*inputs)

    assert all(jnp.all(jnp.isfinite(value)) for value in actual)
    assert all(
        jnp.allclose(left, right, rtol=1e-5, atol=1e-6)
        for left, right in zip(actual, expected, strict=True)
    )


def test_pallas_aux_uses_interval_checkpoints_for_partial_final_chunk():
    inputs = _inputs(time=5)
    gpu_config = WKVGPUConfig(checkpoint_interval=2)
    tpu_config = WKVTPUConfig(checkpoint_interval=2)

    gpu_outputs, gpu_aux = wkv7_pallas_gpu_forward_with_aux(
        *inputs,
        lowering="triton",
        config=gpu_config,
        interpret=True,
    )
    tpu_outputs, tpu_aux = wkv7_pallas_tpu_forward_with_aux(
        *inputs,
        config=tpu_config,
        interpret=True,
    )
    expected = wkv7_reference(*inputs)

    assert gpu_aux[0].shape == inputs[0].shape
    assert gpu_aux[1].shape == (4, 1, 2, 4, 4)
    assert tpu_aux[1].shape == gpu_aux[1].shape
    assert jnp.allclose(gpu_outputs[1], expected[1])
    assert jnp.allclose(tpu_outputs[1], expected[1])
    assert jnp.allclose(gpu_aux[1][-1], expected[1])
    assert jnp.allclose(tpu_aux[1][-1], expected[1])


def test_auto_backend_is_pallas_first_and_never_selects_ffi(monkeypatch):
    monkeypatch.delenv("RWKV7M_WKV_BACKEND", raising=False)

    assert resolve_wkv_backend(platform="cpu") == "reference"
    assert resolve_wkv_backend(platform="tpu") == "pallas_tpu"
    assert resolve_wkv_backend(
        platform="gpu", device_kind="NVIDIA L40S"
    ) == "pallas_gpu_triton"
    assert resolve_wkv_backend(
        platform="gpu", device_kind="NVIDIA H100 80GB HBM3"
    ) == "pallas_gpu_mosaic"
    assert resolve_wkv_backend(
        platform="gpu", device_kind="AMD Instinct MI300X VF"
    ) == "pallas_gpu_triton_reference_vjp"


def test_amd_safe_backend_keeps_pallas_forward_and_reference_vjp():
    inputs = _inputs()

    def loss(backend, *values):
        y, final_state = wkv7(*values, backend, True)
        return jnp.sum(y.astype(jnp.float32) ** 2) + jnp.sum(
            final_state**2
        )

    expected = jax.grad(
        lambda *values: loss("reference", *values),
        argnums=tuple(range(7)),
    )(*inputs)
    actual = jax.grad(
        lambda *values: loss(
            "pallas_gpu_triton_reference_vjp", *values
        ),
        argnums=tuple(range(7)),
    )(*inputs)
    assert all(
        jnp.allclose(left, right, rtol=1e-5, atol=1e-6)
        for left, right in zip(actual, expected, strict=True)
    )


def test_backend_environment_override_is_validated(monkeypatch):
    monkeypatch.setenv("RWKV7M_WKV_BACKEND", "pallas_tpu")
    assert resolve_wkv_backend(platform="cpu") == "pallas_tpu"

    monkeypatch.setenv("RWKV7M_WKV_BACKEND", "not-a-backend")
    with pytest.raises(ValueError, match="unknown WKV backend"):
        resolve_wkv_backend(platform="cpu")


def test_ffi_requires_explicit_registration():
    inputs = _inputs()
    unregister_wkv_ffi_backend()

    with pytest.raises(RuntimeError, match="no FFI backend is registered"):
        wkv7(*inputs, "ffi")


def test_registered_ffi_satisfies_forward_and_backward_contract():
    inputs = _inputs()

    def forward(*values):
        return wkv7_reference(*values)

    def forward_with_aux(*values):
        return wkv7_reference(*values), ()

    def backward(*values):
        primal_values = values[:7]
        cotangents = values[7:9]
        _, pullback = jax.vjp(wkv7_reference, *primal_values)
        return pullback(cotangents)

    register_wkv_ffi_backend(
        WKVFFIBackend(
            forward=forward,
            forward_with_aux=forward_with_aux,
            backward=backward,
        )
    )
    try:
        actual = wkv7(*inputs, "ffi")
        expected = wkv7_reference(*inputs)
        assert all(
            jnp.allclose(left, right)
            for left, right in zip(actual, expected, strict=True)
        )

        actual_gradient = jax.grad(
            lambda *values: jnp.sum(wkv7(*values, "ffi")[0]),
            argnums=tuple(range(7)),
        )(*inputs)
        expected_gradient = jax.grad(
            lambda *values: jnp.sum(wkv7_reference(*values)[0]),
            argnums=tuple(range(7)),
        )(*inputs)
        assert all(
            jnp.allclose(left, right)
            for left, right in zip(
                actual_gradient, expected_gradient, strict=True
            )
        )
    finally:
        unregister_wkv_ffi_backend()
