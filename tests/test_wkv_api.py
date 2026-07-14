import jax
import jax.numpy as jnp
import pytest

from rwkv7m.model.rwkv_core import wkv_step
from rwkv7m.model.wkv import wkv7, wkv7_reference


def _inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 7)
    vectors = tuple(
        (jax.random.normal(key, (3, 1, 2, 4)) * 0.1).astype(dtype)
        for key in keys[:6]
    )
    state = (
        jax.random.normal(keys[6], (1, 2, 4, 4)) * 0.1
    ).astype(jnp.float32)
    return (*vectors, state)


def test_wkv7_reference_matches_step_scan_and_dtype_contract():
    inputs = _inputs(jnp.bfloat16)
    *vectors, initial_state = inputs
    expected_state, expected_y = jax.lax.scan(
        lambda carry, values: wkv_step(carry, *values),
        initial_state,
        tuple(vectors),
    )

    actual_y, actual_state = wkv7_reference(*inputs)

    assert actual_y.dtype == jnp.bfloat16
    assert actual_state.dtype == jnp.float32
    assert jnp.allclose(actual_y, expected_y.astype(jnp.bfloat16))
    assert jnp.allclose(actual_state, expected_state)


def test_wkv7_custom_vjp_matches_reference_gradients():
    inputs = _inputs()

    def loss(fn, *values):
        y, final_state = fn(*values)
        return jnp.sum(jnp.square(y)) + 0.01 * jnp.sum(
            jnp.square(final_state)
        )

    argnums = tuple(range(len(inputs)))
    expected = jax.grad(
        lambda *values: loss(wkv7_reference, *values),
        argnums=argnums,
    )(*inputs)
    actual = jax.grad(
        lambda *values: loss(wkv7, *values),
        argnums=argnums,
    )(*inputs)

    assert all(
        jnp.allclose(left, right, rtol=1e-5, atol=1e-6)
        for left, right in zip(actual, expected, strict=True)
    )


def test_wkv7_rejects_invalid_shapes_and_dtypes():
    inputs = _inputs()
    *vectors, initial_state = inputs

    with pytest.raises(TypeError, match="initial_state must be float32"):
        wkv7_reference(*vectors, initial_state.astype(jnp.bfloat16))
    with pytest.raises(ValueError, match="k must have shape"):
        wkv7_reference(
            vectors[0],
            vectors[1],
            vectors[2][:, :, :, :-1],
            *vectors[3:],
            initial_state,
        )
    with pytest.raises(TypeError, match="bfloat16 or float32"):
        wkv7_reference(
            *(value.astype(jnp.float16) for value in vectors),
            initial_state,
        )


def test_wkv7_documents_reverse_mode_only_contract():
    inputs = _inputs()
    tangents = tuple(jnp.ones_like(value) for value in inputs)

    with pytest.raises(TypeError, match="forward-mode autodiff"):
        jax.jvp(wkv7, inputs, tangents)
