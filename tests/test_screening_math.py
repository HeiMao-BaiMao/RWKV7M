import jax
import jax.numpy as jnp
import pytest
from rwkv7m.model.screening import (
    unit_norm,
    bounded_tau,
    theta_from_tau,
    trim_square,
    relevance_with_warmup,
    tanh_norm,
    update_rate_from_half_life,
    compute_slot_delta,
)


class TestUnitNorm:
    def test_nonzero_row_norm_is_one(self):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (8, 16))
        u = unit_norm(x)
        norms = jnp.linalg.norm(u, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-4)

    def test_zero_vector_handled(self):
        x = jnp.zeros((4, 8))
        u = unit_norm(x)
        assert jnp.allclose(u, 0.0)


class TestBoundedTau:
    def test_range(self):
        for theta in [-5.0, 0.0, 5.0]:
            tau = bounded_tau(jnp.array(theta))
            assert -1.0 < tau < 1.0

    def test_theta_zero_gives_tau_zero(self):
        tau = bounded_tau(jnp.array(0.0))
        assert jnp.allclose(tau, 0.0, atol=1e-6)


class TestThetaFromTau:
    def test_roundtrip(self):
        for tau_val in [-0.5, 0.0, 0.3, 0.8]:
            theta = theta_from_tau(tau_val)
            tau_back = bounded_tau(theta)
            assert jnp.allclose(tau_back, tau_val, atol=1e-6)


class TestTrimSquare:
    def test_at_threshold_is_zero(self):
        tau = 0.3
        sim = tau
        rel = trim_square(sim, tau)
        assert jnp.allclose(rel, 0.0, atol=1e-6)

    def test_below_threshold_is_zero(self):
        tau = 0.3
        sim = 0.1
        rel = trim_square(sim, tau)
        assert jnp.allclose(rel, 0.0)

    def test_at_one_is_one(self):
        tau = 0.0
        sim = 1.0
        rel = trim_square(sim, tau)
        assert jnp.allclose(rel, 1.0, atol=1e-6)

    def test_always_nonnegative(self):
        key = jax.random.PRNGKey(1)
        sim = jax.random.uniform(key, (100,), minval=-1.0, maxval=1.0)
        tau = 0.2
        rel = trim_square(sim, tau)
        assert jnp.all(rel >= 0.0)


class TestTanhNorm:
    def test_norm_bounded_by_cap(self):
        key = jax.random.PRNGKey(2)
        z = jax.random.normal(key, (8, 32)) * 5.0
        cap = 1.0
        u = tanh_norm(z, cap=cap)
        norms = jnp.linalg.norm(u, axis=-1)
        assert jnp.all(norms <= cap + 1e-3)

    def test_small_z_approx_identity(self):
        z = jnp.ones((4, 16)) * 0.01
        cap = 1.0
        u = tanh_norm(z, cap=cap)
        assert jnp.allclose(u, z, atol=1e-4)


def test_update_rate_matches_requested_half_life():
    for half_life in (32.0, 512.0, 4096.0):
        rate = update_rate_from_half_life(half_life)
        retained = jnp.power(1.0 - rate, half_life)
        assert jnp.allclose(retained, 0.5, rtol=1e-4, atol=1e-5)


def _concatenated_slot_delta(x, h, slot_embed, kernel, bias):
    batch_size = x.shape[0]
    n_slots = slot_embed.shape[0]
    x_rep = jnp.broadcast_to(x[:, None, :], (batch_size, n_slots, x.shape[-1]))
    h_rep = jnp.broadcast_to(h[:, None, :], (batch_size, n_slots, h.shape[-1]))
    slot_rep = jnp.broadcast_to(
        slot_embed[None, :, :],
        (batch_size, n_slots, slot_embed.shape[-1]),
    )
    joined = jnp.concatenate([x_rep, h_rep, slot_rep], axis=-1)
    return jnp.tanh(jnp.einsum("bmi,io->bmo", joined, kernel) + bias)


def _slot_delta_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.PRNGKey(23), 5)
    batch_size, d_model, n_slots, d_slot = 2, 8, 3, 4
    return (
        jax.random.normal(keys[0], (batch_size, d_model), dtype=dtype),
        jax.random.normal(keys[1], (batch_size, d_model), dtype=dtype),
        jax.random.normal(keys[2], (n_slots, d_slot), dtype=dtype),
        jax.random.normal(keys[3], (2 * d_model + d_slot, d_slot), dtype=dtype),
        jax.random.normal(keys[4], (d_slot,), dtype=dtype),
    )


def test_split_slot_delta_matches_concatenated_forward_float32():
    inputs = _slot_delta_inputs()
    expected = _concatenated_slot_delta(*inputs)
    actual = compute_slot_delta(*inputs)
    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_split_slot_delta_matches_concatenated_gradients_float32():
    inputs = _slot_delta_inputs()

    def reference_loss(*args):
        return jnp.sum(jnp.square(_concatenated_slot_delta(*args)))

    def split_loss(*args):
        return jnp.sum(jnp.square(compute_slot_delta(*args)))

    expected = jax.grad(reference_loss, argnums=(0, 1, 2, 3, 4))(*inputs)
    actual = jax.grad(split_loss, argnums=(0, 1, 2, 3, 4))(*inputs)
    for actual_grad, expected_grad in zip(actual, expected):
        assert jnp.allclose(actual_grad, expected_grad, rtol=3e-5, atol=3e-6)


def test_split_slot_delta_matches_concatenated_forward_bfloat16():
    inputs = _slot_delta_inputs(jnp.bfloat16)
    expected = _concatenated_slot_delta(*inputs).astype(jnp.float32)
    actual = compute_slot_delta(*inputs).astype(jnp.float32)
    assert jnp.allclose(actual, expected, rtol=1e-2, atol=1e-2)
