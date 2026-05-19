import jax
import jax.numpy as jnp
import pytest
from src.model.screening import (
    unit_norm,
    bounded_tau,
    theta_from_tau,
    trim_square,
    relevance_with_warmup,
    tanh_norm,
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
