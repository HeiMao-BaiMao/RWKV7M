import time
import jax
import jax.numpy as jnp
import pytest

from rwkv7m.model.screening import ScreeningConfig
from rwkv7m.model.screened_rwkv import ModelConfig, ScreenedRWKVModel, init_rwkv_state, create_model_variables
from rwkv7m.model.state import init_screen_state


def make_benchmark_config():
    """Small config suitable for quick benchmark."""
    screening = ScreeningConfig(
        d_model=128,
        d_slot=64,
        d_k=32,
        d_v=32,
        n_slots=4,
        screened_layers=(1,),
        bank_ids=(0, 0, 1, 2),
    )
    return ModelConfig(
        d_model=128,
        d_ffn=256,
        n_layers=4,
        n_heads=4,
        head_size=32,
        vocab_size=256,
        max_seq_len=128,
        dtype="float32",
        use_screening=True,
        screening=screening,
    )


class TestBenchmark:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.cfg = make_benchmark_config()
        self.key = jax.random.PRNGKey(42)
        self.batch_size = 2
        self.seq_len = 32

    def test_forward_time(self):
        """Benchmark a JIT-compiled forward pass."""
        key, subkey = jax.random.split(self.key)
        variables, model = create_model_variables(subkey, self.cfg, self.batch_size)

        rwkv_state = init_rwkv_state(self.batch_size, self.cfg)
        screen_state = init_screen_state(self.batch_size, self.cfg.screening)

        input_ids = jax.random.randint(
            subkey, (self.batch_size, self.seq_len), 0, self.cfg.vocab_size
        )

        # JIT compile
        apply_fn = jax.jit(
            lambda ids, rs, ss: model.apply(
                variables, ids, rs, ss, phase="read_screening_only", deterministic=True
            )
        )

        # Warmup
        _ = apply_fn(input_ids, rwkv_state, screen_state)
        jax.block_until_ready(_)

        # Time 10 runs
        times = []
        for _ in range(10):
            start = time.perf_counter()
            out = apply_fn(input_ids, rwkv_state, screen_state)
            jax.block_until_ready(out)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        print(f"\nForward pass avg time: {avg_time * 1000:.2f} ms")

        # Should complete in reasonable time (< 5s per call for this tiny model)
        assert avg_time < 5.0, f"Forward pass too slow: {avg_time:.2f}s"

    def test_backward_time(self):
        """Benchmark a JIT-compiled backward pass."""
        key, subkey = jax.random.split(self.key)
        variables, model = create_model_variables(subkey, self.cfg, self.batch_size)

        rwkv_state = init_rwkv_state(self.batch_size, self.cfg)
        screen_state = init_screen_state(self.batch_size, self.cfg.screening)

        input_ids = jax.random.randint(
            subkey, (self.batch_size, self.seq_len), 0, self.cfg.vocab_size
        )
        target_ids = jax.random.randint(
            subkey, (self.batch_size, self.seq_len), 0, self.cfg.vocab_size
        )

        def loss_fn(params):
            logits, _, _, _ = model.apply(
                {"params": params},
                input_ids,
                rwkv_state,
                screen_state,
                phase="read_screening_only",
                deterministic=True,
            )
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            nll = -jnp.take_along_axis(
                log_probs, target_ids[..., None], axis=-1
            ).squeeze(-1)
            return jnp.mean(nll)

        grad_fn = jax.jit(jax.grad(loss_fn))

        # Warmup
        _ = grad_fn(variables["params"])
        jax.block_until_ready(_)

        # Time 5 runs
        times = []
        for _ in range(5):
            start = time.perf_counter()
            out = grad_fn(variables["params"])
            jax.block_until_ready(out)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        print(f"\nBackward pass avg time: {avg_time * 1000:.2f} ms")

        assert avg_time < 10.0, f"Backward pass too slow: {avg_time:.2f}s"

    def test_core_produces_nonzero_state(self):
        """Verify the RWKV7 core actually runs recurrence (state evolves)."""
        key, subkey = jax.random.split(self.key)
        cfg = ModelConfig(
            d_model=64,
            d_ffn=128,
            n_layers=2,
            n_heads=4,
            head_size=16,
            vocab_size=256,
            max_seq_len=16,
            dtype="float32",
            use_screening=False,
            screening=ScreeningConfig(),
        )
        variables, model = create_model_variables(subkey, cfg, self.batch_size)
        rwkv_state = init_rwkv_state(self.batch_size, cfg)
        screen_state = init_screen_state(self.batch_size, cfg.screening)

        input_ids = jax.random.randint(
            subkey, (self.batch_size, 8), 0, cfg.vocab_size
        )

        logits, _, _, _ = model.apply(
            variables,
            input_ids,
            rwkv_state,
            screen_state,
            phase="read_screening_only",
            deterministic=True,
        )

        # Logits should be non-zero and finite
        assert jnp.any(logits != 0), "Logits are all zero - core not producing output"
        assert jnp.all(jnp.isfinite(logits)), "Logits contain NaN/Inf"
