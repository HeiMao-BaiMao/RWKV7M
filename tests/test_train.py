import jax
import jax.numpy as jnp
import pytest

from rwkv7m.model.screening import ScreeningConfig
from rwkv7m.model.screened_rwkv import ModelConfig
from rwkv7m.train.train_loop import run_toy_training, generate_toy_batch
from rwkv7m.train.train_state import rwkv_warmup_cosine_schedule


def make_train_config():
    screening = ScreeningConfig(
        d_model=64,
        d_slot=32,
        d_k=16,
        d_v=16,
        n_slots=4,
        screened_layers=(1,),
        bank_ids=(0, 0, 1, 2),
        tau_init=0.0,
        lambda_screen_init=0.01,
    )
    return ModelConfig(
        d_model=64,
        d_ffn=128,
        n_layers=3,
        n_heads=4,
        head_size=16,
        vocab_size=256,
        max_seq_len=16,
        dtype="float32",
        use_screening=True,
        screening=screening,
    )


class TestTraining:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.cfg = make_train_config()
        self.key = jax.random.PRNGKey(42)

    def test_toy_batch_generation(self):
        key, subkey = jax.random.split(self.key)
        batch = generate_toy_batch(subkey, batch_size=2, seq_len=8, vocab_size=256)
        assert batch["input_ids"].shape == (2, 8)
        assert batch["target_ids"].shape == (2, 8)
        assert batch["mask"].shape == (2, 8)

    def test_toy_training_no_nan(self):
        """100 steps of toy training must not produce NaN."""
        key, subkey = jax.random.split(self.key)
        losses, train_state = run_toy_training(
            subkey,
            self.cfg,
            batch_size=2,
            seq_len=8,
            num_steps=100,
            print_every=25,
        )
        assert len(losses) == 100
        for loss in losses:
            assert jnp.isfinite(loss), f"Loss is not finite: {loss}"

    def test_loss_decreases_or_stable(self):
        """Loss should at least not diverge to infinity."""
        key, subkey = jax.random.split(self.key)
        losses, train_state = run_toy_training(
            subkey,
            self.cfg,
            batch_size=2,
            seq_len=8,
            num_steps=100,
            print_every=25,
        )
        first_10_avg = sum(losses[:10]) / 10
        last_10_avg = sum(losses[-10:]) / 10
        # Should not explode
        assert last_10_avg < first_10_avg * 10, "Loss exploded"

    def test_jit_train_step(self):
        """Train step must be JIT-compilable."""
        from rwkv7m.model.screened_rwkv import (
            ScreenedRWKVModel,
            init_rwkv_state,
            create_model_variables,
        )
        from rwkv7m.model.state import init_screen_state
        from rwkv7m.train.train_loop import build_train_state
        from rwkv7m.train.train_step import train_step

        key, subkey = jax.random.split(self.key)
        model = ScreenedRWKVModel(config=self.cfg)
        batch_size = 2
        rwkv_state = init_rwkv_state(batch_size, self.cfg)
        screen_state = init_screen_state(batch_size, self.cfg.screening)

        key, subkey = jax.random.split(key)
        variables, _ = create_model_variables(subkey, self.cfg, batch_size)

        key, subkey = jax.random.split(key)
        train_state = build_train_state(
            subkey, model, variables, self.cfg, total_steps=20
        )

        batch = generate_toy_batch(subkey, batch_size, 8, self.cfg.vocab_size)

        key, subkey = jax.random.split(key)
        rwkv_state = init_rwkv_state(batch_size, self.cfg)
        screen_state = init_screen_state(batch_size, self.cfg.screening)

        train_state, rwkv_state, screen_state, metrics = train_step(
            train_state, batch, rwkv_state, screen_state, phase="read_screening_only"
        )

        assert jnp.isfinite(metrics["loss"])


def test_rwkv_warmup_cosine_schedule_matches_reference_points():
    schedule = rwkv_warmup_cosine_schedule(
        lr_init=1e-3,
        lr_final=1e-5,
        warmup_steps=10,
        total_steps=100,
    )

    assert float(schedule(0)) == pytest.approx(1e-5)
    assert float(schedule(10)) == pytest.approx(1e-3)
    assert float(schedule(100)) == pytest.approx(1e-5)
