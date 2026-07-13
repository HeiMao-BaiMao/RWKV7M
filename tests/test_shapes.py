import jax
import jax.numpy as jnp
import pytest
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict
from rwkv7m.model.screening import ScreeningConfig
from rwkv7m.model.screening import StateLevelScreening, theta_from_tau
from rwkv7m.model.state import LayerScreenState
from rwkv7m.model.state import init_screen_state, tuple_set
from rwkv7m.model.screened_rwkv import (
    ModelConfig,
    ScreenedRWKVModel,
    init_rwkv_state,
    create_model_variables,
)


def make_tiny_config():
    screening = ScreeningConfig(
        d_model=64,
        d_slot=32,
        d_k=16,
        d_v=16,
        n_slots=4,
        screened_layers=(1,),
        bank_ids=(0, 0, 1, 2),
    )
    return ModelConfig(
        d_model=64,
        d_ffn=128,
        n_layers=3,
        n_heads=4,
        head_size=16,
        vocab_size=256,
        max_seq_len=16,
        use_screening=True,
        screening=screening,
    )


class TestShapes:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.cfg = make_tiny_config()
        self.batch_size = 2
        self.seq_len = 8
        self.key = jax.random.PRNGKey(42)

    def test_logits_shape(self):
        key, subkey = jax.random.split(self.key)
        variables, model = create_model_variables(subkey, self.cfg, self.batch_size)
        rwkv_state = init_rwkv_state(self.batch_size, self.cfg)
        screen_state = init_screen_state(self.batch_size, self.cfg.screening)

        input_ids = jnp.zeros((self.batch_size, self.seq_len), dtype=jnp.int32)

        logits, _, _, stats = model.apply(
            variables,
            input_ids,
            rwkv_state,
            screen_state,
            phase="read_screening_only",
            deterministic=True,
        )
        assert logits.shape == (self.batch_size, self.seq_len, self.cfg.vocab_size)

    def test_screen_state_shapes(self):
        screen_state = init_screen_state(self.batch_size, self.cfg.screening)
        assert len(screen_state.layers) == len(self.cfg.screening.screened_layers)
        for l_state in screen_state.layers:
            assert l_state.slots.shape == (self.batch_size, self.cfg.screening.n_slots, self.cfg.screening.d_slot)
            assert l_state.ages.shape == (self.batch_size, self.cfg.screening.n_slots)

    def test_channel_mix_uses_configured_d_ffn(self):
        key, subkey = jax.random.split(self.key)
        variables, _ = create_model_variables(subkey, self.cfg, self.batch_size)
        flat_params = flatten_dict(variables["params"], sep="/")

        ffn_key_kernels = [
            value
            for name, value in flat_params.items()
            if name.endswith("/ffn/key/kernel")
        ]
        ffn_value_kernels = [
            value
            for name, value in flat_params.items()
            if name.endswith("/ffn/value/kernel")
        ]
        assert len(ffn_key_kernels) == self.cfg.n_layers
        assert len(ffn_value_kernels) == self.cfg.n_layers
        for kernel in ffn_key_kernels:
            assert kernel.shape == (self.cfg.d_model, self.cfg.d_ffn)
        for kernel in ffn_value_kernels:
            assert kernel.shape == (self.cfg.d_ffn, self.cfg.d_model)

    def test_state_updates(self):
        key, subkey = jax.random.split(self.key)
        variables, model = create_model_variables(subkey, self.cfg, self.batch_size)
        rwkv_state = init_rwkv_state(self.batch_size, self.cfg)
        screen_state = init_screen_state(self.batch_size, self.cfg.screening)

        input_ids = jnp.zeros((self.batch_size, 4), dtype=jnp.int32)

        _, new_rwkv_state, new_screen_state, stats = model.apply(
            variables,
            input_ids,
            rwkv_state,
            screen_state,
            phase="read_screening_only",
            deterministic=True,
        )
        assert new_screen_state is not None
        assert len(new_screen_state.layers) == len(screen_state.layers)
        for old_l, new_l in zip(screen_state.layers, new_screen_state.layers):
            assert new_l.slots.shape == old_l.slots.shape
            assert new_l.ages.shape == old_l.ages.shape

    def test_scan_consistency(self):
        """T=1 decode repeated T times should match T-token prefill logits."""
        key, subkey = jax.random.split(self.key)
        variables, model = create_model_variables(subkey, self.cfg, self.batch_size)

        T = 4
        input_ids = jax.random.randint(subkey, (self.batch_size, T), 0, self.cfg.vocab_size)

        # Prefill all T tokens at once
        rwkv_state = init_rwkv_state(self.batch_size, self.cfg)
        screen_state = init_screen_state(self.batch_size, self.cfg.screening)
        logits_prefill, _, _, _ = model.apply(
            variables, input_ids, rwkv_state, screen_state,
            phase="read_screening_only", deterministic=True,
        )

        # Decode one step at a time
        rwkv_state = init_rwkv_state(self.batch_size, self.cfg)
        screen_state = init_screen_state(self.batch_size, self.cfg.screening)
        logits_steps = []
        for t in range(T):
            tok = input_ids[:, t : t + 1]
            logits_t, rwkv_state, screen_state, _ = model.apply(
                variables, tok, rwkv_state, screen_state,
                phase="read_screening_only", deterministic=True,
            )
            logits_steps.append(logits_t)

        logits_stepwise = jnp.concatenate(logits_steps, axis=1)
        max_diff = jnp.max(jnp.abs(logits_prefill - logits_stepwise))
        assert max_diff < 1e-4, f"Scan consistency failed: max_diff={max_diff}"

    def test_all_irrelevant_yields_zero_readout(self):
        """A zero query below tau must produce no screening residual."""
        cfg = ScreeningConfig(
            d_model=8,
            d_slot=4,
            d_k=4,
            d_v=4,
            n_slots=2,
            screened_layers=(0,),
            bank_ids=(0, 2),
        )
        module = StateLevelScreening(cfg)
        x = jnp.ones((1, 3, 8), dtype=jnp.float32)
        h_base = jax.random.normal(jax.random.PRNGKey(8), (1, 3, 8))
        state = LayerScreenState(
            slots=jax.random.normal(jax.random.PRNGKey(9), (1, 2, 4)),
            ages=jnp.zeros((1, 2)),
            usage_ema=jnp.zeros((1, 2)),
        )
        variables = module.init(
            jax.random.PRNGKey(10),
            x,
            h_base,
            state,
            phase="read_screening_only",
        )
        mutable = unfreeze(variables)
        mutable["params"]["q_proj_r"]["kernel"] = jnp.zeros_like(
            mutable["params"]["q_proj_r"]["kernel"]
        )
        mutable["params"]["tau_r_raw"] = theta_from_tau(0.5)
        h, _, stats = module.apply(
            freeze(mutable),
            x,
            h_base,
            state,
            phase="read_screening_only",
        )
        assert jnp.allclose(h, h_base)
        assert jnp.allclose(stats["rel_read_mean"], 0.0)
        assert jnp.allclose(stats["u_norm_mean"], 0.0)
