import jax
import jax.numpy as jnp
from flax import linen as nn
from dataclasses import dataclass


def unit_norm(x, axis=-1, eps=1e-6):
    sq_sum = jnp.sum(x * x, axis=axis, keepdims=True)
    norm = jnp.sqrt(sq_sum + eps * eps)
    return x / norm


def bounded_tau(theta):
    return 2.0 * jax.nn.sigmoid(theta) - 1.0


def theta_from_tau(tau):
    p = (tau + 1.0) / 2.0
    p = jnp.clip(p, 1e-4, 1.0 - 1e-4)
    return jnp.log(p) - jnp.log1p(-p)


def trim_square(sim, tau, eps=1e-6):
    x = (sim - tau) / (1.0 - tau + eps)
    return jnp.square(jax.nn.relu(x))


def relevance_with_warmup(sim, tau, alpha, gamma=8.0):
    hard = trim_square(sim, tau)
    soft = jax.nn.sigmoid(gamma * (sim - tau))
    return (1.0 - alpha) * hard + alpha * soft


def tanh_norm(z, cap=1.0, eps=1e-6):
    sq_sum = jnp.sum(z * z, axis=-1, keepdims=True)
    norm = jnp.sqrt(sq_sum + eps * eps)
    scale = cap * jnp.tanh(norm / cap) / norm
    return scale * z


@dataclass
class ScreeningConfig:
    d_model: int = 512
    d_slot: int = 256
    d_k: int = 64
    d_v: int = 128
    n_slots: int = 16
    screened_layers: tuple[int, ...] = ()
    bank_ids: tuple[int, ...] = ()  # 0=short, 1=mid, 2=long; length M
    tau_init: float = 0.0
    tanh_norm_cap: float = 1.0
    lambda_screen_init: float = 0.01
    eps: float = 1e-6
    use_value_unit_norm: bool = True
    use_age_mask: bool = False
    use_bank_bias: bool = False
    use_write_screening: bool = False
    use_leaky_warmup: bool = False
    leaky_alpha: float = 0.0
    leaky_gamma: float = 8.0
    mu_short_max: float = 0.05
    mu_mid_max: float = 0.02
    mu_long_max: float = 0.005
    age_ref: float = 32.0
    age_sigma: float = 8.0


def make_mu_by_slot(cfg, mu_short_raw, mu_mid_raw, mu_long_raw):
    mu_max = jnp.array([cfg.mu_short_max, cfg.mu_mid_max, cfg.mu_long_max])
    mu_raw = jnp.array([mu_short_raw, mu_mid_raw, mu_long_raw])
    mu_by_bank = mu_max * jax.nn.sigmoid(mu_raw)
    bank_ids = jnp.array(cfg.bank_ids)
    return mu_by_bank[bank_ids]


class StateLevelScreening(nn.Module):
    config: ScreeningConfig

    def setup(self):
        cfg = self.config
        self.q_proj_r = nn.Dense(cfg.d_k, use_bias=False, name="q_proj_r")
        self.k_proj_r = nn.Dense(cfg.d_k, use_bias=False, name="k_proj_r")
        self.v_proj = nn.Dense(cfg.d_v, use_bias=False, name="v_proj")
        self.out_proj = nn.Dense(cfg.d_model, use_bias=False, name="out_proj")
        self.gate_proj = nn.Dense(cfg.d_model, name="gate_proj")
        self.delta_proj = nn.Dense(cfg.d_slot, name="delta_proj")
        if cfg.use_write_screening:
            self.q_proj_w = nn.Dense(cfg.d_k, use_bias=False, name="q_proj_w")
            self.k_proj_w = nn.Dense(cfg.d_k, use_bias=False, name="k_proj_w")

    @nn.compact
    def __call__(
        self,
        x_t,
        h_base,
        state,
        *,
        phase="read_only",
        deterministic=True,
        intervention=None,
    ):
        from .state import LayerScreenState

        cfg = self.config
        slots = state.slots
        ages = state.ages

        x_t_f32 = x_t.astype(jnp.float32)
        x_ln = nn.LayerNorm(dtype=jnp.float32, name="screen_ln")(x_t_f32)
        slots_f32 = slots.astype(jnp.float32)
        h_base_f32 = h_base.astype(jnp.float32)

        # Read branch
        q_r = self.q_proj_r(x_ln)
        k_r = self.k_proj_r(slots_f32)
        v = self.v_proj(slots_f32)

        q_r = unit_norm(q_r, eps=cfg.eps)
        k_r = unit_norm(k_r, eps=cfg.eps)
        if cfg.use_value_unit_norm:
            v = unit_norm(v, eps=cfg.eps)

        sim_r = jnp.einsum("bd,bmd->bm", q_r, k_r)

        tau_r_raw = self.param(
            "tau_r_raw",
            lambda rng, shape: theta_from_tau(cfg.tau_init),
            (),
        )
        tau_r = bounded_tau(tau_r_raw)

        if cfg.use_leaky_warmup:
            rel_r = relevance_with_warmup(sim_r, tau_r, cfg.leaky_alpha, cfg.leaky_gamma)
        else:
            rel_r = trim_square(sim_r, tau_r, eps=cfg.eps)

        if cfg.use_age_mask:
            rel_r = rel_r * compute_age_mask(ages, cfg)

        z = jnp.einsum("bm,bmd->bd", rel_r, v)
        u = tanh_norm(z, cap=cfg.tanh_norm_cap, eps=cfg.eps)

        gate = jax.nn.sigmoid(self.gate_proj(x_ln))
        read_out = self.out_proj(u)

        lambda_raw = self.param(
            "lambda_raw",
            nn.initializers.constant(jnp.log(jnp.exp(cfg.lambda_screen_init) - 1)),
            (),
        )
        lambda_screen = jax.nn.softplus(lambda_raw)

        h = h_base + lambda_screen * gate * read_out.astype(h_base.dtype)

        # Candidate update
        slot_embed = self.param(
            "slot_embed",
            nn.initializers.normal(0.02),
            (cfg.n_slots, cfg.d_slot),
        )
        slot_embed_b = jnp.broadcast_to(slot_embed[None, :, :], slots_f32.shape)
        x_rep = jnp.broadcast_to(x_ln[:, None, :], (x_ln.shape[0], cfg.n_slots, cfg.d_model))
        h_rep = jnp.broadcast_to(
            h.astype(jnp.float32)[:, None, :],
            (x_ln.shape[0], cfg.n_slots, cfg.d_model),
        )
        delta_in = jnp.concatenate([x_rep, h_rep, slot_embed_b], axis=-1)
        delta_s = jnp.tanh(self.delta_proj(delta_in))

        # Update rates
        bank_ids = jnp.array(cfg.bank_ids)
        mu_by_bank = self.param(
            "mu_by_bank_raw",
            nn.initializers.constant(0.0),
            (3,),
        )
        mu_max = jnp.array([cfg.mu_short_max, cfg.mu_mid_max, cfg.mu_long_max])
        mu_per_bank = mu_max * jax.nn.sigmoid(mu_by_bank)
        mu = mu_per_bank[bank_ids]  # [M]

        if phase == "read_only" or not cfg.use_write_screening:
            update_strength = mu[None, :, None]
            new_slots = slots_f32 + update_strength * (delta_s - slots_f32)
            new_ages = ages
            rel_w = jnp.zeros_like(rel_r)
            tau_w = jnp.zeros(())
        else:
            q_w_in = jnp.concatenate([x_ln, h_base_f32], axis=-1)
            q_w = self.q_proj_w(q_w_in)
            k_w = self.k_proj_w(slots_f32)
            q_w = unit_norm(q_w, eps=cfg.eps)
            k_w = unit_norm(k_w, eps=cfg.eps)
            sim_w = jnp.einsum("bd,bmd->bm", q_w, k_w)
            tau_w_raw = self.param(
                "tau_w_raw",
                lambda rng, shape: theta_from_tau(cfg.tau_init),
                (),
            )
            tau_w = bounded_tau(tau_w_raw)
            rel_w = trim_square(sim_w, tau_w, eps=cfg.eps)
            update_strength = mu[None, :, None] * rel_w[:, :, None]
            new_slots = slots_f32 + update_strength * (delta_s - slots_f32)
            new_ages = jnp.where(rel_w > 1e-3, 0.0, ages + 1.0)

        new_state = LayerScreenState(
            slots=new_slots.astype(slots.dtype),
            ages=new_ages,
            usage_ema=state.usage_ema,
        )

        # Active slot threshold
        eta_active = 1e-3

        stats = {
            "rel_read_mean": jnp.mean(rel_r),
            "rel_read_max": jnp.max(rel_r),
            "active_slots_mean": jnp.mean(jnp.sum(rel_r > eta_active, axis=-1)),
            "z_norm_mean": jnp.mean(jnp.linalg.norm(z, axis=-1)),
            "u_norm_mean": jnp.mean(jnp.linalg.norm(u, axis=-1)),
            "tau_r": tau_r,
            "lambda_screen": lambda_screen,
            "rel_write_mean": jnp.mean(rel_w),
            "tau_w": tau_w,
        }
        return h, new_state, stats


def compute_age_mask(ages, cfg):
    age_scores = (cfg.age_ref - ages) / (cfg.age_sigma + cfg.eps)
    return jax.nn.sigmoid(age_scores)
