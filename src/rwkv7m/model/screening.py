import jax
import jax.numpy as jnp
from flax import linen as nn
from dataclasses import dataclass


_PHASE_ALIASES = {"read_only": "read_screening_only"}
_VALID_PHASES = {"read_screening_only", "read_write"}


def normalize_phase(phase: str) -> str:
    phase = _PHASE_ALIASES.get(phase, phase)
    if phase not in _VALID_PHASES:
        raise ValueError(
            f"Unknown phase {phase!r}. Expected one of {sorted(_VALID_PHASES)} "
            "or compatibility alias 'read_only'."
        )
    return phase


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


def update_rate_from_half_life(half_life_tokens):
    """Return the EMA update rate whose retained weight halves after H tokens."""
    half_life_tokens = jnp.asarray(half_life_tokens, dtype=jnp.float32)
    return -jnp.expm1(-jnp.log(2.0) / half_life_tokens)


def compute_slot_delta(x, h, slot_embed, kernel, bias):
    """Apply the concatenated delta projection without materializing it.

    The original projection is ``concat([x, h, slot_embed]) @ kernel + bias``
    for every slot. Splitting the kernel along its input axis avoids the
    ``[..., n_slots, 2 * d_model + d_slot]`` intermediate and avoids repeating
    the x/h projections once per slot.
    """
    d_model = x.shape[-1]
    d_slot = slot_embed.shape[-1]
    if h.shape[-1] != d_model:
        raise ValueError("x and h must have the same final dimension")
    if kernel.shape[0] != 2 * d_model + d_slot:
        raise ValueError("delta kernel input dimension does not match x/h/slot dimensions")

    x_kernel = kernel[:d_model, :]
    h_kernel = kernel[d_model : 2 * d_model, :]
    slot_kernel = kernel[2 * d_model :, :]

    x_term = jnp.einsum("...c,co->...o", x, x_kernel)
    h_term = jnp.einsum("...c,co->...o", h, h_kernel)
    slot_term = jnp.einsum("ms,so->mo", slot_embed, slot_kernel)
    return jnp.tanh(
        x_term[..., None, :]
        + h_term[..., None, :]
        + slot_term
        + bias
    )


@dataclass
class ScreeningConfig:
    d_model: int = 512
    d_slot: int = 256
    d_k: int = 64
    d_v: int = 128
    n_slots: int = 16
    screened_layers: tuple[int, ...] = ()
    bank_ids: tuple[int, ...] = ()
    tau_init: float = 0.0
    tanh_norm_cap: float = 1.0
    lambda_screen_init: float = 0.01
    eps: float = 1e-6
    use_value_unit_norm: bool = True
    use_age_mask: bool = False
    use_bank_bias: bool = False
    use_write_screening: bool = False
    write_rel_floor: float = 1e-3
    use_leaky_warmup: bool = False
    leaky_alpha: float = 0.0
    leaky_gamma: float = 8.0
    mu_short_max: float = 0.05
    mu_mid_max: float = 0.02
    mu_long_max: float = 0.005
    short_half_life_tokens: float | None = None
    mid_half_life_tokens: float | None = None
    long_half_life_tokens: float | None = None
    usage_ema_decay: float = 0.99
    age_ref: float = 32.0
    age_sigma: float = 8.0

    def __post_init__(self):
        self.screened_layers = tuple(self.screened_layers)
        self.bank_ids = tuple(self.bank_ids)
        if self.bank_ids and len(self.bank_ids) != self.n_slots:
            raise ValueError("bank_ids must have length n_slots when provided")
        invalid_banks = [bank_id for bank_id in self.bank_ids if bank_id not in (0, 1, 2)]
        if invalid_banks:
            raise ValueError("bank_ids values must be only 0, 1, or 2")
        if self.write_rel_floor < 0.0:
            raise ValueError("write_rel_floor must be non-negative")
        half_lives = (
            self.short_half_life_tokens,
            self.mid_half_life_tokens,
            self.long_half_life_tokens,
        )
        if any(value is not None and value <= 0.0 for value in half_lives):
            raise ValueError("memory half-life values must be positive when provided")
        if not 0.0 <= self.usage_ema_decay < 1.0:
            raise ValueError("usage_ema_decay must be in [0, 1)")


class StateLevelScreening(nn.Module):
    """Screening module that can process full sequences via lax.scan."""

    config: ScreeningConfig

    def setup(self):
        cfg = self.config
        self.q_proj_r = nn.Dense(cfg.d_k, use_bias=False, name="q_proj_r")
        self.k_proj_r = nn.Dense(cfg.d_k, use_bias=False, name="k_proj_r")
        self.v_proj = nn.Dense(cfg.d_v, use_bias=False, name="v_proj")
        self.out_proj = nn.Dense(cfg.d_model, use_bias=False, name="out_proj")
        self.gate_proj = nn.Dense(cfg.d_model, name="gate_proj")
        self.delta_proj = nn.Dense(cfg.d_slot, name="delta_proj")
        self.screen_ln = nn.LayerNorm(dtype=jnp.float32, name="screen_ln")
        if cfg.use_write_screening:
            self.q_proj_w = nn.Dense(cfg.d_k, use_bias=False, name="q_proj_w")
            self.k_proj_w = nn.Dense(cfg.d_k, use_bias=False, name="k_proj_w")

    def _init_params(self, cfg, x_seq, h_base_seq, state, phase):
        """During init, call all submodules once to create their params."""
        from .state import LayerScreenState
        x_t = x_seq[:, 0, :]
        h_t = h_base_seq[:, 0, :]
        x_ln = self.screen_ln(x_t.astype(jnp.float32))
        _ = self.q_proj_r(x_ln)
        slots = state.slots.astype(jnp.float32)
        _ = self.k_proj_r(slots)
        _ = self.v_proj(slots)
        _ = self.gate_proj(x_ln)
        _ = self.out_proj(jnp.zeros((x_ln.shape[0], cfg.d_v)))
        delta_in = jnp.concatenate([x_ln, h_t, slots[:, 0, :]], axis=-1)
        _ = self.delta_proj(delta_in)
        _ = self.param("tau_r_raw", lambda rng, shape: theta_from_tau(cfg.tau_init), ())
        _ = self.param("lambda_raw", nn.initializers.constant(jnp.log(jnp.exp(cfg.lambda_screen_init) - 1)), ())
        _ = self.param("slot_embed", nn.initializers.normal(0.02), (cfg.n_slots, cfg.d_slot))
        _ = self.param("mu_by_bank_raw", nn.initializers.constant(0.0), (3,))
        if cfg.use_write_screening:
            q_w_in = jnp.concatenate([x_ln, h_t.astype(jnp.float32)], axis=-1)
            _ = self.q_proj_w(q_w_in)
            _ = self.k_proj_w(slots)
            _ = self.param("tau_w_raw", lambda rng, shape: theta_from_tau(cfg.tau_init), ())
        return h_base_seq, state, {}

    @nn.compact
    def __call__(
        self,
        x_seq: jnp.ndarray,        # [B, T, d]
        h_base_seq: jnp.ndarray,   # [B, T, d]
        state,
        *,
        phase="read_screening_only",
        deterministic=True,
    ):
        from .state import LayerScreenState

        cfg = self.config
        phase = normalize_phase(phase)
        _, T, _ = x_seq.shape

        # During init, create all params via dummy call
        if self.is_initializing():
            return self._init_params(cfg, x_seq, h_base_seq, state, phase)

        slots = state.slots.astype(jnp.float32)
        ages = state.ages
        usage_ema = state.usage_ema.astype(jnp.float32)

        # --- Pre-compute projections and params for all time steps ---
        x_ln_seq = self.screen_ln(x_seq.astype(jnp.float32))  # [B, T, C]

        # Read query for all time steps
        q_r_seq = self.q_proj_r(x_ln_seq)  # [B, T, d_k]
        q_r_seq = unit_norm(q_r_seq, eps=cfg.eps)

        # Extract projection weights
        p = self.variables["params"]
        k_w = p["k_proj_r"]["kernel"]     # [d_s, d_k]
        v_w = p["v_proj"]["kernel"]       # [d_s, d_v]
        gate_w = p["gate_proj"]["kernel"]  # [C, C]
        gate_b = p["gate_proj"]["bias"]    # [C]
        out_w = p["out_proj"]["kernel"]   # [d_v, C]
        delta_w = p["delta_proj"]["kernel"]  # [2*C + d_s, d_slot]
        delta_b = p["delta_proj"]["bias"]    # [d_slot]

        # Scalar params
        tau_r_raw = p["tau_r_raw"]
        tau_r = bounded_tau(tau_r_raw)
        lambda_raw = p["lambda_raw"]
        lambda_screen = jax.nn.softplus(lambda_raw)

        slot_embed = p["slot_embed"]  # [M, d_s]
        mu = self._compute_mu(p, cfg)

        # Pre-compute write query if needed
        write_enabled = phase == "read_write" and cfg.use_write_screening
        if write_enabled:
            q_w_w = p["q_proj_w"]["kernel"]  # [2*C, d_k]
            k_w_w = p["k_proj_w"]["kernel"]  # [d_s, d_k]
            tau_w_raw = p["tau_w_raw"]
            tau_w = bounded_tau(tau_w_raw)
            q_w_in = jnp.concatenate([x_ln_seq, h_base_seq.astype(jnp.float32)], axis=-1)
            q_w_seq = jnp.einsum("btc,ck->btk", q_w_in, q_w_w)
            q_w_seq = unit_norm(q_w_seq, eps=cfg.eps)
        else:
            q_w_seq = None
            tau_w = jnp.zeros(())

        def step(carry, t):
            """Pure function - no Flax modules called here."""
            slots_t, ages_t, usage_t = carry

            # Read branch
            k_r = jnp.einsum("bms,sk->bmk", slots_t, k_w)
            v = jnp.einsum("bms,sv->bmv", slots_t, v_w)
            k_r = unit_norm(k_r, eps=cfg.eps)
            if cfg.use_value_unit_norm:
                v = unit_norm(v, eps=cfg.eps)

            sim_r = jnp.einsum("bk,bmk->bm", q_r_seq[:, t, :], k_r)

            if cfg.use_leaky_warmup:
                hard = trim_square(sim_r, tau_r)
                soft = jax.nn.sigmoid(cfg.leaky_gamma * (sim_r - tau_r))
                rel_r = (1.0 - cfg.leaky_alpha) * hard + cfg.leaky_alpha * soft
            else:
                rel_r = trim_square(sim_r, tau_r, eps=cfg.eps)

            if cfg.use_age_mask:
                age_scores = (cfg.age_ref - ages_t) / (cfg.age_sigma + cfg.eps)
                rel_r = rel_r * jax.nn.sigmoid(age_scores)

            z = jnp.einsum("bm,bmv->bv", rel_r, v)
            u = tanh_norm(z, cap=cfg.tanh_norm_cap, eps=cfg.eps)

            gate = jax.nn.sigmoid(jnp.einsum("bc,cg->bg", x_ln_seq[:, t, :], gate_w) + gate_b)
            read_out = jnp.einsum("bv,vc->bc", u, out_w)
            h_t = h_base_seq[:, t, :] + lambda_screen * gate * read_out.astype(h_base_seq.dtype)

            # Slot update. Compute the split projection inside the time scan so
            # no [B, T, M, d_slot] delta activation is retained.
            delta_s_t = compute_slot_delta(
                x_ln_seq[:, t, :],
                h_base_seq[:, t, :].astype(jnp.float32),
                slot_embed,
                delta_w,
                delta_b,
            )
            if not write_enabled:
                update_strength = mu[None, :, None]
                new_slots = slots_t + update_strength * (delta_s_t - slots_t)
                new_ages = ages_t
                rel_w = jnp.zeros_like(rel_r)
                rel_w_effective = jnp.ones_like(rel_r)
            else:
                slots_for_write_key = slots_t + slot_embed[None, :, :]
                k_w_t = jnp.einsum("bms,sk->bmk", slots_for_write_key, k_w_w)
                k_w_t = unit_norm(k_w_t, eps=cfg.eps)
                sim_w = jnp.einsum("bk,bmk->bm", q_w_seq[:, t, :], k_w_t)
                rel_w = trim_square(sim_w, tau_w, eps=cfg.eps)
                rel_w_effective = jnp.maximum(rel_w, cfg.write_rel_floor)
                update_strength = mu[None, :, None] * rel_w_effective[:, :, None]
                new_slots = slots_t + update_strength * (delta_s_t - slots_t)
                new_ages = jnp.where(rel_w > 1e-3, 0.0, ages_t + 1.0)

            update_delta = new_slots - slots_t
            activity = jnp.maximum(rel_r, rel_w if write_enabled else rel_r)
            new_usage = (
                cfg.usage_ema_decay * usage_t
                + (1.0 - cfg.usage_ema_decay) * activity
            )

            # Stats
            eta_active = 1e-3
            stats_t = {
                "rel_read_mean": jnp.mean(rel_r),
                "rel_read_max": jnp.max(rel_r),
                "active_slots_mean": jnp.mean(jnp.sum(rel_r > eta_active, axis=-1)),
                "z_norm_mean": jnp.mean(jnp.linalg.norm(z, axis=-1)),
                "u_norm_mean": jnp.mean(jnp.linalg.norm(u, axis=-1)),
                "tau_r": tau_r,
                "lambda_screen": lambda_screen,
                "rel_write_mean": jnp.mean(rel_w),
                "rel_write_effective_mean": jnp.mean(
                    jnp.maximum(rel_w, cfg.write_rel_floor)
                ) if write_enabled else jnp.mean(rel_w),
                "slot_update_norm_mean": jnp.mean(
                    jnp.linalg.norm(update_delta, axis=-1)
                ),
                "slot_usage_ema_mean": jnp.mean(new_usage),
                "tau_w": tau_w,
            }

            return (new_slots, new_ages, new_usage), (h_t, stats_t)

        # lax.scan over time
        (final_slots, final_ages, final_usage), (h_seq, stats_seq) = jax.lax.scan(
            step,
            (slots, ages, usage_ema),
            jnp.arange(T),
        )

        h = jnp.swapaxes(h_seq, 0, 1)

        new_state = LayerScreenState(
            slots=final_slots.astype(state.slots.dtype),
            ages=final_ages,
            usage_ema=final_usage.astype(state.usage_ema.dtype),
        )

        # Aggregate stats over time (scan stacks dict values into arrays)
        agg_stats = {k: jnp.mean(v) for k, v in stats_seq.items()}

        return h, new_state, agg_stats

    def _compute_mu(self, p, cfg):
        mu_by_bank_raw = p["mu_by_bank_raw"]
        mu_max = jnp.array([cfg.mu_short_max, cfg.mu_mid_max, cfg.mu_long_max])
        legacy_mu = mu_max * jax.nn.sigmoid(mu_by_bank_raw)
        half_lives = (
            cfg.short_half_life_tokens,
            cfg.mid_half_life_tokens,
            cfg.long_half_life_tokens,
        )
        mu_per_bank = jnp.stack(
            [
                legacy_mu[index]
                if half_life is None
                else update_rate_from_half_life(half_life)
                for index, half_life in enumerate(half_lives)
            ]
        )
        bank_ids = jnp.array(cfg.bank_ids)
        return mu_per_bank[bank_ids]  # [M]


def compute_age_mask(ages, cfg):
    age_scores = (cfg.age_ref - ages) / (cfg.age_sigma + cfg.eps)
    return jax.nn.sigmoid(age_scores)
