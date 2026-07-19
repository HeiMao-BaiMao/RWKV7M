from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtri
from flax import linen as nn


_PHASE_ALIASES = {"read_only": "read_screening_only"}
_VALID_PHASES = {"read_screening_only", "read_write"}
_VALID_WRITE_MODES = {
    "disabled",
    "legacy_unconditional",
    "legacy_threshold",
    "competitive_novel",
}
_VALID_GATE_SPACES = {"model", "value"}
_VALID_GATE_ACTIVATIONS = {"sigmoid", "tanh_silu"}
_VALID_SEMANTICS_VERSIONS = {
    "screening-v4-legacy",
    "screening-v4-competitive",
    "screening-v5-core",
    "screening-v5-retention",
}
_VALID_CAPACITY_CALIBRATION = {"fixed", "analytic"}
_VALID_EDIT_MODES = {"tied", "capacity_conserving", "free_edit"}


def normalize_phase(phase: str) -> str:
    phase = _PHASE_ALIASES.get(phase, phase)
    if phase not in _VALID_PHASES:
        raise ValueError(
            f"Unknown phase {phase!r}. Expected one of {sorted(_VALID_PHASES)} "
            "or compatibility alias 'read_only'."
        )
    return phase


def normalize_write_mode(write_mode: str) -> str:
    if write_mode not in _VALID_WRITE_MODES:
        raise ValueError(
            f"Unknown write mode {write_mode!r}. Expected one of "
            f"{sorted(_VALID_WRITE_MODES)}."
        )
    return write_mode


def normalize_semantics_version(semantics_version: str) -> str:
    if semantics_version not in _VALID_SEMANTICS_VERSIONS:
        raise ValueError(
            f"Unknown screening semantics version {semantics_version!r}. "
            f"Expected one of {sorted(_VALID_SEMANTICS_VERSIONS)}."
        )
    return semantics_version


def resolve_semantics_version(config) -> str:
    """Resolve missing versions without silently upgrading v4 artifacts."""

    if config.semantics_version is not None:
        return normalize_semantics_version(config.semantics_version)
    if config.write_mode == "competitive_novel":
        return "screening-v4-competitive"
    return "screening-v4-legacy"


def is_v5_semantics(config) -> bool:
    return resolve_semantics_version(config) in {
        "screening-v5-core",
        "screening-v5-retention",
    }


def resolve_write_mode(config, phase: str) -> str:
    """Resolve old phase/config pairs without changing their semantics."""

    phase = normalize_phase(phase)
    if phase != "read_write":
        return "legacy_unconditional"
    if config.write_mode is not None:
        return normalize_write_mode(config.write_mode)
    if config.use_write_screening:
        return "legacy_threshold"
    return "legacy_unconditional"


def write_mode_uses_write_projection(config) -> bool:
    return config.use_write_screening or config.write_mode in {
        "legacy_threshold",
        "competitive_novel",
    }


def apply_screening_gate(logits, activation: str):
    if activation == "sigmoid":
        return jax.nn.sigmoid(logits)
    if activation == "tanh_silu":
        return jnp.tanh(jax.nn.silu(logits))
    raise ValueError(f"unsupported screening gate activation: {activation!r}")


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


def capacity_calibrated_similarity_threshold(
    test_count,
    key_dimension,
    false_positive_rate,
    *,
    tau_min,
    tau_max,
):
    """High-dimensional null-similarity threshold from the v5 contract.

    This is the paper's analytic initialization approximation. It is a
    calibration policy, not a guarantee about learned key distributions.
    """

    tests = jnp.maximum(
        jnp.asarray(test_count, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    dimension = jnp.asarray(key_dimension, dtype=jnp.float32)
    delta = jnp.asarray(false_positive_rate, dtype=jnp.float32)
    # Approximate the paper's null-similarity CDF instead of using the much
    # looser exponential tail bound.  The old sqrt(2 log(N/delta) / d)
    # expression reaches 0.946 for the tracked 4-tile/16-slot read shape and
    # starves virtually every learned read.  A unit-vector dot product has
    # variance 1/d, so the Gaussian null approximation gives the family-wise
    # quantile below.  It remains an initialization policy; empirical
    # false-read calibration is still required for learned key distributions.
    family_cdf = jnp.exp(jnp.log1p(-delta) / tests)
    family_cdf = jnp.clip(family_cdf, 1e-6, 1.0 - 1e-6)
    threshold = ndtri(family_cdf) / jnp.sqrt(dimension)
    return jnp.clip(threshold, tau_min, tau_max)


def smooth_trim_square(sim, tau, temperature, eps=1e-6):
    """Smooth training surrogate with the same normalized relevance scale."""

    x = (sim - tau) / (1.0 - tau + eps)
    positive = temperature * jax.nn.softplus(x / temperature)
    return jnp.square(jnp.clip(positive, 0.0, 1.0))


def bounded_non_amplifying_aggregate(z_sum):
    """Clip only aggregate vectors whose norm exceeds one."""

    energy = jnp.sum(
        z_sum.astype(jnp.float32) ** 2,
        axis=-1,
        keepdims=True,
    )
    denominator = jnp.sqrt(1.0 + jax.nn.relu(energy - 1.0))
    return z_sum / denominator, energy[..., 0]


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


def compute_factorized_slot_delta(
    x,
    h,
    slot_embed,
    x_kernel,
    h_kernel,
    slot_kernel,
    out_kernel,
    out_bias,
):
    """Compute a low-rank slot candidate without per-slot Dense calls."""

    x_latent = jnp.einsum("...c,cr->...r", x, x_kernel)
    h_latent = jnp.einsum("...c,cr->...r", h, h_kernel)
    slot_latent = jnp.einsum("ms,sr->mr", slot_embed, slot_kernel)
    latent = jax.nn.silu(
        x_latent[..., None, :] + h_latent[..., None, :] + slot_latent
    )
    return jnp.tanh(
        jnp.einsum("...mr,rs->...ms", latent, out_kernel) + out_bias
    )


def _masked_softmax(logits, mask, *, axis=-1, eps=1e-6):
    mask = jnp.broadcast_to(mask, logits.shape)
    masked = jnp.where(mask, logits, jnp.asarray(-1e30, logits.dtype))
    maximum = jnp.max(masked, axis=axis, keepdims=True)
    exponent = jnp.where(mask, jnp.exp(masked - maximum), 0.0)
    return exponent / (jnp.sum(exponent, axis=axis, keepdims=True) + eps)


def competitive_write_routing(
    eligibility,
    ages,
    usage,
    admission,
    bank_logits,
    bank_ids,
    *,
    route_power,
    novelty_threshold,
    novelty_temperature,
    allocation_temperature,
    bank_route_temperature,
    allocation_age_weight,
    allocation_usage_weight,
    hard_admission,
    admission_threshold,
    eps,
):
    """Return confidence-preserving matched and sparse novel routes.

    The leading dimensions are arbitrary; the final eligibility/state axis is
    the slot axis and the final bank-logit axis has size three.
    """

    eligibility = eligibility.astype(jnp.float32)
    ages = ages.astype(jnp.float32)
    usage = usage.astype(jnp.float32)
    admission = admission.astype(jnp.float32)
    bank_logits = bank_logits.astype(jnp.float32)
    bank_ids = jnp.asarray(bank_ids, dtype=jnp.int32)
    slot_count = eligibility.shape[-1]

    powered = jnp.power(eligibility, route_power)
    powered_sum = jnp.sum(powered, axis=-1, keepdims=True)
    matched_distribution = powered / jnp.where(
        powered_sum > 0.0, powered_sum, 1.0
    )
    confidence = jnp.max(eligibility, axis=-1)
    raw_matched_route = confidence[..., None] * matched_distribution
    is_novel = confidence < novelty_threshold
    novel_soft = jax.nn.sigmoid(
        (novelty_threshold - confidence) / novelty_temperature
    )
    novel_hard = is_novel.astype(jnp.float32)
    novel_gate = novel_soft + jax.lax.stop_gradient(
        novel_hard - novel_soft
    )

    admission_soft = admission
    if hard_admission:
        admission_hard = (
            admission_soft >= admission_threshold
        ).astype(jnp.float32)
        admission = admission_soft + jax.lax.stop_gradient(
            admission_hard - admission_soft
        )

    # GPU Triton kernels may pad the public three-bank vector to a power-of-two
    # width. Padded banks have no matching slot IDs and are therefore masked
    # out without changing the public three-bank routing semantics.
    bank_count = bank_logits.shape[-1]
    if bank_count < 3:
        raise ValueError("bank_logits must contain at least three banks")

    # Mosaic TPU requires ``iota`` itself to have an integer/index dtype.
    # Convert only after materializing the tie-break indices.
    bank_index = jnp.arange(bank_count, dtype=jnp.int32).astype(jnp.float32)
    bank_membership = (
        bank_ids[:, None]
        == jnp.arange(bank_count, dtype=jnp.int32)[None, :]
    )
    available_banks = jnp.sum(
        bank_membership.astype(jnp.int32), axis=0
    ) > 0
    bank_scores = (
        bank_logits / bank_route_temperature - 1e-6 * bank_index
    )
    bank_soft = _masked_softmax(
        bank_scores,
        available_banks,
        eps=eps,
    )
    bank_max = jnp.max(
        jnp.where(available_banks, bank_scores, -1e30),
        axis=-1,
        keepdims=True,
    )
    bank_hard = (
        (bank_scores == bank_max) & available_banks
    ).astype(jnp.float32)
    bank_route = bank_soft + jax.lax.stop_gradient(bank_hard - bank_soft)

    slot_index = jnp.arange(slot_count, dtype=jnp.int32).astype(jnp.float32)
    victim_route = jnp.zeros_like(eligibility, dtype=jnp.float32)
    for bank_id in range(bank_count):
        mask = bank_ids == bank_id
        mask_broadcast = jnp.broadcast_to(mask, eligibility.shape)
        age_min = jnp.min(
            jnp.where(mask_broadcast, ages, 1e30),
            axis=-1,
            keepdims=True,
        )
        age_max = jnp.max(
            jnp.where(mask_broadcast, ages, -1e30),
            axis=-1,
            keepdims=True,
        )
        normalized_age = (ages - age_min) / (age_max - age_min + eps)
        slot_scores = (
            allocation_age_weight * normalized_age
            - allocation_usage_weight * usage
        )
        slot_scores = (
            slot_scores / allocation_temperature - 1e-6 * slot_index
        )
        slot_soft = _masked_softmax(
            slot_scores,
            mask_broadcast,
            eps=eps,
        )
        slot_max = jnp.max(
            jnp.where(mask_broadcast, slot_scores, -1e30),
            axis=-1,
            keepdims=True,
        )
        slot_hard = (
            (slot_scores == slot_max) & mask_broadcast
        ).astype(jnp.float32)
        slot_route = slot_soft + jax.lax.stop_gradient(
            slot_hard - slot_soft
        )
        bank_selector = (
            jnp.arange(bank_count, dtype=jnp.int32) == bank_id
        ).astype(jnp.float32)
        bank_weight = jnp.sum(bank_route * bank_selector, axis=-1)
        victim_route += bank_weight[..., None] * slot_route

    novel_route = (
        novel_gate[..., None] * admission[..., None] * victim_route
    )
    matched_route = (1.0 - novel_gate[..., None]) * raw_matched_route
    write_route = novel_route + matched_route
    return (
        write_route,
        matched_route,
        novel_route,
        confidence,
        is_novel,
        admission_soft,
        victim_route,
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
    lambda_screen_warmup_floor: float = 0.0
    lambda_screen_warmup_steps: int = 0
    eps: float = 1e-6
    use_value_unit_norm: bool = True
    use_age_mask: bool = False
    use_bank_bias: bool = False
    use_write_screening: bool = False
    # Missing versions retain the historical v4 migration rule.
    semantics_version: str | None = None
    # ``None`` preserves the legacy phase/use_write_screening mapping.
    write_mode: str | None = None
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
    gate_space: str = "model"
    gate_activation: str = "sigmoid"
    candidate_rank: int | None = None
    route_power: float = 1.0
    novelty_threshold: float = 0.1
    novelty_temperature: float = 0.1
    admission_init: float = 0.1
    allocation_temperature: float = 1.0
    bank_route_temperature: float = 1.0
    allocation_top_k: int = 1
    allocation_age_weight: float = 1.0
    allocation_usage_weight: float = 1.0
    admission_threshold: float | None = 0.5
    checkpoint_interval: int | None = None
    n_read_tiles: int = 1
    capacity_calibration: str = "fixed"
    tau_min: float = -0.95
    tau_max: float = 0.95
    target_false_read_rate: float = 0.05
    target_false_write_rate: float = 0.05
    target_false_match_rate: float = 0.01
    threshold_warmup_by_load: bool = True
    threshold_warmup_tau: float = -0.25
    eta_ambiguity: float = 0.5
    edit_mode: str = "capacity_conserving"
    erase_gate_init: float = 0.9
    write_gate_init: float = 0.9
    write_accounting_floor: float = 1e-4
    allocation_redundancy_weight: float = 1.0
    # Initial anti-starvation curriculum; this is not a permanent write quota.
    admission_floor_target_initial: float = 0.0
    admission_floor_weight: float = 0.0
    admission_floor_steps: int = 0
    # Training-only soft-to-hard read curriculum. Inference remains hard.
    read_soft_warmup_steps: int = 0
    read_soft_warmup_temperature: float = 0.1
    # Upper write budget prevents the all-novel/all-write collapse.
    write_budget_target_max: float = 1.0
    write_budget_weight: float = 0.0
    # Temporary self-indexing makes a newly written candidate retrievable by
    # the query that admitted it. This bootstraps read/write key geometry.
    self_index_margin: float = 0.0
    self_index_loss_weight: float = 0.0
    self_index_loss_steps: int = 0

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
        if self.lambda_screen_warmup_floor < 0.0:
            raise ValueError("lambda_screen_warmup_floor must be non-negative")
        if self.lambda_screen_warmup_steps < 0:
            raise ValueError("lambda_screen_warmup_steps must be non-negative")
        if (
            self.lambda_screen_warmup_floor > 0.0
            and self.lambda_screen_warmup_steps == 0
        ):
            raise ValueError(
                "an enabled lambda warm-up floor requires "
                "lambda_screen_warmup_steps > 0"
            )
        for name, value in (
            ("mu_short_max", self.mu_short_max),
            ("mu_mid_max", self.mu_mid_max),
            ("mu_long_max", self.mu_long_max),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.write_mode is not None:
            normalize_write_mode(self.write_mode)
        semantics_version = resolve_semantics_version(self)
        if (
            semantics_version == "screening-v4-legacy"
            and self.write_mode == "competitive_novel"
        ):
            raise ValueError(
                "screening-v4-legacy does not allow competitive_novel"
            )
        if (
            semantics_version
            in {"screening-v4-competitive", "screening-v5-core", "screening-v5-retention"}
            and self.write_mode != "competitive_novel"
        ):
            raise ValueError(
                f"{semantics_version} requires write_mode='competitive_novel'"
            )
        if self.gate_space not in _VALID_GATE_SPACES:
            raise ValueError(
                f"gate_space must be one of {sorted(_VALID_GATE_SPACES)}"
            )
        if self.gate_activation not in _VALID_GATE_ACTIVATIONS:
            raise ValueError(
                "gate_activation must be one of "
                f"{sorted(_VALID_GATE_ACTIVATIONS)}"
            )
        if self.candidate_rank is not None:
            if self.candidate_rank <= 0:
                raise ValueError("candidate_rank must be positive when set")
            legacy_params = (2 * self.d_model + self.d_slot) * self.d_slot
            legacy_params += self.d_slot
            factorized_params = self.candidate_rank * (
                2 * self.d_model + 2 * self.d_slot
            ) + self.d_slot
            if factorized_params >= legacy_params:
                raise ValueError(
                    "candidate_rank must reduce candidate projection parameters "
                    f"for this shape ({factorized_params} >= {legacy_params})"
                )
        if self.route_power <= 0.0:
            raise ValueError("route_power must be positive")
        if not 0.0 <= self.novelty_threshold <= 1.0:
            raise ValueError("novelty_threshold must be in [0, 1]")
        if self.novelty_temperature <= 0.0:
            raise ValueError("novelty_temperature must be positive")
        if not 0.0 < self.admission_init < 1.0:
            raise ValueError("admission_init must be in (0, 1)")
        if self.allocation_temperature <= 0.0:
            raise ValueError("allocation_temperature must be positive")
        if self.bank_route_temperature <= 0.0:
            raise ValueError("bank_route_temperature must be positive")
        if self.allocation_top_k != 1:
            raise ValueError("only allocation_top_k=1 is currently implemented")
        if self.allocation_age_weight < 0.0:
            raise ValueError("allocation_age_weight must be non-negative")
        if self.allocation_usage_weight < 0.0:
            raise ValueError("allocation_usage_weight must be non-negative")
        if (
            self.admission_threshold is not None
            and not 0.0 <= self.admission_threshold <= 1.0
        ):
            raise ValueError("admission_threshold must be in [0, 1] when set")
        if self.checkpoint_interval is not None and self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive when set")
        if self.n_read_tiles <= 0:
            raise ValueError("n_read_tiles must be positive")
        if self.d_k % self.n_read_tiles != 0:
            raise ValueError("d_k must be divisible by n_read_tiles")
        if self.d_v % self.n_read_tiles != 0:
            raise ValueError("d_v must be divisible by n_read_tiles")
        if self.capacity_calibration not in _VALID_CAPACITY_CALIBRATION:
            raise ValueError(
                "capacity_calibration must be one of "
                f"{sorted(_VALID_CAPACITY_CALIBRATION)}"
            )
        if not -1.0 < self.tau_min < self.tau_max < 1.0:
            raise ValueError("tau_min and tau_max must satisfy -1 < min < max < 1")
        if not self.tau_min <= self.tau_init <= self.tau_max:
            raise ValueError("tau_init must lie within [tau_min, tau_max]")
        if not self.tau_min <= self.threshold_warmup_tau <= self.tau_max:
            raise ValueError(
                "threshold_warmup_tau must lie within [tau_min, tau_max]"
            )
        for name, value in (
            ("target_false_read_rate", self.target_false_read_rate),
            ("target_false_write_rate", self.target_false_write_rate),
            ("target_false_match_rate", self.target_false_match_rate),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
        if self.eta_ambiguity < 0.0:
            raise ValueError("eta_ambiguity must be non-negative")
        if self.edit_mode not in _VALID_EDIT_MODES:
            raise ValueError(
                f"edit_mode must be one of {sorted(_VALID_EDIT_MODES)}"
            )
        for name, value in (
            ("erase_gate_init", self.erase_gate_init),
            ("write_gate_init", self.write_gate_init),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
        if not 0.0 < self.write_accounting_floor <= 1.0:
            raise ValueError("write_accounting_floor must be in (0, 1]")
        if self.allocation_redundancy_weight < 0.0:
            raise ValueError(
                "allocation_redundancy_weight must be non-negative"
            )
        if not 0.0 <= self.admission_floor_target_initial <= 1.0:
            raise ValueError(
                "admission_floor_target_initial must be in [0, 1]"
            )
        if self.admission_floor_weight < 0.0:
            raise ValueError("admission_floor_weight must be non-negative")
        if self.admission_floor_steps < 0:
            raise ValueError("admission_floor_steps must be non-negative")
        if (
            self.admission_floor_target_initial > 0.0
            and self.admission_floor_weight > 0.0
            and self.admission_floor_steps == 0
        ):
            raise ValueError(
                "an enabled admission floor requires admission_floor_steps > 0"
            )
        if self.read_soft_warmup_steps < 0:
            raise ValueError("read_soft_warmup_steps must be non-negative")
        if self.read_soft_warmup_temperature <= 0.0:
            raise ValueError("read_soft_warmup_temperature must be positive")
        if not 0.0 <= self.write_budget_target_max <= 1.0:
            raise ValueError("write_budget_target_max must be in [0, 1]")
        if self.write_budget_weight < 0.0:
            raise ValueError("write_budget_weight must be non-negative")
        if not 0.0 <= self.self_index_margin < 1.0:
            raise ValueError("self_index_margin must be in [0, 1)")
        if self.self_index_loss_weight < 0.0:
            raise ValueError("self_index_loss_weight must be non-negative")
        if self.self_index_loss_steps < 0:
            raise ValueError("self_index_loss_steps must be non-negative")
        if self.self_index_loss_weight > 0.0 and self.self_index_loss_steps == 0:
            raise ValueError(
                "an enabled self-index loss requires self_index_loss_steps > 0"
            )
        if semantics_version in {"screening-v5-core", "screening-v5-retention"}:
            if self.gate_space != "value":
                raise ValueError("v5 semantics requires gate_space='value'")
            if self.candidate_rank is None:
                raise ValueError("v5 semantics requires a factorized candidate")
            if self.capacity_calibration != "analytic":
                raise ValueError(
                    "v5 semantics requires capacity_calibration='analytic'"
                )
            if self.use_age_mask:
                raise ValueError("v5 semantics does not use age as a read mask")
            if self.use_leaky_warmup:
                raise ValueError(
                    "v5 hard-read semantics does not use the v4 leaky warmup"
                )
            if self.checkpoint_interval is not None:
                raise ValueError(
                    "v5 checkpoint redesign is not implemented; "
                    "checkpoint_interval must be None"
                )
        elif (
            self.admission_floor_target_initial > 0.0
            or self.admission_floor_weight > 0.0
            or self.admission_floor_steps > 0
            or self.lambda_screen_warmup_floor > 0.0
            or self.lambda_screen_warmup_steps > 0
            or self.read_soft_warmup_steps > 0
            or self.write_budget_target_max != 1.0
            or self.write_budget_weight > 0.0
            or self.self_index_margin > 0.0
            or self.self_index_loss_weight > 0.0
            or self.self_index_loss_steps > 0
        ):
            raise ValueError(
                "anti-starvation curricula are defined only for v5 semantics"
            )
        half_lives = (
            self.short_half_life_tokens,
            self.mid_half_life_tokens,
            self.long_half_life_tokens,
        )
        if any(value is not None and value <= 0.0 for value in half_lives):
            raise ValueError("memory half-life values must be positive when provided")
        if self.checkpoint_interval is not None:
            update_rate_limits = tuple(
                maximum
                if half_life is None
                else -math.expm1(-math.log(2.0) / half_life)
                for maximum, half_life in zip(
                    (
                        self.mu_short_max,
                        self.mu_mid_max,
                        self.mu_long_max,
                    ),
                    half_lives,
                    strict=True,
                )
            )
            threshold_routing_possible = (
                self.write_mode == "legacy_threshold"
                or (self.write_mode is None and self.use_write_screening)
            )
            route_limit = (
                max(1.0, self.write_rel_floor)
                if threshold_routing_possible
                else 1.0
            )
            maximum_effective_strength = (
                max(update_rate_limits) * route_limit
            )
            if maximum_effective_strength > 0.95:
                raise ValueError(
                    "checkpoint_interval requires maximum effective screening "
                    "update strength <= 0.95; got "
                    f"{maximum_effective_strength:.6g}"
                )
        if not 0.0 <= self.usage_ema_decay < 1.0:
            raise ValueError("usage_ema_decay must be in [0, 1)")


class StateLevelScreening(nn.Module):
    """Screening module that can process full sequences via lax.scan."""

    config: ScreeningConfig

    def setup(self):
        cfg = self.config
        if is_v5_semantics(cfg):
            raise NotImplementedError(
                "Screening v5 model integration is NNX-only; the portable "
                "semantic reference is screening_v5_recurrence_reference"
            )
        self.q_proj_r = nn.Dense(cfg.d_k, use_bias=False, name="q_proj_r")
        self.k_proj_r = nn.Dense(cfg.d_k, use_bias=False, name="k_proj_r")
        self.v_proj = nn.Dense(cfg.d_v, use_bias=False, name="v_proj")
        self.out_proj = nn.Dense(cfg.d_model, use_bias=False, name="out_proj")
        gate_size = cfg.d_model if cfg.gate_space == "model" else cfg.d_v
        self.gate_proj = nn.Dense(gate_size, name="gate_proj")
        if cfg.candidate_rank is None:
            self.delta_proj = nn.Dense(cfg.d_slot, name="delta_proj")
        else:
            self.delta_context_proj = nn.Dense(
                cfg.candidate_rank,
                use_bias=False,
                name="delta_context_proj",
            )
            self.delta_slot_proj = nn.Dense(
                cfg.candidate_rank,
                use_bias=False,
                name="delta_slot_proj",
            )
            self.delta_out_proj = nn.Dense(
                cfg.d_slot,
                name="delta_out_proj",
            )
        self.screen_ln = nn.LayerNorm(dtype=jnp.float32, name="screen_ln")
        if write_mode_uses_write_projection(cfg):
            self.q_proj_w = nn.Dense(cfg.d_k, use_bias=False, name="q_proj_w")
            self.k_proj_w = nn.Dense(cfg.d_k, use_bias=False, name="k_proj_w")
        if cfg.write_mode == "competitive_novel":
            admission_bias = jnp.log(
                cfg.admission_init / (1.0 - cfg.admission_init)
            )
            self.admission_proj = nn.Dense(
                1,
                bias_init=nn.initializers.constant(admission_bias),
                name="admission_proj",
            )
            self.bank_route_proj = nn.Dense(3, name="bank_route_proj")

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
        if cfg.candidate_rank is None:
            delta_in = jnp.concatenate([x_ln, h_t, slots[:, 0, :]], axis=-1)
            _ = self.delta_proj(delta_in)
        else:
            route_context = jnp.concatenate([x_ln, h_t], axis=-1)
            context_latent = self.delta_context_proj(route_context)
            slot_latent = self.delta_slot_proj(slots[:, 0, :])
            _ = self.delta_out_proj(jax.nn.silu(context_latent + slot_latent))
        tau_r_shape = () if cfg.n_read_tiles == 1 else (cfg.n_read_tiles,)
        _ = self.param(
            "tau_r_raw",
            lambda rng, shape: jnp.full(shape, theta_from_tau(cfg.tau_init)),
            tau_r_shape,
        )
        _ = self.param("lambda_raw", nn.initializers.constant(jnp.log(jnp.exp(cfg.lambda_screen_init) - 1)), ())
        _ = self.param("slot_embed", nn.initializers.normal(0.02), (cfg.n_slots, cfg.d_slot))
        _ = self.param("mu_by_bank_raw", nn.initializers.constant(0.0), (3,))
        if write_mode_uses_write_projection(cfg):
            q_w_in = jnp.concatenate([x_ln, h_t.astype(jnp.float32)], axis=-1)
            _ = self.q_proj_w(q_w_in)
            _ = self.k_proj_w(slots)
            _ = self.param("tau_w_raw", lambda rng, shape: theta_from_tau(cfg.tau_init), ())
        if cfg.write_mode == "competitive_novel":
            route_input = jnp.concatenate([x_ln, h_t], axis=-1)
            _ = self.admission_proj(route_input)
            _ = self.bank_route_proj(route_input)
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

        from .screening_recurrence import (
            ACTIVE_SLOTS,
            ADMISSION_HIGH_RATE,
            ADMISSION_LOW_RATE,
            ADMISSION_MEAN,
            BANK_LONG_WRITE_MASS,
            BANK_MID_WRITE_MASS,
            BANK_SHORT_WRITE_MASS,
            EVICTION_AGE_MEAN,
            EVICTION_USAGE_MEAN,
            MATCHED_ROUTE_MASS,
            NOVEL_RATE,
            NOVEL_ROUTE_MASS,
            READ_MAX,
            READ_MEAN,
            REJECTED_RATE,
            ROUTE_ENTROPY,
            ROUTE_TOP1,
            U_NORM,
            USAGE_MEAN,
            WRITE_EFFECTIVE_MEAN,
            WRITE_MEAN,
            Z_NORM,
            ScreeningRecurrenceConfig,
            screening_recurrence_reference,
        )

        slots = state.slots.astype(jnp.float32)
        ages = state.ages.astype(jnp.float32)
        usage_ema = state.usage_ema.astype(jnp.float32)
        write_mode = resolve_write_mode(cfg, phase)
        p = self.variables["params"]
        x_ln_seq = self.screen_ln(x_seq.astype(jnp.float32))
        q_r_raw = self.q_proj_r(x_ln_seq).astype(jnp.float32)
        q_r_tiled = q_r_raw.reshape(
            *q_r_raw.shape[:-1],
            cfg.n_read_tiles,
            cfg.d_k // cfg.n_read_tiles,
        )
        q_r_seq = unit_norm(q_r_tiled, eps=cfg.eps).reshape(q_r_raw.shape)
        gate_seq = apply_screening_gate(
            self.gate_proj(x_ln_seq).astype(jnp.float32),
            cfg.gate_activation,
        )
        slot_embed = p["slot_embed"]
        if cfg.candidate_rank is None:
            delta_s_seq = compute_slot_delta(
                x_ln_seq,
                h_base_seq.astype(jnp.float32),
                slot_embed,
                p["delta_proj"]["kernel"],
                p["delta_proj"]["bias"],
            )
        else:
            route_context = jnp.concatenate(
                [x_ln_seq, h_base_seq.astype(jnp.float32)], axis=-1
            )
            context_latent = self.delta_context_proj(route_context)
            slot_latent = self.delta_slot_proj(slot_embed)
            delta_s_seq = jnp.tanh(
                self.delta_out_proj(
                    jax.nn.silu(
                        context_latent[..., None, :]
                        + slot_latent[None, None, :, :]
                    )
                )
            )

        initial_read_keys = self.k_proj_r(slots)
        initial_values = self.v_proj(slots)
        delta_read_keys = self.k_proj_r(delta_s_seq)
        delta_values = self.v_proj(delta_s_seq)
        tau_r = bounded_tau(p["tau_r_raw"]).astype(jnp.float32)
        tau_w = jnp.zeros((), dtype=jnp.float32)
        uses_write_score = write_mode in (
            "legacy_threshold",
            "competitive_novel",
        )
        if uses_write_score:
            route_input = jnp.concatenate(
                [x_ln_seq, h_base_seq.astype(jnp.float32)], axis=-1
            )
            q_w_seq = unit_norm(
                self.q_proj_w(route_input).astype(jnp.float32), eps=cfg.eps
            )
            tau_w = bounded_tau(p["tau_w_raw"]).astype(jnp.float32)
            initial_write_keys = self.k_proj_w(
                slots + slot_embed[None, :, :]
            )
            delta_write_keys = self.k_proj_w(
                delta_s_seq + slot_embed[None, None, :, :]
            )
        else:
            q_w_seq = jnp.zeros_like(q_r_seq)
            initial_write_keys = jnp.zeros_like(initial_read_keys)
            delta_write_keys = jnp.zeros_like(delta_read_keys)

        if write_mode == "competitive_novel":
            admission_seq = jax.nn.sigmoid(
                self.admission_proj(route_input).astype(jnp.float32)[..., 0]
            )
            bank_logits_seq = self.bank_route_proj(route_input).astype(
                jnp.float32
            )
        else:
            admission_seq = jnp.zeros(q_r_seq.shape[:2], dtype=jnp.float32)
            bank_logits_seq = jnp.zeros(
                (*q_r_seq.shape[:2], 3), dtype=jnp.float32
            )

        recurrence_config = ScreeningRecurrenceConfig(
            write_enabled=uses_write_score,
            use_value_unit_norm=cfg.use_value_unit_norm,
            use_leaky_warmup=cfg.use_leaky_warmup,
            leaky_alpha=cfg.leaky_alpha,
            leaky_gamma=cfg.leaky_gamma,
            use_age_mask=cfg.use_age_mask,
            age_ref=cfg.age_ref,
            age_sigma=cfg.age_sigma,
            write_rel_floor=cfg.write_rel_floor,
            usage_ema_decay=cfg.usage_ema_decay,
            tanh_norm_cap=cfg.tanh_norm_cap,
            eps=cfg.eps,
            write_mode=write_mode,
            bank_ids=cfg.bank_ids,
            route_power=cfg.route_power,
            novelty_threshold=cfg.novelty_threshold,
            novelty_temperature=cfg.novelty_temperature,
            allocation_temperature=cfg.allocation_temperature,
            bank_route_temperature=cfg.bank_route_temperature,
            allocation_age_weight=cfg.allocation_age_weight,
            allocation_usage_weight=cfg.allocation_usage_weight,
            hard_admission=(write_mode == "competitive_novel"),
            admission_threshold=(
                cfg.admission_threshold
                if cfg.admission_threshold is not None
                else 0.5
            ),
            n_read_tiles=cfg.n_read_tiles,
            checkpoint_interval=cfg.checkpoint_interval,
        )
        time_inputs = tuple(
            jnp.swapaxes(value, 0, 1)
            for value in (
                q_r_seq,
                q_w_seq,
                admission_seq,
                bank_logits_seq,
                delta_s_seq,
                delta_read_keys,
                delta_values,
                delta_write_keys,
            )
        )
        (
            u_time,
            final_slots,
            final_ages,
            final_usage,
            step_statistics,
            update_squared,
        ) = screening_recurrence_reference(
            *time_inputs,
            slots,
            initial_read_keys,
            initial_values,
            initial_write_keys,
            ages,
            usage_ema,
            self._compute_mu(p, cfg).astype(jnp.float32),
            tau_r,
            tau_w,
            recurrence_config,
        )

        u_seq = jnp.swapaxes(u_time, 0, 1)
        lambda_screen = jax.nn.softplus(p["lambda_raw"]).astype(jnp.float32)
        effective_lambda = lambda_screen / jnp.sqrt(
            jnp.asarray(cfg.n_read_tiles, dtype=jnp.float32)
        )
        if cfg.gate_space == "value":
            memory_branch = self.out_proj(
                u_seq * gate_seq.astype(u_seq.dtype)
            ).astype(h_base_seq.dtype)
        else:
            memory_branch = (
                gate_seq * self.out_proj(u_seq).astype(jnp.float32)
            ).astype(h_base_seq.dtype)
        h = (h_base_seq + effective_lambda * memory_branch).astype(
            h_base_seq.dtype
        )

        slot_norm = unit_norm(final_slots, eps=cfg.eps)
        slot_similarity = jnp.einsum("bms,bns->bmn", slot_norm, slot_norm)
        off_diagonal = 1.0 - jnp.eye(cfg.n_slots, dtype=jnp.float32)
        redundancy_denominator = max(cfg.n_slots * (cfg.n_slots - 1), 1)
        stat_map = {
            "rel_read_mean": READ_MEAN,
            "rel_read_max": READ_MAX,
            "active_slots_mean": ACTIVE_SLOTS,
            "z_norm_mean": Z_NORM,
            "u_norm_mean": U_NORM,
            "rel_write_mean": WRITE_MEAN,
            "rel_write_effective_mean": WRITE_EFFECTIVE_MEAN,
            "slot_usage_ema_mean": USAGE_MEAN,
            "matched_route_mass": MATCHED_ROUTE_MASS,
            "novel_route_mass": NOVEL_ROUTE_MASS,
            "route_entropy": ROUTE_ENTROPY,
            "route_top1_concentration": ROUTE_TOP1,
            "admission_mean": ADMISSION_MEAN,
            "admission_low_rate": ADMISSION_LOW_RATE,
            "admission_high_rate": ADMISSION_HIGH_RATE,
            "novel_token_rate": NOVEL_RATE,
            "rejected_write_rate": REJECTED_RATE,
            "short_bank_write_mass": BANK_SHORT_WRITE_MASS,
            "mid_bank_write_mass": BANK_MID_WRITE_MASS,
            "long_bank_write_mass": BANK_LONG_WRITE_MASS,
            "eviction_age_mean": EVICTION_AGE_MEAN,
            "eviction_usage_mean": EVICTION_USAGE_MEAN,
        }
        agg_stats = {
            key: jnp.mean(step_statistics[..., index])
            for key, index in stat_map.items()
        }
        agg_stats.update(
            {
                "tau_r": jnp.mean(tau_r),
                "tau_r_min": jnp.min(tau_r),
                "tau_r_max": jnp.max(tau_r),
                "lambda_screen": effective_lambda,
                "tau_w": tau_w,
                "slot_update_norm_mean": jnp.mean(jnp.sqrt(update_squared)),
                "slot_utilization": jnp.mean(final_usage > 1e-3),
                "dead_slot_rate": jnp.mean(final_usage <= 1e-3),
                "slot_cosine_redundancy": jnp.sum(
                    jnp.abs(slot_similarity) * off_diagonal[None, :, :]
                ) / (final_slots.shape[0] * redundancy_denominator),
            }
        )
        new_state = LayerScreenState(
            slots=final_slots.astype(state.slots.dtype),
            ages=final_ages,
            usage_ema=final_usage.astype(state.usage_ema.dtype),
        )
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
