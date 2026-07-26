import argparse
import json

from ..kernels import available_optimizer_backends


def add_training_vocab_tiling_args(parser: argparse.ArgumentParser):
    """Add execution-only vocabulary-tiling controls to a CLI parser."""

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--training-vocab-tile-size",
        type=int,
        default=None,
        help=(
            "vocabulary tile used by the accelerator streaming training "
            "loss; the model-config value is retained when omitted"
        ),
    )
    group.add_argument(
        "--no-training-vocab-tiling",
        action="store_true",
        help="use the portable full-logits training loss",
    )


def add_optimizer_backend_arg(parser: argparse.ArgumentParser):
    """Add the opt-in fused-optimizer selector to a training CLI."""

    parser.add_argument(
        "--optimizer-backend",
        choices=available_optimizer_backends(),
        default=None,
        help="optimizer implementation; Pallas GPU paths are opt-in",
    )


def add_screening_v2_args(parser: argparse.ArgumentParser):
    """Add versioned architecture controls for Screening v4/v5 paths."""

    parser.add_argument(
        "--screening-semantics-version",
        choices=(
            "screening-v4-legacy",
            "screening-v4-competitive",
            "screening-v5-core",
            "screening-v5-retention",
        ),
        default=None,
    )
    parser.add_argument(
        "--write-mode",
        choices=(
            "disabled",
            "legacy_unconditional",
            "legacy_threshold",
            "competitive_novel",
        ),
        default=None,
    )
    parser.add_argument(
        "--screening-gate-space",
        choices=("model", "value"),
        default="model",
    )
    parser.add_argument(
        "--screening-gate-activation",
        choices=("sigmoid", "tanh_silu"),
        default="sigmoid",
    )
    parser.add_argument(
        "--screening-lambda-warmup-floor", type=float, default=None
    )
    parser.add_argument(
        "--screening-lambda-warmup-steps", type=int, default=None
    )
    parser.add_argument("--screening-candidate-rank", type=int, default=None)
    parser.add_argument("--screening-route-power", type=float, default=1.0)
    parser.add_argument(
        "--screening-novelty-threshold", type=float, default=0.1
    )
    parser.add_argument(
        "--screening-novelty-temperature", type=float, default=0.1
    )
    parser.add_argument("--screening-admission-init", type=float, default=0.1)
    parser.add_argument(
        "--screening-allocation-temperature", type=float, default=1.0
    )
    parser.add_argument(
        "--screening-bank-route-temperature", type=float, default=1.0
    )
    parser.add_argument(
        "--screening-allocation-age-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--screening-allocation-usage-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--screening-admission-threshold", type=float, default=0.5
    )
    parser.add_argument(
        "--screening-checkpoint-interval", type=int, default=None
    )
    parser.add_argument("--screening-read-tiles", type=int, default=1)
    parser.add_argument(
        "--screening-capacity-calibration",
        choices=("fixed", "analytic"),
        default="fixed",
    )
    parser.add_argument("--screening-tau-min", type=float, default=-0.95)
    parser.add_argument("--screening-tau-max", type=float, default=0.95)
    parser.add_argument(
        "--screening-target-false-read-rate",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--screening-target-false-write-rate",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--screening-target-false-match-rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--screening-threshold-warmup-by-load",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--screening-threshold-warmup-tau",
        type=float,
        default=-0.25,
    )
    parser.add_argument(
        "--screening-eta-ambiguity",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--screening-edit-mode",
        choices=("tied", "capacity_conserving", "free_edit"),
        default="capacity_conserving",
    )
    parser.add_argument(
        "--screening-erase-gate-init",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--screening-write-gate-init",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--screening-write-accounting-floor",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--screening-allocation-redundancy-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--screening-admission-floor-target-initial",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--screening-admission-floor-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--screening-admission-floor-steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--screening-read-soft-warmup-steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--screening-read-soft-warmup-temperature",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--screening-write-budget-target-max",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--screening-write-budget-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--screening-self-index-margin",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--screening-self-index-loss-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--screening-self-index-loss-steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--screening-activation-step",
        type=int,
        default=None,
        help="keep v5 Screening inert before this optimizer step",
    )
    parser.add_argument(
        "--screening-activation-warmup-steps",
        type=int,
        default=None,
        help="linearly ramp v5 Screening after its activation step",
    )
    parser.add_argument(
        "--screening-optimizer-lr-multiplier",
        type=float,
        default=None,
        help="multiply optimizer updates for Screening parameters",
    )
    parser.add_argument(
        "--screening-write-budget-min-slot-utilization",
        type=float,
        default=None,
        help=(
            "utilization at which the legacy upper v5 write budget reaches "
            "full strength; lower utilization is scaled continuously"
        ),
    )
    parser.add_argument(
        "--screening-admission-controller",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="control realized hard writes with a bounded per-layer PI actuator",
    )
    parser.add_argument(
        "--screening-admission-controller-target",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--screening-admission-controller-rate-ema-decay",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--screening-admission-controller-kp",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--screening-admission-controller-ki",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--screening-admission-controller-max-step",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--screening-admission-controller-bias-limit",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--screening-admission-quota",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "select a stable per-window hard-forward admission quota while "
            "retaining soft admission gradients"
        ),
    )
    parser.add_argument(
        "--screening-admission-quota-target",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--screening-admission-quota-window",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--screening-detach-inputs-steps",
        type=int,
        default=None,
        help="stop Screening-to-trunk input gradients for early v5 steps",
    )


def screening_v2_kwargs(args):
    def value_or_default(name, default):
        value = getattr(args, name, None)
        return default if value is None else value

    return {
        "semantics_version": getattr(
            args, "screening_semantics_version", None
        ),
        "write_mode": getattr(args, "write_mode", None),
        "gate_space": getattr(args, "screening_gate_space", "model"),
        "gate_activation": getattr(
            args, "screening_gate_activation", "sigmoid"
        ),
        "lambda_screen_warmup_floor": value_or_default(
            "screening_lambda_warmup_floor", 0.0
        ),
        "lambda_screen_warmup_steps": value_or_default(
            "screening_lambda_warmup_steps", 0
        ),
        "candidate_rank": getattr(args, "screening_candidate_rank", None),
        "route_power": getattr(args, "screening_route_power", 1.0),
        "novelty_threshold": getattr(
            args, "screening_novelty_threshold", 0.1
        ),
        "novelty_temperature": getattr(
            args, "screening_novelty_temperature", 0.1
        ),
        "admission_init": getattr(args, "screening_admission_init", 0.1),
        "allocation_temperature": getattr(
            args, "screening_allocation_temperature", 1.0
        ),
        "bank_route_temperature": getattr(
            args, "screening_bank_route_temperature", 1.0
        ),
        "allocation_age_weight": getattr(
            args, "screening_allocation_age_weight", 1.0
        ),
        "allocation_usage_weight": getattr(
            args, "screening_allocation_usage_weight", 1.0
        ),
        "admission_threshold": getattr(
            args, "screening_admission_threshold", 0.5
        ),
        "checkpoint_interval": getattr(
            args, "screening_checkpoint_interval", None
        ),
        "n_read_tiles": getattr(args, "screening_read_tiles", 1),
        "capacity_calibration": getattr(
            args, "screening_capacity_calibration", "fixed"
        ),
        "tau_min": getattr(args, "screening_tau_min", -0.95),
        "tau_max": getattr(args, "screening_tau_max", 0.95),
        "target_false_read_rate": getattr(
            args, "screening_target_false_read_rate", 0.05
        ),
        "target_false_write_rate": getattr(
            args, "screening_target_false_write_rate", 0.05
        ),
        "target_false_match_rate": getattr(
            args, "screening_target_false_match_rate", 0.01
        ),
        "threshold_warmup_by_load": getattr(
            args, "screening_threshold_warmup_by_load", True
        ),
        "threshold_warmup_tau": getattr(
            args, "screening_threshold_warmup_tau", -0.25
        ),
        "eta_ambiguity": getattr(
            args, "screening_eta_ambiguity", 0.5
        ),
        "edit_mode": getattr(
            args, "screening_edit_mode", "capacity_conserving"
        ),
        "erase_gate_init": getattr(
            args, "screening_erase_gate_init", 0.9
        ),
        "write_gate_init": getattr(
            args, "screening_write_gate_init", 0.9
        ),
        "write_accounting_floor": getattr(
            args, "screening_write_accounting_floor", 1e-4
        ),
        "allocation_redundancy_weight": getattr(
            args, "screening_allocation_redundancy_weight", 1.0
        ),
        "admission_floor_target_initial": value_or_default(
            "screening_admission_floor_target_initial", 0.0
        ),
        "admission_floor_weight": value_or_default(
            "screening_admission_floor_weight", 0.0
        ),
        "admission_floor_steps": value_or_default(
            "screening_admission_floor_steps", 0
        ),
        "read_soft_warmup_steps": value_or_default(
            "screening_read_soft_warmup_steps", 0
        ),
        "read_soft_warmup_temperature": value_or_default(
            "screening_read_soft_warmup_temperature", 0.1
        ),
        "write_budget_target_max": value_or_default(
            "screening_write_budget_target_max", 1.0
        ),
        "write_budget_weight": value_or_default(
            "screening_write_budget_weight", 0.0
        ),
        "self_index_margin": value_or_default(
            "screening_self_index_margin", 0.0
        ),
        "self_index_loss_weight": value_or_default(
            "screening_self_index_loss_weight", 0.0
        ),
        "self_index_loss_steps": value_or_default(
            "screening_self_index_loss_steps", 0
        ),
        "activation_step": (
            getattr(args, "screening_activation_step", None) or 0
        ),
        "activation_warmup_steps": (
            getattr(args, "screening_activation_warmup_steps", None) or 0
        ),
        "optimizer_lr_multiplier": (
            1.0
            if getattr(args, "screening_optimizer_lr_multiplier", None)
            is None
            else args.screening_optimizer_lr_multiplier
        ),
        "write_budget_min_slot_utilization": (
            0.0
            if getattr(
                args,
                "screening_write_budget_min_slot_utilization",
                None,
            )
            is None
            else args.screening_write_budget_min_slot_utilization
        ),
        "admission_controller_enabled": value_or_default(
            "screening_admission_controller", False
        ),
        "admission_controller_target": value_or_default(
            "screening_admission_controller_target", 0.05
        ),
        "admission_controller_rate_ema_decay": value_or_default(
            "screening_admission_controller_rate_ema_decay", 0.9
        ),
        "admission_controller_kp": value_or_default(
            "screening_admission_controller_kp", 0.1
        ),
        "admission_controller_ki": value_or_default(
            "screening_admission_controller_ki", 0.02
        ),
        "admission_controller_max_step": value_or_default(
            "screening_admission_controller_max_step", 0.1
        ),
        "admission_controller_bias_limit": value_or_default(
            "screening_admission_controller_bias_limit", 6.0
        ),
        "admission_quota_enabled": value_or_default(
            "screening_admission_quota", False
        ),
        "admission_quota_target": value_or_default(
            "screening_admission_quota_target", 0.05
        ),
        "admission_quota_window": value_or_default(
            "screening_admission_quota_window", 128
        ),
        "detach_screening_inputs_steps": value_or_default(
            "screening_detach_inputs_steps", 0
        ),
    }


def apply_execution_overrides(config, args):
    """Apply opt-in remat/chunk controls without replacing model presets."""
    remat_blocks = getattr(args, "remat_blocks", None)
    if remat_blocks is not None:
        config.remat_blocks = bool(remat_blocks)

    sequence_chunk_size = getattr(args, "sequence_chunk_size", None)
    no_sequence_chunking = getattr(args, "no_sequence_chunking", False)
    if no_sequence_chunking and sequence_chunk_size is not None:
        raise ValueError(
            "--sequence-chunk-size and --no-sequence-chunking are mutually exclusive"
        )
    if no_sequence_chunking:
        config.sequence_chunk_size = None
    elif sequence_chunk_size is not None:
        if sequence_chunk_size <= 0:
            raise ValueError("--sequence-chunk-size must be positive")
        config.sequence_chunk_size = int(sequence_chunk_size)

    head_chunk_size = getattr(args, "head_chunk_size", None)
    no_head_chunking = getattr(args, "no_head_chunking", False)
    if no_head_chunking and head_chunk_size is not None:
        raise ValueError(
            "--head-chunk-size and --no-head-chunking are mutually exclusive"
        )
    if no_head_chunking:
        config.head_chunk_size = int(
            getattr(args, "ctx_len", config.max_seq_len)
        )
    elif head_chunk_size is not None:
        if head_chunk_size <= 0:
            raise ValueError("--head-chunk-size must be positive")
        config.head_chunk_size = int(head_chunk_size)

    vocab_tile_size = getattr(args, "training_vocab_tile_size", None)
    no_vocab_tiling = getattr(args, "no_training_vocab_tiling", False)
    if no_vocab_tiling:
        config.training_vocab_tile_size = None
    elif vocab_tile_size is not None:
        if vocab_tile_size <= 0:
            raise ValueError("--training-vocab-tile-size must be positive")
        if (
            config.vocab_size > vocab_tile_size
            and config.vocab_size % vocab_tile_size != 0
        ):
            raise ValueError(
                "vocab_size must be divisible by "
                "--training-vocab-tile-size when multiple tiles are required"
            )
        config.training_vocab_tile_size = int(vocab_tile_size)
    optimizer_backend = getattr(args, "optimizer_backend", None)
    if optimizer_backend is not None:
        config.optimizer_backend = optimizer_backend
    warmup_steps = getattr(args, "warmup_steps", None)
    if warmup_steps is not None:
        if warmup_steps < 0:
            raise ValueError("--warmup-steps must be non-negative")
        config.warmup_steps = int(warmup_steps)
    optimizer_overrides = (
        ("lr_init", "lr_init"),
        ("lr_final", "lr_final"),
        ("lr_schedule", "lr_schedule"),
        ("max_grad_norm", "max_grad_norm"),
        ("gradient_spike_max_abs", "gradient_spike_max_abs"),
        ("weight_decay", "weight_decay"),
        ("adam_beta1", "adam_beta1"),
        ("adam_beta2", "adam_beta2"),
        ("adam_eps", "adam_eps"),
    )
    for argument_name, field_name in optimizer_overrides:
        value = getattr(args, argument_name, None)
        if value is not None:
            setattr(config, field_name, value)
    if hasattr(args, "use_screening") and not args.use_screening:
        config.use_screening = False
    screening_overrides = (
        ("screening_activation_step", "activation_step"),
        ("screening_activation_warmup_steps", "activation_warmup_steps"),
        ("screening_optimizer_lr_multiplier", "optimizer_lr_multiplier"),
        ("screening_lambda_warmup_floor", "lambda_screen_warmup_floor"),
        ("screening_lambda_warmup_steps", "lambda_screen_warmup_steps"),
        (
            "screening_admission_floor_target_initial",
            "admission_floor_target_initial",
        ),
        ("screening_admission_floor_weight", "admission_floor_weight"),
        ("screening_admission_floor_steps", "admission_floor_steps"),
        ("screening_read_soft_warmup_steps", "read_soft_warmup_steps"),
        (
            "screening_read_soft_warmup_temperature",
            "read_soft_warmup_temperature",
        ),
        ("screening_write_budget_target_max", "write_budget_target_max"),
        ("screening_write_budget_weight", "write_budget_weight"),
        ("screening_self_index_margin", "self_index_margin"),
        ("screening_self_index_loss_weight", "self_index_loss_weight"),
        ("screening_self_index_loss_steps", "self_index_loss_steps"),
        (
            "screening_write_budget_min_slot_utilization",
            "write_budget_min_slot_utilization",
        ),
        (
            "screening_admission_controller",
            "admission_controller_enabled",
        ),
        (
            "screening_admission_controller_target",
            "admission_controller_target",
        ),
        (
            "screening_admission_controller_rate_ema_decay",
            "admission_controller_rate_ema_decay",
        ),
        (
            "screening_admission_controller_kp",
            "admission_controller_kp",
        ),
        (
            "screening_admission_controller_ki",
            "admission_controller_ki",
        ),
        (
            "screening_admission_controller_max_step",
            "admission_controller_max_step",
        ),
        (
            "screening_admission_controller_bias_limit",
            "admission_controller_bias_limit",
        ),
        (
            "screening_admission_quota",
            "admission_quota_enabled",
        ),
        (
            "screening_admission_quota_target",
            "admission_quota_target",
        ),
        (
            "screening_admission_quota_window",
            "admission_quota_window",
        ),
        (
            "screening_detach_inputs_steps",
            "detach_screening_inputs_steps",
        ),
    )
    for argument_name, field_name in screening_overrides:
        value = getattr(args, argument_name, None)
        if value is not None:
            setattr(config.screening, field_name, value)
    config.screening.__post_init__()
    config.__post_init__()
    return config


def parse_args_with_config(parser: argparse.ArgumentParser, argv=None):
    parser.add_argument("--config", default=None, help="JSON config file with CLI option defaults")

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    if pre_args.config is None:
        return parser.parse_args(argv)

    with open(pre_args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError("config file must contain a JSON object")

    actions_by_dest = {
        action.dest: action
        for action in parser._actions
        if action.dest not in (argparse.SUPPRESS, "help")
    }
    unknown = sorted(set(config) - set(actions_by_dest))
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")

    parser.set_defaults(**config)
    for key in config:
        actions_by_dest[key].required = False
    return parser.parse_args(argv)
