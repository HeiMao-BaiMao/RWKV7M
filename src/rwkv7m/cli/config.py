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
        "--screening-lambda-warmup-floor", type=float, default=0.0
    )
    parser.add_argument(
        "--screening-lambda-warmup-steps", type=int, default=0
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
        default=0.0,
    )
    parser.add_argument(
        "--screening-admission-floor-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--screening-admission-floor-steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--screening-read-soft-warmup-steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--screening-read-soft-warmup-temperature",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--screening-write-budget-target-max",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--screening-write-budget-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--screening-self-index-margin",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--screening-self-index-loss-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--screening-self-index-loss-steps",
        type=int,
        default=0,
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
            "enable the upper v5 write budget only after each screened "
            "layer reaches this occupied-slot fraction"
        ),
    )


def screening_v2_kwargs(args):
    return {
        "semantics_version": getattr(
            args, "screening_semantics_version", None
        ),
        "write_mode": getattr(args, "write_mode", None),
        "gate_space": getattr(args, "screening_gate_space", "model"),
        "gate_activation": getattr(
            args, "screening_gate_activation", "sigmoid"
        ),
        "lambda_screen_warmup_floor": getattr(
            args, "screening_lambda_warmup_floor", 0.0
        ),
        "lambda_screen_warmup_steps": getattr(
            args, "screening_lambda_warmup_steps", 0
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
        "admission_floor_target_initial": getattr(
            args, "screening_admission_floor_target_initial", 0.0
        ),
        "admission_floor_weight": getattr(
            args, "screening_admission_floor_weight", 0.0
        ),
        "admission_floor_steps": getattr(
            args, "screening_admission_floor_steps", 0
        ),
        "read_soft_warmup_steps": getattr(
            args, "screening_read_soft_warmup_steps", 0
        ),
        "read_soft_warmup_temperature": getattr(
            args, "screening_read_soft_warmup_temperature", 0.1
        ),
        "write_budget_target_max": getattr(
            args, "screening_write_budget_target_max", 1.0
        ),
        "write_budget_weight": getattr(
            args, "screening_write_budget_weight", 0.0
        ),
        "self_index_margin": getattr(
            args, "screening_self_index_margin", 0.0
        ),
        "self_index_loss_weight": getattr(
            args, "screening_self_index_loss_weight", 0.0
        ),
        "self_index_loss_steps": getattr(
            args, "screening_self_index_loss_steps", 0
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
        (
            "screening_write_budget_min_slot_utilization",
            "write_budget_min_slot_utilization",
        ),
    )
    for argument_name, field_name in screening_overrides:
        value = getattr(args, argument_name, None)
        if value is not None:
            setattr(config.screening, field_name, value)
    config.screening.__post_init__()
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
