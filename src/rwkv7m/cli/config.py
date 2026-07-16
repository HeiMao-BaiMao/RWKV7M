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
    """Add architecture controls for the opt-in Screening v2 path."""

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
    parser.add_argument("--screening-candidate-rank", type=int, default=None)
    parser.add_argument("--screening-route-power", type=float, default=1.0)
    parser.add_argument(
        "--screening-novelty-threshold", type=float, default=0.1
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
        "--screening-admission-threshold", type=float, default=None
    )
    parser.add_argument(
        "--screening-checkpoint-interval", type=int, default=None
    )
    parser.add_argument("--screening-read-tiles", type=int, default=1)


def screening_v2_kwargs(args):
    return {
        "write_mode": getattr(args, "write_mode", None),
        "gate_space": getattr(args, "screening_gate_space", "model"),
        "gate_activation": getattr(
            args, "screening_gate_activation", "sigmoid"
        ),
        "candidate_rank": getattr(args, "screening_candidate_rank", None),
        "route_power": getattr(args, "screening_route_power", 1.0),
        "novelty_threshold": getattr(
            args, "screening_novelty_threshold", 0.1
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
            args, "screening_admission_threshold", None
        ),
        "checkpoint_interval": getattr(
            args, "screening_checkpoint_interval", None
        ),
        "n_read_tiles": getattr(args, "screening_read_tiles", 1),
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
