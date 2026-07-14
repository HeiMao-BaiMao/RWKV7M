import argparse
import json


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
