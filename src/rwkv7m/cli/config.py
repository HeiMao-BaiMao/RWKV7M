import argparse
import json


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
