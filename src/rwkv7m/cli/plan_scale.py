import argparse
import json

from ..distributed.environment import runtime_version_manifest
from ..distributed.scaling import DTypePolicy, estimate_training_memory
from ..io import load_model_config
from ..model import MODEL_PRESET_NAMES, model_preset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Estimate parameter-related memory for a model configuration."
    )
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--model-config")
    model_source.add_argument("--model-preset", choices=MODEL_PRESET_NAMES)
    parser.add_argument("--model-axis-size", type=int, required=True)
    parser.add_argument(
        "--dtype-profile",
        choices=("config", "stability", "memory"),
        default="config",
    )
    return parser.parse_args(argv)


def build_report(args):
    config = (
        load_model_config(args.model_config)
        if args.model_config is not None
        else model_preset(args.model_preset)
    )
    if args.dtype_profile == "memory":
        policy = DTypePolicy.memory()
    elif args.dtype_profile == "stability":
        policy = DTypePolicy.stability()
    else:
        policy = DTypePolicy.from_config(config)
    estimate = estimate_training_memory(
        config,
        policy,
        model_axis_size=args.model_axis_size,
    )
    return {
        "environment": runtime_version_manifest(),
        "estimate": estimate.to_dict(),
    }


def main(argv=None):
    print(json.dumps(build_report(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
