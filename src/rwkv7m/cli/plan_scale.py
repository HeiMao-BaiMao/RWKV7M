import argparse
import json

from ..distributed.environment import runtime_version_manifest
from ..distributed.scaling import DTypePolicy, estimate_training_memory
from ..io import model_config_from_dict


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Estimate parameter-related memory for a model configuration."
    )
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--model-axis-size", type=int, required=True)
    parser.add_argument(
        "--dtype-profile",
        choices=("stability", "memory"),
        default="stability",
    )
    return parser.parse_args(argv)


def build_report(args):
    with open(args.model_config, "r", encoding="utf-8") as stream:
        config = model_config_from_dict(json.load(stream))
    policy = (
        DTypePolicy.memory()
        if args.dtype_profile == "memory"
        else DTypePolicy.stability()
    )
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
