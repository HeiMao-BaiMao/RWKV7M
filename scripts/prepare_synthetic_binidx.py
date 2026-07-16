"""Create a deterministic synthetic binidx dataset for accelerator benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rwkv7m.data import MMapIndexedDatasetBuilder, data_file_path, index_file_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", default="data/bench/synthetic")
    parser.add_argument("--tokens", type=int, default=4_194_304)
    parser.add_argument("--vocab-size", type=int, default=65_536)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")
    if not 2 <= args.vocab_size <= np.iinfo(np.uint16).max + 1:
        raise ValueError("--vocab-size must be in [2, 65536]")

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix_str = str(prefix)
    tokens = (np.arange(args.tokens, dtype=np.uint64) % args.vocab_size).astype(
        np.uint16
    )
    builder = MMapIndexedDatasetBuilder(data_file_path(prefix_str), dtype=np.uint16)
    builder.add_item(tokens)
    builder.end_document()
    builder.finalize(index_file_path(prefix_str))
    print(f"wrote {args.tokens} tokens to {prefix}.bin/.idx")


if __name__ == "__main__":
    main()
