"""Materialize one framework-neutral fixed batch from an RWKV binidx dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rwkv7m.data import create_binidx_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--ctx-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--magic-prime", type=int, default=None)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.ctx_len <= 0 or args.batch_size <= 0 or args.step < 0:
        parser.error("length and batch must be positive; step must be non-negative")

    dataset = create_binidx_dataset(
        args.data_file,
        ctx_len=args.ctx_len,
        batch_size=args.batch_size,
        magic_prime=args.magic_prime,
        epoch_steps=max(args.step + 1, 1),
    )
    try:
        batch = dataset.get_batch(args.step)
    finally:
        dataset.close()
    metadata = {
        "format": "rwkv7m-compute-benchmark-batch",
        "format_version": 1,
        "ctx_len": args.ctx_len,
        "batch_size": args.batch_size,
        "step": args.step,
        "magic_prime": args.magic_prime,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        input_ids=np.asarray(batch["input_ids"], dtype=np.int64),
        target_ids=np.asarray(batch["target_ids"], dtype=np.int64),
        mask=np.asarray(batch.get("mask", np.ones_like(batch["input_ids"])), dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"wrote fixed compute benchmark batch: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
