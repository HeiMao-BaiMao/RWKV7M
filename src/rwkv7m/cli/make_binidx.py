import argparse
import json
import random
from pathlib import Path

import numpy as np

from ..data.binidx import MMapIndexedDataset, MMapIndexedDatasetBuilder
from ..data.dataset import find_magic_prime
from ..tokenizer import RWKVTokenizer, default_vocab_path


def build_binidx(input_file, output_prefix, *, vocab_file, epochs, seed, dtype):
    input_path = Path(input_file)
    output_prefix = Path(output_prefix)
    tokenizer = RWKVTokenizer(vocab_file)
    rng = random.Random(seed)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"no non-empty JSONL rows found in {input_file}")

    builder = MMapIndexedDatasetBuilder(str(output_prefix) + ".bin", dtype=np.dtype(dtype).type)
    rows = list(lines)
    item_count = 0
    for _ in range(int(epochs)):
        rng.shuffle(rows)
        for line in rows:
            text = json.loads(line)["text"]
            tokens = tokenizer.encode(text, add_eos=True)
            builder.add_item(np.asarray(tokens, dtype=np.dtype(dtype).type))
            builder.end_document()
            item_count += 1
    builder.finalize(str(output_prefix) + ".idx")

    data = MMapIndexedDataset(str(output_prefix))
    try:
        return {
            "items": item_count,
            "tokens": data.data_size,
            "dtype": np.dtype(data.dtype).name,
        }
    finally:
        data.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Convert JSONL text to RWKV .bin/.idx data.")
    parser.add_argument("input_file")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--vocab-file", default=str(default_vocab_path()))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--ctx-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["uint16", "uint32"], default="uint16")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = str(Path(args.input_file).with_suffix(""))
    stats = build_binidx(
        args.input_file,
        output_prefix,
        vocab_file=args.vocab_file,
        epochs=args.epochs,
        seed=args.seed,
        dtype=args.dtype,
    )
    print(
        f"wrote {output_prefix}.bin/.idx items={stats['items']} "
        f"tokens={stats['tokens']} dtype={stats['dtype']}"
    )
    if stats["tokens"] >= args.ctx_len + 1:
        print(f"magic_prime={find_magic_prime(stats['tokens'], args.ctx_len)} ctx_len={args.ctx_len}")


if __name__ == "__main__":
    main()
