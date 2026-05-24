import argparse
import json
import random
from pathlib import Path

import numpy as np

from ..data.binidx import MMapIndexedDataset, MMapIndexedDatasetBuilder
from ..data.dataset import find_magic_prime


class Trie:
    __slots__ = ("ch", "to", "values", "front")

    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for _ in range(256)]
        self.values = set()
        self.front = front

    def add(self, key: bytes, idx: int = 0, val=None):
        if idx == len(key):
            self.values.add(key if val is None else val)
            return self
        ch = key[idx]
        if self.to[ch] is None:
            self.to[ch] = Trie(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx + 1, val=val)

    def find_longest(self, key: bytes, idx: int = 0):
        node = self
        ch = key[idx]
        ret = None
        while node.to[ch] is not None:
            node = node.to[ch]
            idx += 1
            if node.values:
                ret = idx, node.values
            if idx == len(key):
                break
            ch = key[idx]
        if ret is None:
            raise ValueError(f"tokenizer could not encode byte at offset {idx}")
        return ret


class RWKVTokenizer:
    def __init__(self, vocab_file):
        self.idx2token = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                idx = int(line[: line.index(" ")])
                token = eval(line[line.index(" ") : line.rindex(" ")])
                token = token.encode("utf-8") if isinstance(token, str) else token
                self.idx2token[idx] = token

        self.root = Trie()
        for idx, token in self.idx2token.items():
            self.root.add(token, val=(token, int(idx)))

    def encode(self, text):
        src = text.encode("utf-8")
        idx = 0
        tokens = []
        while idx < len(src):
            next_idx, values = self.root.find_longest(src, idx)
            _, token_id = next(iter(values))
            tokens.append(token_id)
            idx = next_idx
        return tokens


def default_vocab_path():
    return Path(__file__).resolve().parents[3] / "data" / "rwkv_vocab_v20230424.txt"


def build_binidx(input_file, output_prefix, *, vocab_file, epochs, seed, dtype):
    input_path = Path(input_file)
    output_prefix = Path(output_prefix)
    vocab_file = Path(vocab_file)
    if not vocab_file.exists():
        raise FileNotFoundError(
            f"vocab file not found: {vocab_file}. "
            "Expected the repository copy at data/rwkv_vocab_v20230424.txt, "
            "or pass --vocab-file explicitly."
        )
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
            tokens = tokenizer.encode(text)
            tokens.append(0)
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
