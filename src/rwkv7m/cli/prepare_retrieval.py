"""Create a document-aligned delayed key/value retrieval binidx dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..data import (
    MMapIndexedDatasetBuilder,
    data_file_path,
    index_file_path,
)


FILLER_TOKEN = 1
STORE_TOKEN = 2
QUERY_TOKEN = 3
ANSWER_MARKER_TOKEN = 4


def build_retrieval_dataset(
    output_prefix,
    *,
    documents,
    ctx_len,
    chunks_per_document,
    vocab_size,
    key_count,
    distractors,
    seed,
):
    if documents <= 0:
        raise ValueError("documents must be positive")
    if ctx_len < 16:
        raise ValueError("ctx_len must be at least 16")
    if chunks_per_document < 2:
        raise ValueError("chunks_per_document must be at least 2")
    if key_count <= distractors:
        raise ValueError("key_count must be greater than distractors")
    key_base = 16
    value_base = key_base + key_count
    if value_base + key_count > vocab_size:
        raise ValueError("vocab_size is too small for disjoint key/value tokens")

    output_prefix = str(output_prefix)
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dtype = np.uint16 if vocab_size <= 65536 else np.int32
    builder = MMapIndexedDatasetBuilder(
        data_file_path(output_prefix),
        dtype=dtype,
    )
    document_length = ctx_len * chunks_per_document + 1
    for _ in range(documents):
        tokens = np.full(document_length, FILLER_TOKEN, dtype=dtype)
        selected = rng.choice(key_count, size=distractors + 1, replace=False)
        query_key = int(selected[0])
        query_value = int(rng.integers(0, key_count))
        tokens[:3] = (STORE_TOKEN, key_base + query_key, value_base + query_value)

        available_positions = np.arange(8, (chunks_per_document - 1) * ctx_len - 3, 4)
        if distractors > len(available_positions):
            raise ValueError("too many distractors for the requested document length")
        for position, distractor_key in zip(
            available_positions[:distractors],
            selected[1:],
            strict=True,
        ):
            distractor_value = int(rng.integers(0, key_count))
            tokens[position : position + 3] = (
                STORE_TOKEN,
                key_base + int(distractor_key),
                value_base + distractor_value,
            )

        query_position = (chunks_per_document - 1) * ctx_len + ctx_len // 2
        tokens[query_position : query_position + 4] = (
            QUERY_TOKEN,
            key_base + query_key,
            ANSWER_MARKER_TOKEN,
            value_base + query_value,
        )
        builder.add_item(tokens)
        builder.end_document()
    builder.finalize(index_file_path(output_prefix))

    metadata = {
        "format": "rwkv7m-delayed-kv-v1",
        "documents": int(documents),
        "ctx_len": int(ctx_len),
        "chunks_per_document": int(chunks_per_document),
        "tokens_per_document": int(document_length),
        "vocab_size": int(vocab_size),
        "key_count": int(key_count),
        "distractors": int(distractors),
        "seed": int(seed),
        "tokens": {
            "filler": FILLER_TOKEN,
            "store": STORE_TOKEN,
            "query": QUERY_TOKEN,
            "answer_marker": ANSWER_MARKER_TOKEN,
            "key_base": key_base,
            "value_base": value_base,
        },
        "training_contract": {
            "sampling_mode": "document",
            "carry_state": False,
            "ctx_len": int(document_length - 1),
            "loss_mask_after_token": ANSWER_MARKER_TOKEN,
        },
        "streaming_evaluation_contract": {
            "sampling_mode": "document_sequential",
            "carry_state": True,
            "ctx_len": int(ctx_len),
            "loss_mask_after_token": ANSWER_MARKER_TOKEN,
        },
    }
    metadata_path = Path(output_prefix + ".retrieval.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--documents", type=int, default=1024)
    parser.add_argument("--ctx-len", type=int, default=128)
    parser.add_argument("--chunks-per-document", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--key-count", type=int, default=256)
    parser.add_argument("--distractors", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    metadata = build_retrieval_dataset(
        args.output_prefix,
        documents=args.documents,
        ctx_len=args.ctx_len,
        chunks_per_document=args.chunks_per_document,
        vocab_size=args.vocab_size,
        key_count=args.key_count,
        distractors=args.distractors,
        seed=args.seed,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
