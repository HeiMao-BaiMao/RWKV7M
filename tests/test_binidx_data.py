import numpy as np
import jax
import jax.numpy as jnp

from rwkv7m import create_binidx_dataset, tiny_config, train_binidx
from rwkv7m.data import (
    MMapIndexedDataset,
    MMapIndexedDatasetBuilder,
    data_file_path,
    find_magic_prime,
    index_file_path,
)
from rwkv7m.cli.bench_binidx import parse_args as parse_bench_args
from rwkv7m.cli.make_binidx import parse_args as parse_make_args


def write_demo_binidx(tmp_path, *, n_tokens=257, dtype=np.uint16):
    prefix = str(tmp_path / "demo")
    builder = MMapIndexedDatasetBuilder(data_file_path(prefix), dtype=dtype)
    builder.add_item(np.arange(n_tokens, dtype=dtype))
    builder.end_document()
    builder.finalize(index_file_path(prefix))
    return prefix


def test_mmap_indexed_dataset_reads_rwkv_binidx(tmp_path):
    prefix = write_demo_binidx(tmp_path)
    dataset = MMapIndexedDataset(prefix)
    try:
        assert dataset.data_size == 257
        assert dataset.dtype == np.uint16
        assert np.array_equal(dataset.get(0, offset=3, length=4), np.array([3, 4, 5, 6], dtype=np.uint16))
        assert np.array_equal(dataset.get_global(offset=3, length=4), np.array([3, 4, 5, 6], dtype=np.uint16))
    finally:
        dataset.close()


def test_mmap_indexed_dataset_reads_global_span_across_documents(tmp_path):
    prefix = str(tmp_path / "multi")
    builder = MMapIndexedDatasetBuilder(data_file_path(prefix), dtype=np.uint16)
    builder.add_item(np.array([0, 1, 2], dtype=np.uint16))
    builder.end_document()
    builder.add_item(np.array([3, 4, 5], dtype=np.uint16))
    builder.end_document()
    builder.finalize(index_file_path(prefix))

    dataset = MMapIndexedDataset(prefix)
    try:
        assert np.array_equal(
            dataset.get_global(offset=2, length=3),
            np.array([2, 3, 4], dtype=np.uint16),
        )
    finally:
        dataset.close()


def test_find_magic_prime_matches_rwkv_constraints():
    magic_prime = find_magic_prime(data_size=257, ctx_len=4)
    assert magic_prime == 59
    assert magic_prime % 3 == 2


def test_find_magic_prime_avoids_exact_multiple_overrun():
    magic_prime = find_magic_prime(data_size=240, ctx_len=4)
    assert magic_prime == 59
    assert magic_prime * 4 + 1 <= 240


def test_binidx_batch_dataset_returns_train_step_batch(tmp_path):
    prefix = write_demo_binidx(tmp_path)
    dataset = create_binidx_dataset(prefix, ctx_len=4, batch_size=2)
    try:
        batch = dataset.get_batch(0)
        assert batch["input_ids"].shape == (2, 4)
        assert batch["target_ids"].shape == (2, 4)
        assert batch["mask"].shape == (2, 4)
        assert batch["input_ids"].dtype == jnp.int32
        assert jnp.all(batch["target_ids"] == batch["input_ids"] + 1)
    finally:
        dataset.close()


def test_binidx_batch_dataset_supports_multi_document_global_sampling(tmp_path):
    prefix = str(tmp_path / "multi")
    builder = MMapIndexedDatasetBuilder(data_file_path(prefix), dtype=np.uint16)
    builder.add_item(np.arange(128, dtype=np.uint16))
    builder.end_document()
    builder.add_item(np.arange(128, 257, dtype=np.uint16))
    builder.end_document()
    builder.finalize(index_file_path(prefix))

    dataset = create_binidx_dataset(prefix, ctx_len=4, batch_size=2)
    try:
        batch = dataset.get_batch(0)
        assert batch["input_ids"].shape == (2, 4)
        assert jnp.all(batch["target_ids"] == batch["input_ids"] + 1)
    finally:
        dataset.close()


def test_train_binidx_runs_one_step(tmp_path):
    prefix = write_demo_binidx(tmp_path, n_tokens=257)
    cfg = tiny_config(vocab_size=512, d_model=32, n_layers=2, n_heads=2, head_size=16)
    losses, _, _ = train_binidx(
        jax.random.PRNGKey(0),
        cfg,
        prefix,
        ctx_len=4,
        batch_size=1,
        num_steps=1,
    )
    assert len(losses) == 1
    assert np.isfinite(losses[0])


def test_new_cli_parsers_import():
    make_args = parse_make_args(["data/demo.jsonl"])
    assert make_args.vocab_file.endswith("data\\rwkv_vocab_v20230424.txt") or make_args.vocab_file.endswith("data/rwkv_vocab_v20230424.txt")
    bench_args = parse_bench_args(["--data-file", "data/demo"])
    assert bench_args.variants == ["baseline", "screening", "read_write"]
