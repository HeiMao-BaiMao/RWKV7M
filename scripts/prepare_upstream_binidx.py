"""Create a zero-copy, single-item binidx view for official RWKV-LM-V7."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import struct


MAGIC = b"MMIDIDX\x00\x00"
DTYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 4, 7: 8, 8: 2}


def read_dtype_code(index_path: Path) -> int:
    with index_path.open("rb") as handle:
        if handle.read(9) != MAGIC:
            raise ValueError(f"index file does not match MMIDIDX format: {index_path}")
        version = struct.unpack("<Q", handle.read(8))[0]
        if version != 1:
            raise ValueError(f"unsupported index version: {version}")
        dtype_code = struct.unpack("<B", handle.read(1))[0]
    if dtype_code not in DTYPE_SIZES:
        raise ValueError(f"unsupported dtype code: {dtype_code}")
    return dtype_code


def write_single_item_index(path: Path, dtype_code: int, token_count: int) -> None:
    if token_count >= 2**31:
        raise ValueError("single-item MMIDIDX size exceeds signed int32 capacity")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<B", dtype_code))
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<Q", 2))
        handle.write(struct.pack("<i", token_count))
        handle.write(struct.pack("<q", 0))
        handle.write(struct.pack("<qq", 0, 1))
    os.replace(temporary, path)


def prepare_single_item_view(source_prefix: Path, output_prefix: Path) -> int:
    source_bin = Path(f"{source_prefix}.bin")
    source_idx = Path(f"{source_prefix}.idx")
    output_bin = Path(f"{output_prefix}.bin")
    output_idx = Path(f"{output_prefix}.idx")
    if not source_bin.is_file() or not source_idx.is_file():
        raise FileNotFoundError(f"missing source binidx files for {source_prefix}")

    dtype_code = read_dtype_code(source_idx)
    byte_size = source_bin.stat().st_size
    dtype_size = DTYPE_SIZES[dtype_code]
    if byte_size % dtype_size:
        raise ValueError("bin file size is not divisible by its indexed dtype size")
    token_count = byte_size // dtype_size

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    if output_bin.exists():
        if not os.path.samefile(source_bin, output_bin):
            raise FileExistsError(f"output bin exists but is not the source file: {output_bin}")
    else:
        os.link(source_bin, output_bin)
    write_single_item_index(output_idx, dtype_code, token_count)
    return token_count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_prefix")
    parser.add_argument("output_prefix")
    args = parser.parse_args(argv)
    count = prepare_single_item_view(Path(args.source_prefix), Path(args.output_prefix))
    print(f"prepared official single-item binidx view: {args.output_prefix} ({count} tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
