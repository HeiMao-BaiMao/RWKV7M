import importlib.util
from pathlib import Path
import struct

import numpy as np

from rwkv7m.data.binidx import MMapIndexedDataset


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_upstream_binidx.py"
SPEC = importlib.util.spec_from_file_location("prepare_upstream_binidx", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
prepare_single_item_view = MODULE.prepare_single_item_view


def test_prepare_single_item_view_reuses_bin_and_flattens_index(tmp_path: Path):
    source = tmp_path / "source.v1"
    output = tmp_path / "flat.v1"
    np.arange(10, dtype=np.uint16).tofile(Path(f"{source}.bin"))
    with MMapIndexedDataset.Index.writer(Path(f"{source}.idx"), np.uint16) as writer:
        writer.write([4, 6], [0, 2])

    assert prepare_single_item_view(source, output) == 10
    assert Path(f"{output}.bin").samefile(Path(f"{source}.bin"))

    with Path(f"{output}.idx").open("rb") as handle:
        assert handle.read(9) == b"MMIDIDX\x00\x00"
        assert struct.unpack("<Q", handle.read(8))[0] == 1
        assert struct.unpack("<B", handle.read(1))[0] == 8
        assert struct.unpack("<Q", handle.read(8))[0] == 1
        assert struct.unpack("<Q", handle.read(8))[0] == 2
        assert struct.unpack("<i", handle.read(4))[0] == 10

    dataset = MMapIndexedDataset(str(output))
    try:
        np.testing.assert_array_equal(dataset[0], np.arange(10, dtype=np.uint16))
    finally:
        dataset.close()
