import os
import struct
from functools import lru_cache
from itertools import accumulate

import numpy as np


DTYPES = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}


def dtype_code(dtype):
    dtype = np.dtype(dtype).type
    for code, candidate in DTYPES.items():
        if np.dtype(candidate).type == dtype:
            return code
    raise ValueError(f"unsupported binidx dtype: {dtype}")


def index_file_path(prefix_path):
    return prefix_path + ".idx"


def data_file_path(prefix_path):
    return prefix_path + ".bin"


class MMapIndexedDataset:
    """Reader for RWKV-LM / Megatron-style .bin/.idx token datasets."""

    class Index:
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @classmethod
        def writer(cls, path, dtype):
            class Writer:
                def __enter__(self):
                    self._file = open(path, "wb")
                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack("<Q", 1))
                    self._file.write(struct.pack("<B", dtype_code(dtype)))
                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = np.dtype(dtype).itemsize
                    address = 0
                    pointers = []
                    for size in sizes:
                        pointers.append(address)
                        address += int(size) * dtype_size
                    return pointers

                def write(self, sizes, doc_idx):
                    pointers = self._get_pointers(sizes)
                    self._file.write(struct.pack("<Q", len(sizes)))
                    self._file.write(struct.pack("<Q", len(doc_idx)))
                    self._file.write(np.asarray(sizes, dtype=np.int32).tobytes(order="C"))
                    self._file.write(np.asarray(pointers, dtype=np.int64).tobytes(order="C"))
                    self._file.write(np.asarray(doc_idx, dtype=np.int64).tobytes(order="C"))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return Writer()

        def __init__(self, path):
            with open(path, "rb") as stream:
                magic = stream.read(9)
                if magic != self._HDR_MAGIC:
                    raise ValueError("index file does not match MMIDIDX format")
                version = struct.unpack("<Q", stream.read(8))[0]
                if version != 1:
                    raise ValueError(f"unsupported index version: {version}")
                dtype_id = struct.unpack("<B", stream.read(1))[0]
                if dtype_id not in DTYPES:
                    raise ValueError(f"unsupported dtype code in index: {dtype_id}")
                self._dtype = DTYPES[dtype_id]
                self._dtype_size = np.dtype(self._dtype).itemsize
                self._len = struct.unpack("<Q", stream.read(8))[0]
                self._doc_count = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            self._sizes = np.frombuffer(
                self._bin_buffer,
                dtype=np.int32,
                count=self._len,
                offset=offset,
            )
            self._pointers = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._len,
                offset=offset + self._sizes.nbytes,
            )
            self._doc_idx = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._doc_count,
                offset=offset + self._sizes.nbytes + self._pointers.nbytes,
            )

        def close(self):
            self._sizes = None
            self._pointers = None
            self._doc_idx = None
            bin_buffer = getattr(self, "_bin_buffer", None)
            if bin_buffer is not None:
                bin_buffer.release()
                self._bin_buffer = None
            mmap = getattr(getattr(self, "_bin_buffer_mmap", None), "_mmap", None)
            if mmap is not None:
                mmap.close()
                self._bin_buffer_mmap = None

        def __del__(self):
            try:
                self.close()
            except Exception:
                pass

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path):
        self._path = path
        self._index = self.Index(index_file_path(path))
        self._bin_buffer_mmap = np.memmap(data_file_path(path), mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def close(self):
        bin_buffer = getattr(self, "_bin_buffer", None)
        if bin_buffer is not None:
            bin_buffer.release()
            self._bin_buffer = None
        mmap = getattr(getattr(self, "_bin_buffer_mmap", None), "_mmap", None)
        if mmap is not None:
            mmap.close()
            self._bin_buffer_mmap = None
        if getattr(self, "_index", None) is not None:
            self._index.close()
            self._index = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            return np.frombuffer(
                self._bin_buffer,
                dtype=self._index.dtype,
                count=size,
                offset=ptr,
            )
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("binidx slices must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = int(sum(sizes))
            array = np.frombuffer(
                self._bin_buffer,
                dtype=self._index.dtype,
                count=total_size,
                offset=ptr,
            )
            return np.split(array, offsets[:-1])
        raise TypeError(f"unsupported index type: {type(idx)!r}")

    def get(self, idx, offset=0, length=None):
        ptr, size = self._index[idx]
        if length is None:
            length = int(size) - offset
        if offset < 0 or length < 0 or offset + length > int(size):
            raise IndexError("requested binidx span is outside item bounds")
        ptr += offset * np.dtype(self._index.dtype).itemsize
        return np.frombuffer(
            self._bin_buffer,
            dtype=self._index.dtype,
            count=length,
            offset=ptr,
        )

    def get_global(self, offset=0, length=None):
        if length is None:
            length = self.data_size - int(offset)
        offset = int(offset)
        length = int(length)
        if offset < 0 or length < 0 or offset + length > self.data_size:
            raise IndexError("requested global binidx span is outside data buffer")
        byte_offset = offset * np.dtype(self._index.dtype).itemsize
        return np.frombuffer(
            self._bin_buffer,
            dtype=self._index.dtype,
            count=length,
            offset=byte_offset,
        )

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    @property
    def dtype(self):
        return self._index.dtype

    @property
    def data_size(self):
        return len(self._bin_buffer) // np.dtype(self._index.dtype).itemsize

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))


class MMapIndexedDatasetBuilder:
    def __init__(self, out_file, dtype=np.uint16):
        self._dtype = np.dtype(dtype).type
        self._data_file = open(out_file, "wb")
        self._sizes = []
        self._doc_idx = [0]

    def add_item(self, array):
        array = np.asarray(array, dtype=self._dtype)
        self._data_file.write(array.tobytes(order="C"))
        self._sizes.append(len(array))

    def end_document(self):
        self._doc_idx.append(len(self._sizes))

    def finalize(self, index_file):
        self._data_file.close()
        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)
