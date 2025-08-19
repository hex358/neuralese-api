# pip install lmdb numpy torch
import io, os, struct, lmdb, numpy as np
from typing import Iterable, Tuple, Optional

def _nd_to_bytes(a: np.ndarray) -> bytes:
	buf = io.BytesIO()
	np.save(buf, a, allow_pickle=False)
	return buf.getvalue()

def _bytes_to_nd(b: bytes) -> np.ndarray:
	return np.load(io.BytesIO(b), allow_pickle=False)

def _pack_pair(x: np.ndarray, y: np.ndarray) -> bytes:
	bx, by = _nd_to_bytes(x), _nd_to_bytes(y)
	return struct.pack(">I", len(bx)) + bx + struct.pack(">I", len(by)) + by

def _unpack_pair(b: bytes) -> Tuple[np.ndarray, np.ndarray]:
	lx = struct.unpack(">I", b[:4])[0]
	o  = 4
	bx = b[o:o+lx]
	o += lx
	ly = struct.unpack(">I", b[o:o+4])[0]
	o += 4
	by = b[o:o+ly]
	return _bytes_to_nd(bx), _bytes_to_nd(by)

class DatasetWriter:
	def __init__(self, path: str, map_size: int = 1 << 30):
		os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
		self.env = lmdb.open(path, map_size=map_size, subdir=False, lock=False, sync=True)
		self.n = 0

	def add_many(self, samples):
		n = self.n
		txn = self.env.begin(write=True)
		try:
			for (x, y) in samples:
				key = f"{n:016d}".encode("ascii")
				txn.put(key, _pack_pair(np.asarray(x), np.asarray(y)), append=True)
				n += 1
			txn.put(b"__len__", struct.pack(">Q", n))
			txn.commit()
		finally:
			self.n = n

	def close(self):
		self.env.close()

class DatasetIterable:
	def __init__(self, path: str):
		self.env = lmdb.open(path, readonly=True, lock=False, subdir=False, readahead=True, max_readers=512)

	def __iter__(self):
		with self.env.begin(write=False) as txn:
			with txn.cursor() as cur:
				for k, v in cur:
					if k == b"__len__":
						continue
					x, y = _unpack_pair(v)
					yield x, y

	def __len__(self) -> int:
		with self.env.begin(write=False) as txn:
			raw = txn.get(b"__len__")
			return 0 if raw is None else struct.unpack(">Q", raw)[0]

class DatasetIndexable:
	def __init__(self, path: str):
		self.env = lmdb.open(path, readonly=True, lock=False, subdir=False, readahead=True, max_readers=512)
		with self.env.begin() as txn:
			raw = txn.get(b"__len__")
			self._len = 0 if raw is None else struct.unpack(">Q", raw)[0]

	def __len__(self): return self._len

	def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
		if not (0 <= idx < self._len):
			raise IndexError
		key = f"{idx:016d}".encode("ascii")
		with self.env.begin() as txn:
			v = txn.get(key)
		if v is None:
			raise KeyError(f"missing sample {idx}")
		return _unpack_pair(v)
