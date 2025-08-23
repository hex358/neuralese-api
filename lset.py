# pip install lmdb numpy torch
import io, os, struct, lmdb, numpy as np
from typing import Iterable, Tuple

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
	o = 0
	lx = struct.unpack(">I", b[o:o+4])[0]; o += 4
	bx = b[o:o+lx]; o += lx
	ly = struct.unpack(">I", b[o:o+4])[0]; o += 4
	by = b[o:o+ly]
	return _bytes_to_nd(bx), _bytes_to_nd(by)

def _env_used_bytes(env: lmdb.Environment) -> int:
	"""Bytes actually used by the DB (page_count * page_size)."""
	info = env.info()
	stat = env.stat()
	psize = stat["psize"]
	last_pgno = info["last_pgno"]  # 0-based index of last used page
	return (last_pgno + 1) * psize

def _ceil_to(n: int, k: int) -> int:
	return ((n + k - 1) // k) * k

class DatasetWriter:
	def __init__(
		self,
		path: str,
		map_size: int = 64 << 20,        # start small (64MB); will auto-grow
		auto_grow: bool = True,
		growth_factor: float = 2.0,
		commit_every: int = 2048,        # periodic commits for safety
		shrink_on_close: bool = True,
		close_headroom_pages: int = 8     # a few extra pages so reopen can append
	):
		"""
		path: single-file DB (subdir=False). map_size is just a start; we'll grow as needed.
		"""
		os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
		self.path = path
		self.env = lmdb.open(
			path,
			map_size=map_size,
			subdir=False,
			lock=False,
			sync=True,
			max_dbs=1,
			readahead=False,   # writers don't need this
			writemap=False
		)
		self.n = 0
		self.auto_grow = auto_grow
		self.growth_factor = max(1.2, float(growth_factor))
		self.commit_every = max(1, int(commit_every))
		self.shrink_on_close = shrink_on_close
		self.close_headroom_pages = max(2, int(close_headroom_pages))

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc, tb):
		self.close()

	def _bump_mapsize(self):
		info = self.env.info()
		current = info["map_size"]
		new_size = int(current * self.growth_factor)
		self.env.set_mapsize(new_size)

	def add_many(self, samples: Iterable[Tuple[np.ndarray, np.ndarray]]):
		"""
		Robust writer:
		- periodic commits
		- auto-grow on MapFullError and retry the current sample
		- updates __len__ after each commit
		"""
		n = self.n
		count_since_commit = 0
		txn = self.env.begin(write=True)
		try:
			for x, y in samples:
				while True:
					try:
						key = f"{n:016d}".encode("ascii")
						txn.put(key, _pack_pair(np.asarray(x), np.asarray(y)), append=True)
						n += 1
						count_since_commit += 1
						if count_since_commit >= self.commit_every:
							txn.put(b"__len__", struct.pack(">Q", n))
							txn.commit()
							txn = self.env.begin(write=True)
							count_since_commit = 0
						break  # success; move to next sample
					except lmdb.MapFullError:
						# roll back current txn, grow, and retry same (x, y)
						try:
							txn.abort()
						except Exception:
							pass
						if not self.auto_grow:
							raise
						self._bump_mapsize()
						txn = self.env.begin(write=True)

			# final commit
			txn.put(b"__len__", struct.pack(">Q", n))
			txn.commit()
		finally:
			self.n = n

	def _shrink_to_fit(self):
		"""
		Shrinks mapsize to fit the actual used bytes (+ headroom pages).
		Uses LMDB's set_mapsize so meta pages match the new size.
		"""
		# must have no active write txn here
		used = _env_used_bytes(self.env)
		psize = self.env.stat()["psize"]
		target = _ceil_to(used + self.close_headroom_pages * psize, psize)
		# Never shrink below 1MB to avoid over-aggressive truncation
		target = max(target, 1 << 20)
		self.env.set_mapsize(target)
		self.env.sync()

	def close(self):
		# shrink first (while env is open), then close
		if self.shrink_on_close:
			try:
				with self.env.begin(write=False):
					pass  # ensure no write txn open
				self._shrink_to_fit()
			except Exception:
				# don't block close if shrink fails (e.g., exotic FS)
				pass
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
					yield _unpack_pair(v)

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
