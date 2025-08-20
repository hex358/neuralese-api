# pip install kagglehub lset numpy pillow
import os, glob, gzip, shutil, struct
from pathlib import Path
import numpy as np
import kagglehub
from lset import DatasetWriter
from PIL import Image
import concurrent.futures as cf

# ---------- Config ----------
DISTORT_IMAGES   = True
AUGMENT_VARIANTS = 10    # N variants per original (train only)
BASE_AUG_SEED    = None  # set int for reproducibility; None -> random each run

WORKERS          = max(1, (os.cpu_count() or 2) - 1)  # set 0/1 to disable pool
BATCH_SIZE       = 4096   # number of (idx,variant) tasks per pool submit

# Augmentation ranges
ROT_DEG_MAX = 20.0
SCALE_MIN, SCALE_MAX = 0.7, 1.4
SHIFT_PIX = 5
NOISE_STD_MAX = 0.05

# Debug export
DUMP_DEBUG_EVERY = 100
DUMP_DEBUG_DIR   = "digs"

# ---------- Staging helpers ----------
def _is_file(p: str) -> bool: return os.path.isfile(p)

def _find_file(root: str, patterns):
	candidates = []
	for pat in patterns:
		candidates += glob.glob(os.path.join(root, "**", pat), recursive=True)
	candidates = [c for c in candidates if _is_file(c)]
	if not candidates:
		raise FileNotFoundError(f"No files found for patterns {patterns} under {root}")
	candidates.sort(key=lambda p: (0 if p.endswith(".gz") else 1, len(p)))
	return candidates[0]

def _copy_or_decompress_to_cwd(src_path: str, dst_name_no_gz: str) -> Path:
	src = Path(src_path)
	dst = Path.cwd() / dst_name_no_gz
	if src.suffix == ".gz":
		with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
			shutil.copyfileobj(fin, fout)
	else:
		if not src.is_file():
			raise PermissionError(f"Source is not a readable file: {src}")
		shutil.copy2(src, dst)
	return dst

def stage_mnist_files_to_cwd(kaggle_root: str):
	files = {
		"train_images": _find_file(kaggle_root, ["train-images-idx3-ubyte.gz", "train-images-idx3-ubyte"]),
		"train_labels": _find_file(kaggle_root, ["train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte"]),
		"test_images":  _find_file(kaggle_root, ["t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte"]),
		"test_labels":  _find_file(kaggle_root, ["t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte"]),
	}
	return {
		"train_images": _copy_or_decompress_to_cwd(files["train_images"], "train-images-idx3-ubyte"),
		"train_labels": _copy_or_decompress_to_cwd(files["train_labels"], "train-labels-idx1-ubyte"),
		"test_images":  _copy_or_decompress_to_cwd(files["test_images"],  "t10k-images-idx3-ubyte"),
		"test_labels":  _copy_or_decompress_to_cwd(files["test_labels"],  "t10k-labels-idx1-ubyte"),
	}

# ---------- Cleanup ----------
def cleanup_staged(staged: dict):
	for _, p in staged.items():
		try:
			if p.parent == Path.cwd() and p.is_file():
				p.unlink()
				print(f"Removed staged file: {p.name}")
		except FileNotFoundError:
			pass

# ---------- Worker-side helpers (must be top-level for pickling) ----------
def _mix_seed(base_seed: int, idx: int, variant: int) -> int:
	# 64-bit xor-mix (constants are "nothing up my sleeve" primes/golden ratio)
	return int(np.uint64(base_seed)
	           ^ (np.uint64(idx)     * np.uint64(0x9E3779B97F4A7C15))
	           ^ (np.uint64(variant) * np.uint64(0xD2B74407B1CE6E93)))

def _augment_once(img_np: np.ndarray, rng: np.random.Generator) -> np.ndarray:
	pil = Image.fromarray((img_np * 255.0).astype(np.uint8), mode="L")

	# sample params; nudge away from identity
	for _ in range(3):
		scale = float(rng.uniform(SCALE_MIN, SCALE_MAX))
		angle = float(rng.uniform(-ROT_DEG_MAX, ROT_DEG_MAX))
		dx    = int(rng.integers(-SHIFT_PIX, SHIFT_PIX + 1))
		dy    = int(rng.integers(-SHIFT_PIX, SHIFT_PIX + 1))
		if abs(scale - 1.0) > 1e-3 or abs(angle) > 0.1 or dx != 0 or dy != 0:
			break

	new_size = max(16, int(round(28 * scale)))
	resized = pil.resize((new_size, new_size), resample=Image.BILINEAR)

	canvas = Image.new("L", (28, 28), 0)
	if new_size >= 28:
		l = (new_size - 28) // 2
		resized = resized.crop((l, l, l + 28, l + 28))
		canvas.paste(resized, (0, 0))
	else:
		off = ((28 - new_size) // 2, (28 - new_size) // 2)
		canvas.paste(resized, off)

	canvas = canvas.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=0)
	canvas = canvas.transform((28, 28), Image.AFFINE, (1, 0, dx, 0, 1, dy),
	                          resample=Image.BILINEAR, fillcolor=0)

	arr = np.asarray(canvas, dtype=np.float32) / 255.0
	sigma = float(rng.uniform(0.0, NOISE_STD_MAX))
	if sigma > 0.0:
		arr += rng.normal(0.0, sigma, arr.shape).astype(np.float32)
		arr = np.clip(arr, 0.0, 1.0)
	return arr

def _dump_png(img_01: np.ndarray, label: int, idx: int, split: str, out_dir: str, augmented: bool, variant: int):
	# Safe across processes due to unique filenames + exist_ok
	Path(out_dir).mkdir(parents=True, exist_ok=True)
	fn = f"{split}_{idx:05d}_var{variant}_y{label}_{'aug' if augmented else 'raw'}.png"
	Image.fromarray((img_01 * 255.0).astype(np.uint8), mode="L").save(Path(out_dir) / fn)

def _augment_worker(task):
	"""
	task: (img_2d_float32, label_int, idx_int, variant_int,
	       base_seed_int, dump_every_int, dump_dir, split_str)
	returns: (flat_float32, onehot_list)
	"""
	(img, label, idx, variant, base_seed, dump_every, dump_dir, split) = task
	rng = np.random.default_rng(_mix_seed(base_seed, idx, variant))
	img_aug = _augment_once(img, rng)

	if dump_every > 0 and (idx % dump_every == 0):
		_dump_png(img_aug, int(label), int(idx), split, dump_dir, augmented=True, variant=int(variant))

	onehot = [0.0] * 10
	onehot[int(label)] = 1.0
	return img_aug.flatten(), onehot

# ---------- IDX reader (parallel for augment=True) ----------
def iter_idx_images_labels_parallel(images_path: str, labels_path: str,
                                    augment: bool = False, variants: int = 1,
                                    dump_every: int = 0, dump_dir: str = "digs",
                                    split: str = "train",
                                    workers: int = 0, batch_size: int = 1024,
                                    base_seed: int | None = None):
	# Read labels
	with open(labels_path, "rb") as lf:
		magic, size = struct.unpack(">II", lf.read(8))
		if magic != 2049:
			raise ValueError(f"Labels magic mismatch, expected 2049, got {magic}")
		labels_raw = lf.read()
		if len(labels_raw) != size:
			raise ValueError("Label count mismatch")

	with open(images_path, "rb") as ifp:
		magic, size_imgs, rows, cols = struct.unpack(">IIII", ifp.read(16))
		if magic != 2051:
			raise ValueError(f"Images magic mismatch, expected 2051, got {magic}")
		if size_imgs != size:
			raise ValueError(f"Image/label count mismatch: {size_imgs} vs {size}")
		bytes_per_sample = rows * cols

		# Serial fallback (no augmentation or no workers)
		if not augment or workers <= 1 or variants <= 1:
			for i in range(size):
				label = int(labels_raw[i])
				img_b = ifp.read(bytes_per_sample)
				if len(img_b) != bytes_per_sample: break
				img = np.frombuffer(img_b, dtype=np.uint8).reshape(rows, cols).astype(np.float32) / 255.0

				if augment:
					for v in range(variants):
						seed = (base_seed if base_seed is not None else int.from_bytes(os.urandom(8), "little"))
						rng = np.random.default_rng(_mix_seed(seed, i, v))
						img_aug = _augment_once(img, rng)
						if dump_every > 0 and (i % dump_every == 0):
							_dump_png(img_aug, label, i, split, dump_dir, augmented=True, variant=v)
						onehot = [0.0]*10; onehot[label] = 1.0
						yield img_aug.flatten(), onehot
				else:
					if dump_every > 0 and (i % dump_every == 0):
						_dump_png(img, label, i, split, dump_dir, augmented=False, variant=0)
					onehot = [0.0]*10; onehot[label] = 1.0
					yield img.flatten(), onehot
			return

		# Parallel path (augment=True and variants>1 and workers>1)
		seed0 = base_seed if base_seed is not None else int.from_bytes(os.urandom(8), "little")
		tasks = []
		with cf.ProcessPoolExecutor(max_workers=workers) as ex:
			def flush():
				nonlocal tasks
				if not tasks:
					return
				# map preserves input order; chunksize reduces IPC overhead
				for flat, onehot in ex.map(_augment_worker, tasks, chunksize=32):
					yield flat, onehot
				tasks = []

			for i in range(size):
				label = int(labels_raw[i])
				img_b = ifp.read(bytes_per_sample)
				if len(img_b) != bytes_per_sample: break
				img = (np.frombuffer(img_b, dtype=np.uint8)
				       .reshape(rows, cols).astype(np.float32) / 255.0)

				# Enqueue all variants for this index
				for v in range(variants):
					tasks.append((img, label, i, v, seed0, dump_every, DUMP_DEBUG_DIR, split))

				# Flush in batches
				if len(tasks) >= batch_size:
					for out in flush():
						yield out

			# Final flush
			for out in flush():
				yield out

def write_lset(ds_path: str, images_path: str, labels_path: str,
               augment: bool = False, variants: int = 1,
               dump_every: int = 0, dump_dir: str = "digs", split: str = "train",
               workers: int = 0, batch_size: int = 1024, base_seed: int | None = None):
	try:
		os.remove(ds_path)
	except FileNotFoundError:
		pass
	writer = DatasetWriter(ds_path)
	writer.add_many(iter_idx_images_labels_parallel(
		images_path, labels_path,
		augment=augment, variants=variants,
		dump_every=dump_every, dump_dir=dump_dir, split=split,
		workers=workers, batch_size=batch_size, base_seed=base_seed))
	writer.close()

def main():
	# 1) Download & stage
	root = kagglehub.dataset_download("hojjatk/mnist-dataset")
	staged = stage_mnist_files_to_cwd(root)

	try:
		# 2) Train (parallel augmented)
		write_lset(
			"mnist_ds.ds",
			str(staged["train_images"]), str(staged["train_labels"]),
			augment=DISTORT_IMAGES, variants=AUGMENT_VARIANTS,
			dump_every=DUMP_DEBUG_EVERY, dump_dir=DUMP_DEBUG_DIR, split="train",
			workers=WORKERS, batch_size=BATCH_SIZE, base_seed=BASE_AUG_SEED
		)
		# 3) Test (serial, clean)
		write_lset(
			"mnist_test.ds",
			str(staged["test_images"]), str(staged["test_labels"]),
			augment=False, variants=1,
			dump_every=DUMP_DEBUG_EVERY, dump_dir=DUMP_DEBUG_DIR, split="test",
			workers=0, batch_size=BATCH_SIZE, base_seed=None
		)
		print(f"Done → mnist_ds.ds (train, {AUGMENT_VARIANTS} variants x augmented={DISTORT_IMAGES}, workers={WORKERS}); mnist_test.ds (test).")
	finally:
		# 4) Cleanup staged files
		cleanup_staged(staged)

if __name__ == "__main__":
	main()
