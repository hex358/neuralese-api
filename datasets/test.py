# save_mnist_images.py
# pip install numpy pillow

import os
import numpy as np
from PIL import Image
from lset import DatasetIterable   # assuming lset is the file/module where DatasetIterable is defined

OUTPUT_DIR = "mnist_noisy_imgs"
DATASET_PATH = "mnist_noisy.ds"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = DatasetIterable(DATASET_PATH)

    for i, (x, y) in enumerate(dataset):
        # ensure x is 28x28
        arr = np.asarray(x).reshape(28, 28)

        # scale to 0–255 uint8 if needed
        if arr.dtype != np.uint8:
            arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)

        img = Image.fromarray(arr, mode="L")
        img.save(os.path.join(OUTPUT_DIR, f"img_{i:05d}_label_.png"))

        if i % 1000 == 0:
            print(f"saved {i} images...")

    print(f"done. total images: {len(dataset)}")

if __name__ == "__main__":
    main()
