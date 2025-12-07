#
# data.py
#
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import random_split, Subset, Dataset, DataLoader
from torchvision import transforms as tvtransforms
import datasets as ds

def split_train_val(dataset, val_fraction=0.1, seed=42):
    n = len(dataset)
    n_val = int(round(val_fraction * n))
    n_train = n - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)
    return train_ds, val_ds

class HF_MNIST(Dataset):
    def __init__(self, split="train", only_digit=None, cache_dir=None):
        if cache_dir == None:
            hf_dataset = ds.load_dataset("ylecun/mnist", split=split)
        else:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            hf_dataset = ds.load_dataset("ylecun/mnist", split=split, cache_dir=cache_dir)
        imgs, labels = hf_dataset["image"], hf_dataset["label"]
        if only_digit is not None:
            keep = [i for i, y in enumerate(labels) if y == only_digit]
            imgs   = [imgs[i] for i in keep]
            labels = [labels[i] for i in keep]
        self.imgs, self.labels = imgs, labels

    def __len__(self):
        return len(self.labels)

    def _to_tensor(self, img):
        if isinstance(img, list):
            arr = np.asarray(img, dtype=np.uint8).copy()
        else:
            arr = np.asarray(img, dtype=np.uint8).copy()
        t = torch.from_numpy(arr).unsqueeze(0).float() / 255.0
        return t

    def __getitem__(self, i):
        x = self._to_tensor(self.imgs[i])  # tensor (1,28,28)
        y = int(self.labels[i]) # python int
        return x, y
    
def make_mnist_loader(mnist_dataset, batch_size, shuffle=True, num_workers=0,
                      persistent_workers=False, prefetch_factor=2,
                      multiprocessing_context="spawn", pin_memory=False,
                      generator=None, worker_init_fn=None):
    return DataLoader(
        mnist_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        multiprocessing_context=multiprocessing_context,
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=worker_init_fn
    ), len(mnist_dataset)

def main():
    parser = argparse.ArgumentParser("MNIST Data Loader")
    parser.add_argument("--only-digit", type=int, default=1, help="-1 to use all digits")
    parser.add_argument("--split", type=str, default="train", help="train or test")
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    parser.add_argument("--cache-dir", type=str, default="./hf_cache", help="specify a directory where data sits")
    args = parser.parse_args()

    only_digit = None if args.only_digit == -1 else args.only_digit
    split = "train" if args.split == "train" else "test"

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    mnist_dataset = HF_MNIST(split=split, only_digit=only_digit, cache_dir=args.cache_dir)
    print("Total images", len(mnist_dataset))
    batch_size = 16
    mnist_loader, _ = make_mnist_loader(mnist_dataset=mnist_dataset, batch_size=batch_size, num_workers=args.num_workers)
    images, labels = next(iter(mnist_loader))
    print("Batch size", batch_size)
    print("Batch images", images.shape, "Labels", np.unique(labels))

if __name__ == "__main__":
    main()