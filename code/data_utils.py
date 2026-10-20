"""Data loading and preprocessing utilities for CIFAR-10N challenge.

Strict Data Isolation Rules:
- NoisyDataset: strictly exposes noisy_label; no access to clean_label.
- SyntheticDataset: strictly exposes synthetic_label; no access to clean_label.
- CleanDataset: strictly reserved for Task 1 clean reference training and
  explicitly permitted offline evaluation (Task 1 CM, Task 2 dynamics, Task 3 AUROC check).
- All image normalizations are derived exclusively from the provided 45,000 training images.
"""

import os
from typing import Literal, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

# Normalization constants calculated strictly from provided train_images.npy on train_idx (45,000 images)
# Calculated without downloading any external datasets.
DATA_MEAN = (0.49153, 0.48230, 0.44667)
DATA_STD = (0.24701, 0.24345, 0.26157)


def find_data_dir(data_dir: Optional[str] = None) -> str:
    """Locate data directory across local and Google Colab environments."""
    if data_dir is not None and os.path.isdir(data_dir):
        return os.path.abspath(data_dir)

    candidates = [
        os.path.join(".", "data"),
        os.path.join("..", "data"),
        "/content/data",
        "/content/drive/MyDrive/data",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)

    raise FileNotFoundError(
        "Could not find 'data/' directory. Please specify data_dir explicitly."
    )


class NoisyDataset(Dataset):
    """Dataset strictly exposing noisy labels for training and model selection.

    It has NO access to clean labels or DataFrame columns beyond noisy_label.
    """

    def __init__(self, images: np.ndarray, noisy_labels: np.ndarray, indices: np.ndarray, transform=None):
        assert len(images) == len(noisy_labels) == len(indices)
        self.images = images  # uint8, shape (N, 32, 32, 3)
        self.labels = np.asarray(noisy_labels, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        img = self.images[idx]
        label = int(self.labels[idx])
        original_idx = int(self.indices[idx])

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, label, original_idx


class CleanDataset(Dataset):
    """Dataset exposing clean labels.

    STRICT RESTRICTION:
    Permitted ONLY for:
      1. Training Task 1 clean-reference baseline.
      2. Explicitly permitted offline evaluations (Task 1 CM, Task 2 dynamics logging, Task 3 AUROC).
    Must NEVER be used for training, sample filtering, or model selection in Tasks 2-5.
    """

    def __init__(self, images: np.ndarray, clean_labels: np.ndarray, indices: np.ndarray, transform=None):
        assert len(images) == len(clean_labels) == len(indices)
        self.images = images
        self.labels = np.asarray(clean_labels, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        img = self.images[idx]
        label = int(self.labels[idx])
        original_idx = int(self.indices[idx])

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, label, original_idx


class SyntheticDataset(Dataset):
    """Dataset exposing synthetic random noise labels for Task 5.

    It has NO access to clean labels.
    """

    def __init__(self, images: np.ndarray, synthetic_labels: np.ndarray, indices: np.ndarray, transform=None):
        assert len(images) == len(synthetic_labels) == len(indices)
        self.images = images
        self.labels = np.asarray(synthetic_labels, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        img = self.images[idx]
        label = int(self.labels[idx])
        original_idx = int(self.indices[idx])

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, label, original_idx


class TestDataset(Dataset):
    """Dataset for test images.

    Returns (transformed_image, image_id) where image_id is in 0..9999.
    """

    def __init__(self, test_images: np.ndarray, transform=None):
        self.images = test_images  # shape (10000, 32, 32, 3), uint8
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(idx)


def get_transforms():
    """Standard augmentations and normalization.

    Transforms applied:
    - Train: RandomCrop(32, padding=4), RandomHorizontalFlip(p=0.5), ToTensor(), Normalize()
    - Eval/Test: ToTensor(), Normalize()
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(DATA_MEAN, DATA_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(DATA_MEAN, DATA_STD),
    ])

    return train_transform, eval_transform


def load_raw_data(data_dir: Optional[str] = None):
    """Load raw arrays and index lists from disk."""
    base = find_data_dir(data_dir)

    train_images = np.load(os.path.join(base, "train_images.npy"))
    test_images = np.load(os.path.join(base, "test_images.npy"))
    labels_df = pd.read_csv(os.path.join(base, "labels.csv"))

    with open(os.path.join(base, "train_idx.txt"), encoding="utf-8") as f:
        train_idx = np.array([int(line.strip()) for line in f if line.strip()], dtype=np.int64)

    with open(os.path.join(base, "val_idx.txt"), encoding="utf-8") as f:
        val_idx = np.array([int(line.strip()) for line in f if line.strip()], dtype=np.int64)

    return train_images, test_images, labels_df, train_idx, val_idx


def get_loaders(
    data_dir: Optional[str] = None,
    label_type: Literal["noisy", "clean", "synthetic"] = "noisy",
    batch_size: int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Factory creating train_loader, val_loader, test_loader with strictly isolated labels.

    Args:
        data_dir: path to directory with data files (auto-detected if None).
        label_type: 'noisy' (Tasks 1-4), 'clean' (Task 1 reference only), or 'synthetic' (Task 5).
        batch_size: default 128.
        num_workers: DataLoader worker count.
    """
    train_images, test_images, labels_df, train_idx, val_idx = load_raw_data(data_dir)
    train_transform, eval_transform = get_transforms()

    if label_type == "noisy":
        train_labels = labels_df.loc[train_idx, "noisy_label"].to_numpy()
        val_labels = labels_df.loc[val_idx, "noisy_label"].to_numpy()
        train_ds = NoisyDataset(train_images[train_idx], train_labels, train_idx, train_transform)
        val_ds = NoisyDataset(train_images[val_idx], val_labels, val_idx, eval_transform)

    elif label_type == "synthetic":
        train_labels = labels_df.loc[train_idx, "synthetic_label"].to_numpy()
        val_labels = labels_df.loc[val_idx, "synthetic_label"].to_numpy()
        train_ds = SyntheticDataset(train_images[train_idx], train_labels, train_idx, train_transform)
        val_ds = SyntheticDataset(train_images[val_idx], val_labels, val_idx, eval_transform)

    elif label_type == "clean":
        train_labels = labels_df.loc[train_idx, "clean_label"].to_numpy()
        val_labels = labels_df.loc[val_idx, "clean_label"].to_numpy()
        train_ds = CleanDataset(train_images[train_idx], train_labels, train_idx, train_transform)
        val_ds = CleanDataset(train_images[val_idx], val_labels, val_idx, eval_transform)

    else:
        raise ValueError(f"Unsupported label_type: {label_type}")

    test_ds = TestDataset(test_images, eval_transform)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


def get_task2_offline_eval_data(data_dir: Optional[str] = None):
    """Load offline reference labels strictly for Task 2 dynamics logging and Task 3 AUROC check.

    CRITICAL RESTRICTION:
    The returned clean labels and clean validation loader are for offline metrics ONLY.
    They must never be passed to the optimizer, loss function, or checkpoint selector.
    """
    train_images, _, labels_df, train_idx, val_idx = load_raw_data(data_dir)
    _, eval_transform = get_transforms()

    train_clean_labels = labels_df.loc[train_idx, "clean_label"].to_numpy()
    train_noisy_labels = labels_df.loc[train_idx, "noisy_label"].to_numpy()

    val_clean_labels = labels_df.loc[val_idx, "clean_label"].to_numpy()
    val_clean_ds = CleanDataset(train_images[val_idx], val_clean_labels, val_idx, eval_transform)
    clean_val_loader = DataLoader(
        val_clean_ds, batch_size=128, shuffle=False, pin_memory=torch.cuda.is_available()
    )

    return train_noisy_labels, train_clean_labels, train_idx, clean_val_loader
