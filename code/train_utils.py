"""Training and evaluation primitives for CIFAR-10N ResNet-18.

Strict Restraints:
- Preserves the exact ResNet-18 architecture defined in model.py.
- Uses fixed hyperparameter setup across all methods:
    SGD(momentum=0.9, weight_decay=5e-4), CosineAnnealingLR(T_max=60, eta_min=1e-4), batch_size=128.
- Model selection / early stopping is strictly decoupled:
    Early stopping observes only val_acc_noisy (Tasks 1-4) or val_acc_synthetic (Task 5).
- Task 2 offline metrics (train_acc_correct, train_acc_mislabeled, val_acc_clean) are logged
    passively and NEVER affect optimization or checkpoint selection.
"""

import csv
import os
import random
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to sys.path if needed to locate model.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, PARENT_DIR, "/content"]:
    if p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)

from model import ResNet18

# Standardized hyperparameter configuration
SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 60
INITIAL_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
MIN_LR = 1e-4
NUM_CLASSES = 10


def set_seed(seed: int = SEED):
    """Ensure strict reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Instantiate standard CIFAR ResNet-18 from model.py."""
    return ResNet18(num_classes=num_classes)


def get_optimizer_and_scheduler(
    model: nn.Module,
    epochs: int = MAX_EPOCHS,
    lr: float = INITIAL_LR,
    weight_decay: float = WEIGHT_DECAY,
):
    """Instantiate standard SGD optimizer and CosineAnnealingLR scheduler."""
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=MOMENTUM,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=MIN_LR,
    )
    return optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, Dict[int, int]]:
    """Train model for one epoch.

    Returns:
        avg_loss: average cross-entropy loss.
        accuracy: top-1 training accuracy (0.0 to 100.0).
        predictions_by_idx: dict mapping sample_index -> predicted_class (for offline dynamics).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    preds_dict: Dict[int, int] = {}

    for images, labels, indices in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        preds_cpu = preds.detach().cpu().numpy()
        idx_cpu = indices.numpy()
        for idx_val, pred_val in zip(idx_cpu, preds_cpu):
            preds_dict[int(idx_val)] = int(pred_val)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, preds_dict


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> Tuple[float, float]:
    """Evaluate model on a dataset.

    Returns:
        (avg_loss, accuracy_percentage)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        images = batch[0].to(device)
        labels = batch[1].to(device)

        outputs = model(images)
        if criterion is not None:
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = (total_loss / total) if (criterion is not None and total > 0) else 0.0
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def predict_test(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Dict[int, int]:
    """Generate predictions for unlabelled test images.

    Returns:
        dict mapping id (0..9999) -> predicted_class (0..9)
    """
    model.eval()
    predictions: Dict[int, int] = {}

    for images, ids in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)

        preds_cpu = preds.cpu().numpy()
        ids_cpu = ids.numpy()

        for pid, pred in zip(ids_cpu, preds_cpu):
            predictions[int(pid)] = int(pred)

    return predictions


def save_predictions_csv(predictions: Dict[int, int], output_path: str):
    """Save predictions in official format: id,pred with 10,000 rows sorted by id."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "pred"])
        for pid in sorted(predictions.keys()):
            writer.writerow([pid, predictions[pid]])


class NoisyValEarlyStopping:
    """Tracks best model checkpoint based strictly on validation accuracy.

    STRICT GUARANTEE:
    This tracker only receives a single scalar: noisy validation accuracy.
    It has no reference or access to clean labels or offline dynamics.
    """

    def __init__(self):
        self.best_acc: float = -1.0
        self.best_epoch: Optional[int] = None
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    def update(self, epoch: int, val_acc_noisy: float, model: nn.Module) -> bool:
        """Update tracker if new best noisy validation accuracy is reached.

        Returns True if a new best is found, False otherwise.
        """
        if val_acc_noisy > self.best_acc:
            self.best_acc = val_acc_noisy
            self.best_epoch = epoch
            self.best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            return True
        return False

    def load_best_weights(self, model: nn.Module):
        """Load best recorded weights into the model."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)


class Task2DynamicsTracker:
    """Offline observer for logging Task 2 dynamics curves.

    STRICT GUARANTEE:
    Calculates offline metrics for logging to predictions/t2_dynamics.csv.
    These metrics are written to disk and NEVER passed to the optimizer or early stopping.
    """

    def __init__(
        self,
        train_noisy_labels: np.ndarray,
        train_clean_labels: np.ndarray,
        train_indices: np.ndarray,
        clean_val_loader: DataLoader,
        mislabeled_target: str = "clean",
    ):
        self.train_indices = train_indices
        self.noisy_labels = train_noisy_labels
        self.clean_labels = train_clean_labels
        self.clean_val_loader = clean_val_loader
        self.mislabeled_target = mislabeled_target

        # Precompute masks over training indices
        self.is_correct_label = (self.noisy_labels == self.clean_labels)
        self.is_mislabeled = ~self.is_correct_label

        # Map index -> position in array
        self.idx_to_pos = {int(idx): pos for pos, idx in enumerate(self.train_indices)}
        self.records: List[Dict[str, float]] = []

    def load_records(self, records: List[Dict]):
        """Restore historical records when resuming training from checkpoint."""
        self.records = [dict(r) for r in records]

    def record_epoch(
        self,
        epoch: int,
        train_predictions: Dict[int, int],
        val_acc_noisy: float,
        model: nn.Module,
        device: torch.device,
    ) -> Dict[str, float]:
        """Compute the 4 required dynamics metrics for the given epoch."""
        # Align predictions with training positions
        pred_arr = np.zeros(len(self.train_indices), dtype=np.int64)
        for idx_val, pos in self.idx_to_pos.items():
            pred_arr[pos] = train_predictions.get(idx_val, -1)

        # Accuracy on correctly labeled subset
        correct_subset = self.is_correct_label
        acc_correct = (
            100.0 * (pred_arr[correct_subset] == self.clean_labels[correct_subset]).mean()
            if correct_subset.any() else 0.0
        )

        # Accuracy on mislabeled subset
        # By default, measured against clean labels as originally implemented in train_utils.py.
        # If mislabeled_target == "noisy", measured against assigned noisy labels to track memorization rate.
        mislabeled_subset = self.is_mislabeled
        if self.mislabeled_target == "noisy":
            acc_mislabeled = (
                100.0 * (pred_arr[mislabeled_subset] == self.noisy_labels[mislabeled_subset]).mean()
                if mislabeled_subset.any() else 0.0
            )
        else:
            acc_mislabeled = (
                100.0 * (pred_arr[mislabeled_subset] == self.clean_labels[mislabeled_subset]).mean()
                if mislabeled_subset.any() else 0.0
            )

        # Validation accuracy against clean labels
        _, val_acc_clean = evaluate(model, self.clean_val_loader, device)

        row = {
            "epoch": epoch,
            "train_acc_correct": round(float(acc_correct), 4),
            "train_acc_mislabeled": round(float(acc_mislabeled), 4),
            "val_acc_noisy": round(float(val_acc_noisy), 4),
            "val_acc_clean": round(float(val_acc_clean), 4),
        }
        self.records.append(row)
        return row

    def save_csv(self, output_path: str):
        """Save t2_dynamics.csv with exact required header."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        header = ["epoch", "train_acc_correct", "train_acc_mislabeled", "val_acc_noisy", "val_acc_clean"]
        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for r in self.records:
                writer.writerow(r)


def evaluate_noise_scores_offline(
    scores: Dict[int, float],
    clean_labels: np.ndarray,
    noisy_labels: np.ndarray,
    train_indices: np.ndarray,
) -> Tuple[float, float]:
    """Calculate AUROC and Precision@K for Task 3 scores offline.

    STRICT GUARANTEE:
    This function is strictly for post-hoc validation. It must not be called
    during score generation or used to tune scoring hyperparameters.

    y_true = 1 if sample is mislabeled, 0 if correctly labeled.
    Higher score indicates higher probability of mislabeling.
    """
    idx_to_pos = {int(idx): pos for pos, idx in enumerate(train_indices)}
    y_true = (noisy_labels != clean_labels).astype(np.int64)

    score_arr = np.zeros(len(train_indices), dtype=np.float64)
    for idx_val, pos in idx_to_pos.items():
        score_arr[pos] = scores.get(idx_val, -1e9)

    # 1. Precision@K where K is the number of true mislabeled images
    k = int(y_true.sum())
    top_k_indices = np.argsort(score_arr)[::-1][:k]
    precision_at_k = float(y_true[top_k_indices].mean())

    # 2. AUROC via Mann-Whitney U rank statistic (pure numpy, no sklearn dependency required)
    positives = score_arr[y_true == 1]
    negatives = score_arr[y_true == 0]
    n_pos = len(positives)
    n_neg = len(negatives)

    # Ranks
    ranks = np.argsort(np.argsort(score_arr)) + 1
    sum_ranks_pos = ranks[y_true == 1].sum()
    u_stat = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    auroc = float(u_stat / (n_pos * n_neg))

    return auroc, precision_at_k
