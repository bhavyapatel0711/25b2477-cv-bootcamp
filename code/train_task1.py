"""Task 1 Baselines Training Script.

Trains two separate ResNet-18 models:
1. Clean Reference: trained on clean_label (ceiling benchmark).
2. Noisy Baseline: trained on noisy_label with Cross-Entropy loss.

Strict Rules:
- Uses exact ResNet-18 architecture from model.py.
- Fixed hyperparameter suite: SGD (lr=0.1, momentum=0.9, weight_decay=5e-4),
  CosineAnnealingLR (T_max=60, eta_min=1e-4), batch_size=128, max_epochs=60, seed=42.
- Clean-reference validated on clean labels; noisy baseline validated on noisy labels.
- Clean labels NEVER used to select or train the noisy baseline.
- Scheduler stepped exactly once per completed epoch.
- Persistent checkpointing for Google Colab with resumption support up to 60 epochs.
- Outputs test predictions: predictions/t1_clean_ref.csv and predictions/t1_noisy_ce.csv.
"""

import argparse
import os
import random
import sys
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add script directory and parent directory to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, PARENT_DIR, "/content"]:
    if p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)

from data_utils import find_data_dir, get_loaders, get_task2_offline_eval_data
from model import ResNet18
from train_utils import (
    BATCH_SIZE,
    INITIAL_LR,
    MAX_EPOCHS,
    MOMENTUM,
    NUM_CLASSES,
    SEED,
    WEIGHT_DECAY,
    Task2DynamicsTracker,
    evaluate,
    get_device,
    get_model,
    get_optimizer_and_scheduler,
    predict_test,
    save_predictions_csv,
    set_seed,
    train_one_epoch,
)


def get_checkpoint_dir(custom_dir: Optional[str] = None) -> str:
    """Return persistent directory for checkpoints (supporting Google Drive)."""
    if custom_dir is not None:
        return os.path.abspath(custom_dir)

    # If running in Google Colab with Drive mounted, default to Drive
    colab_drive_path = "/content/drive/MyDrive/cv_bootcamp/checkpoints/task1"
    if os.path.isdir("/content/drive/MyDrive"):
        return colab_drive_path

    # Local fallback
    return os.path.abspath(os.path.join(PARENT_DIR, "checkpoints", "task1"))


def get_predictions_dir(custom_dir: Optional[str] = None) -> str:
    """Return directory for submission predictions."""
    if custom_dir is not None:
        return os.path.abspath(custom_dir)
    return os.path.abspath(os.path.join(PARENT_DIR, "predictions"))


def capture_rng_state() -> Dict:
    """Capture current RNG states for reproducibility across pauses."""
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_state(rng_state: Dict):
    """Restore saved RNG states."""
    if "torch" in rng_state and rng_state["torch"] is not None:
        torch_state = torch.as_tensor(rng_state["torch"], dtype=torch.uint8, device="cpu")
        torch.set_rng_state(torch_state)
    if (
        torch.cuda.is_available()
        and "torch_cuda" in rng_state
        and rng_state["torch_cuda"] is not None
    ):
        cuda_states = [
            torch.as_tensor(s, dtype=torch.uint8, device="cpu")
            for s in rng_state["torch_cuda"]
        ]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
        elif len(cuda_states) > 0:
            torch.cuda.set_rng_state(cuda_states[0], 0)
    if "numpy" in rng_state and rng_state["numpy"] is not None:
        np.random.set_state(rng_state["numpy"])
    if "python" in rng_state and rng_state["python"] is not None:
        random.setstate(rng_state["python"])


def save_checkpoint(state: Dict, filepath: str):
    """Save checkpoint dictionary to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    torch.save(state, filepath)


def train_single_model(
    model_name: str,
    label_type: str,
    data_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    output_csv_path: Optional[str] = None,
    max_epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    resume: bool = True,
    dry_run: bool = False,
    eval_checkpoint: str = "final",
    track_dynamics: bool = False,
    dynamics_csv_path: Optional[str] = None,
    early_stop_csv_path: Optional[str] = None,
    mislabeled_target: str = "clean",
) -> Dict:
    """Train a single ResNet-18 model under Task 1 guidelines.

    Args:
        model_name: identifier string e.g. 'clean_ref' or 'noisy_ce'
        label_type: 'clean' for clean reference, 'noisy' for noisy baseline
        data_dir: dataset directory
        checkpoint_dir: directory to store persistent checkpoints
        output_csv_path: destination for test predictions
        max_epochs: budget (capped at 60)
        batch_size: mini-batch size (128)
        num_workers: DataLoader worker count
        resume: whether to resume from existing latest checkpoint if present
        dry_run: run a single forward/backward verification step without full training
        eval_checkpoint: 'final' (epoch 60, required for Task 1) or 'best'
        track_dynamics: whether to passively log Task 2 dynamics metrics
        dynamics_csv_path: destination path for t2_dynamics.csv
        early_stop_csv_path: destination path for t2_early_stop.csv (from best noisy val acc)
        mislabeled_target: 'clean' (default) or 'noisy' for train_acc_mislabeled evaluation
    """
    assert max_epochs <= MAX_EPOCHS, f"Epoch budget {max_epochs} exceeds max limit of {MAX_EPOCHS}"
    assert label_type in ("clean", "noisy"), f"Invalid label_type for Task 1: {label_type}"

    set_seed(SEED)
    device = get_device()
    ckpt_dir = get_checkpoint_dir(checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    latest_ckpt_path = os.path.join(ckpt_dir, f"{model_name}_latest.pth")
    best_ckpt_path = os.path.join(ckpt_dir, f"{model_name}_best.pth")

    print(f"\n{'='*70}")
    print(f"Task 1 Setup: {model_name.upper()} (training on {label_type}_label)")
    print(f"Device: {device}")
    print(f"Epoch Budget: {max_epochs} | Batch Size: {batch_size} | Seed: {SEED}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"{'='*70}")

    # 1. Load data
    train_loader, val_loader, test_loader = get_loaders(
        data_dir=data_dir,
        label_type=label_type,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # 2. Build model, optimizer, scheduler, criterion
    model = get_model(num_classes=NUM_CLASSES).to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model, epochs=max_epochs)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    best_val_acc = -1.0
    best_epoch = -1

    # Initialize offline dynamics tracker if requested (Task 2)
    # STRICT COMPLIANCE: Clean labels and clean_val_loader are used EXCLUSIVELY
    # for offline dynamics tracking; they never touch loss, optimizer, or checkpoint selection.
    dynamics_tracker = None
    if track_dynamics:
        train_noisy_labels, train_clean_labels, train_idx, clean_val_loader = (
            get_task2_offline_eval_data(data_dir)
        )
        dynamics_tracker = Task2DynamicsTracker(
            train_noisy_labels=train_noisy_labels,
            train_clean_labels=train_clean_labels,
            train_indices=train_idx,
            clean_val_loader=clean_val_loader,
            mislabeled_target=mislabeled_target,
        )

    # 3. Resume from checkpoint if present
    if resume and os.path.isfile(latest_ckpt_path):
        print(f"Found existing checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", -1.0)
        best_epoch = checkpoint.get("best_epoch", -1)
        if "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"])
        if track_dynamics and dynamics_tracker is not None:
            saved_records = checkpoint.get("dynamics_records")
            completed_epochs = checkpoint.get("epoch", 0)
            if saved_records is not None:
                dynamics_tracker.load_records(saved_records)

            # Safety Guard: Check if historical dynamics records match completed epochs
            if completed_epochs > 0 and len(dynamics_tracker.records) < completed_epochs:
                if not dry_run:
                    raise RuntimeError(
                        f"\n[SAFETY GUARD] Resume conflict for '{model_name}':\n"
                        f"Checkpoint at '{latest_ckpt_path}' was trained through epoch {completed_epochs}, "
                        f"but contains {len(dynamics_tracker.records)} per-epoch dynamics records.\n"
                        f"This checkpoint was saved before dynamics tracking was enabled. Historical metrics "
                        f"for epochs 1..{completed_epochs} cannot be recovered from weights alone.\n"
                        f"Resuming with track_dynamics=True would generate an incomplete or empty dynamics CSV "
                        f"({dynamics_csv_path}), violating Task 2 completeness requirements.\n"
                        f"Remedy: To log full 1..{max_epochs} dynamics, run with a dedicated checkpoint directory "
                        f"(e.g. --checkpoint_dir checkpoints/task2) or specify --no_resume with an isolated checkpoint dir. "
                        f"Execution halted before modifying any output files."
                    )
                else:
                    print(
                        f"[SAFETY GUARD WARNING] Checkpoint lacks dynamics records ({len(dynamics_tracker.records)}/{completed_epochs}). "
                        f"Dry-run proceeding for batch verification only."
                    )
        print(f"Resumed from completed epoch {checkpoint['epoch']} (best val acc: {best_val_acc:.2f}%)")

    # If training was already completed
    if start_epoch > max_epochs and not dry_run:
        print(f"Model {model_name} has already completed all {max_epochs} epochs.")
    elif dry_run:
        print("[DRY RUN] Testing single forward and backward pass on 1 batch...")
        model.train()
        images, labels, _ = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Test single validation batch
        val_images, val_labels, _ = next(iter(val_loader))
        val_images, val_labels = val_images.to(device), val_labels.to(device)
        model.eval()
        with torch.no_grad():
            val_out = model(val_images)
            val_loss = criterion(val_out, val_labels).item()
            val_acc = 100.0 * val_out.max(1)[1].eq(val_labels).sum().item() / val_labels.size(0)
        print(f"[DRY RUN SUCCESS] Single step loss: {loss.item():.4f}, val batch acc: {val_acc:.2f}%")
        return {"status": "dry_run_passed"}
    else:
        # 4. Training loop (epoch budget strictly respected)
        for epoch in range(start_epoch, max_epochs + 1):
            train_loss, train_acc, preds_dict = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            # Scheduler step executed exactly once per completed epoch
            scheduler.step()

            # Evaluate on permitted validation split (noisy labels for Tasks 1-4)
            val_loss, val_acc = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                criterion=criterion,
            )

            # Model selection / early stopping is based STRICTLY on permitted validation accuracy
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                best_epoch = epoch

            # Passive offline dynamics logging (Task 2)
            # Clean labels are never used to train, filter, or select models.
            dyn_info = ""
            if track_dynamics and dynamics_tracker is not None:
                row = dynamics_tracker.record_epoch(
                    epoch=epoch,
                    train_predictions=preds_dict,
                    val_acc_noisy=val_acc,
                    model=model,
                    device=device,
                )
                if dynamics_csv_path is not None:
                    dynamics_tracker.save_csv(dynamics_csv_path)
                dyn_info = (
                    f" | Dyn: TrCorr={row['train_acc_correct']:.2f}% "
                    f"TrMis={row['train_acc_mislabeled']:.2f}% "
                    f"ValClean={row['val_acc_clean']:.2f}%"
                )

            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch [{epoch:02d}/{max_epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc ({label_type}): {val_acc:6.2f}% | "
                f"LR: {current_lr:.5f} {'[BEST]' if is_best else ''}{dyn_info}"
            )

            # Save latest checkpoint
            latest_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "rng_state": capture_rng_state(),
                "dynamics_records": (
                    dynamics_tracker.records
                    if (track_dynamics and dynamics_tracker is not None)
                    else None
                ),
            }
            save_checkpoint(latest_state, latest_ckpt_path)

            # Save best checkpoint
            if is_best:
                best_state = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "rng_state": capture_rng_state(),
                }
                save_checkpoint(best_state, best_ckpt_path)

    # Save final dynamics log if tracking enabled
    if track_dynamics and dynamics_tracker is not None and dynamics_csv_path is not None:
        if len(dynamics_tracker.records) == 0:
            print(f"[SAFETY GUARD] No dynamics records were captured; skipping save to avoid writing empty {dynamics_csv_path}.")
        else:
            dynamics_tracker.save_csv(dynamics_csv_path)
            print(f"Saved Task 2 dynamics log ({len(dynamics_tracker.records)} epochs) -> {dynamics_csv_path}")

    # 5a. Generate early-stop predictions if requested (Task 2)
    # Selected strictly based on best noisy validation accuracy
    if early_stop_csv_path is not None and not dry_run:
        if os.path.isfile(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt["model_state_dict"])
            if "epoch" in best_ckpt and best_ckpt["epoch"] is not None:
                best_epoch = best_ckpt["epoch"]
            if "val_acc" in best_ckpt and best_ckpt["val_acc"] is not None:
                best_val_acc = best_ckpt["val_acc"]
            print(f"Loaded early-stop checkpoint from epoch {best_epoch} (noisy val acc: {best_val_acc:.2f}%)...")
            print(f"Generating early-stop test predictions -> {early_stop_csv_path}")
            early_preds = predict_test(model, test_loader, device)
            save_predictions_csv(early_preds, early_stop_csv_path)
            print(f"Saved {len(early_preds)} early-stop test predictions to {early_stop_csv_path}")
        else:
            print(f"Warning: No best checkpoint found at {best_ckpt_path} for early-stop predictions.")

    # 5b. Generate primary output predictions (Task 1: final epoch 60; or according to eval_checkpoint)
    if output_csv_path is not None and not dry_run:
        if eval_checkpoint == "best":
            if os.path.isfile(best_ckpt_path):
                best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(best_ckpt["model_state_dict"])
                if "epoch" in best_ckpt and best_ckpt["epoch"] is not None:
                    best_epoch = best_ckpt["epoch"]
                if "val_acc" in best_ckpt and best_ckpt["val_acc"] is not None:
                    best_val_acc = best_ckpt["val_acc"]
                print(f"Using best checkpoint (epoch {best_epoch}, val acc: {best_val_acc:.2f}%) for {output_csv_path}")
        else:
            # eval_checkpoint == "final": Competition requires final epoch 60 predictions for Task 1
            if os.path.isfile(latest_ckpt_path):
                latest_ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(latest_ckpt["model_state_dict"])
                final_epoch = latest_ckpt.get("epoch", max_epochs)
                final_val_acc = latest_ckpt.get("val_acc", -1.0)
                print(f"Using final checkpoint (epoch {final_epoch}, val acc: {final_val_acc:.2f}%) for {output_csv_path}")
            else:
                print(f"Using current model state (final epoch) for {output_csv_path}")

        print(f"Generating test predictions -> {output_csv_path}")
        predictions = predict_test(model, test_loader, device)
        save_predictions_csv(predictions, output_csv_path)
        print(f"Saved {len(predictions)} test predictions to {output_csv_path}")

    return {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_ckpt_path": best_ckpt_path,
        "latest_ckpt_path": latest_ckpt_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Task 1 Baseline Training")
    parser.add_argument(
        "--model",
        choices=["all", "clean_ref", "noisy_ce"],
        default="all",
        help="Model to train: 'clean_ref', 'noisy_ce', or 'all'",
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Path to data/ directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to save checkpoints")
    parser.add_argument("--predictions_dir", type=str, default=None, help="Path to predictions/ directory")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Epoch budget (max 60)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size (default 128)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--no_resume", action="store_true", help="Do not resume from existing checkpoint")
    parser.add_argument("--dry_run", action="store_true", help="Run lightweight single-batch check only")
    parser.add_argument(
        "--eval_checkpoint",
        choices=["final", "best"],
        default="final",
        help="Checkpoint to evaluate for predictions: 'final' (Task 1 rule: epoch 60) or 'best'",
    )
    parser.add_argument("--track_dynamics", action="store_true", help="Log Task 2 dynamics curves")
    parser.add_argument("--dynamics_csv", type=str, default=None, help="Path for t2_dynamics.csv")
    parser.add_argument("--early_stop_csv", type=str, default=None, help="Path for t2_early_stop.csv")
    parser.add_argument(
        "--mislabeled_target",
        choices=["clean", "noisy"],
        default="clean",
        help="Target labels for mislabeled subset accuracy: 'clean' (default) or 'noisy'",
    )
    args = parser.parse_args()

    pred_dir = get_predictions_dir(args.predictions_dir)
    os.makedirs(pred_dir, exist_ok=True)

    results = {}

    # 1. Clean Reference Model
    if args.model in ("all", "clean_ref"):
        out_csv = os.path.join(pred_dir, "t1_clean_ref.csv")
        res_clean = train_single_model(
            model_name="clean_ref",
            label_type="clean",
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            output_csv_path=out_csv,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            resume=not args.no_resume,
            dry_run=args.dry_run,
            eval_checkpoint=args.eval_checkpoint,
        )
        results["clean_ref"] = res_clean

    # 2. Noisy Baseline Model
    if args.model in ("all", "noisy_ce"):
        out_csv = os.path.join(pred_dir, "t1_noisy_ce.csv")
        res_noisy = train_single_model(
            model_name="noisy_ce",
            label_type="noisy",
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            output_csv_path=out_csv,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            resume=not args.no_resume,
            dry_run=args.dry_run,
            eval_checkpoint=args.eval_checkpoint,
            track_dynamics=args.track_dynamics,
            dynamics_csv_path=args.dynamics_csv,
            early_stop_csv_path=args.early_stop_csv,
            mislabeled_target=args.mislabeled_target,
        )
        results["noisy_ce"] = res_noisy

    print("\nTask 1 execution finished.")


if __name__ == "__main__":
    main()
