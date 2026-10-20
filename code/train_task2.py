"""Task 2 Training Dynamics and Early Stopping Script.

Strict Rules & Competition Constraints:
1. Model: ResNet-18 defined in model.py, trained from scratch (no pretrained weights).
2. Fixed Hyperparameters:
   SGD (lr=0.1, momentum=0.9, weight_decay=5e-4),
   CosineAnnealingLR (T_max=60, eta_min=1e-4),
   batch_size=128, max_epochs=60, seed=42.
3. Clean Labels Isolation:
   Clean labels and clean validation split are used ONLY for passive offline dynamics measurement
   in Task2DynamicsTracker. They must NEVER be passed to the loss function, optimizer,
   or early-stopping checkpoint selector.
4. Model Selection / Early Stopping:
   The early-stopping checkpoint is selected strictly using noisy validation accuracy (val_acc_noisy).
5. Output Requirements:
   - predictions/t2_dynamics.csv: exact header
     epoch,train_acc_correct,train_acc_mislabeled,val_acc_noisy,val_acc_clean
   - predictions/t2_early_stop.csv: 10,000 rows (id,pred) from early-stopped checkpoint.
6. Checkpoint Safety:
   Checkpoints default to checkpoints/task2 to avoid overwriting existing checkpoints
   (e.g. checkpoints/task1, checkpoints/fresh_audit).
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional
import torch

# Add script directory and parent directory to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, PARENT_DIR, "/content"]:
    if p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)

from data_utils import get_loaders
from model import ResNet18
from train_task1 import get_predictions_dir, train_single_model
from train_utils import (
    BATCH_SIZE,
    MAX_EPOCHS,
    NUM_CLASSES,
    SEED,
    get_device,
    get_model,
    predict_test,
    save_predictions_csv,
    set_seed,
)


def get_task2_checkpoint_dir(custom_dir: Optional[str] = None) -> str:
    """Return persistent checkpoint directory for Task 2 (isolating from Task 1)."""
    if custom_dir is not None:
        return os.path.abspath(custom_dir)

    # Google Colab with Drive mounted
    colab_drive_path = "/content/drive/MyDrive/cv_bootcamp/checkpoints/task2"
    if os.path.isdir("/content/drive/MyDrive"):
        return colab_drive_path

    # Local fallback
    return os.path.abspath(os.path.join(PARENT_DIR, "checkpoints", "task2"))


def validate_dynamics_file(csv_path: str) -> bool:
    """Validate t2_dynamics.csv against official check_format.py specifications."""
    expected_header = ["epoch", "train_acc_correct", "train_acc_mislabeled", "val_acc_noisy", "val_acc_clean"]
    if not os.path.isfile(csv_path):
        print(f"[FAIL] Dynamics file not found: {csv_path}")
        return False

    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if any(c.strip() for c in r)]

    if not rows:
        print(f"[FAIL] Dynamics file is empty: {csv_path}")
        return False

    header = [h.strip().lower() for h in rows[0]]
    if header != expected_header:
        print(f"[FAIL] Dynamics header mismatch. Expected {expected_header}, got {header}")
        return False

    data_rows = rows[1:]
    if len(data_rows) > MAX_EPOCHS:
        print(f"[FAIL] Dynamics file has {len(data_rows)} rows, exceeds max budget {MAX_EPOCHS}")
        return False

    epochs = []
    for i, r in enumerate(data_rows, start=2):
        if len(r) != 5:
            print(f"[FAIL] Row {i}: expected 5 columns, found {len(r)}")
            return False
        try:
            ep = int(r[0].strip())
            epochs.append(ep)
            vals = [float(v.strip()) for v in r[1:]]
            if any(v < 0.0 or v > 100.0 for v in vals):
                print(f"[FAIL] Row {i}: values outside [0, 100]: {vals}")
                return False
        except ValueError as e:
            print(f"[FAIL] Row {i}: invalid numeric format: {e}")
            return False

    valid_seq = list(range(1, len(data_rows) + 1))
    valid_zero_seq = list(range(len(data_rows)))
    if epochs not in (valid_seq, valid_zero_seq):
        print(f"[FAIL] Epoch sequence not contiguous 1..N or 0..N-1: {epochs[:5]}...{epochs[-3:]}")
        return False

    print(f"[PASS] Dynamics file valid: {len(data_rows)} epochs logged, exact header matched.")
    return True


def validate_predictions_file(csv_path: str) -> bool:
    """Validate test predictions file against official check_format.py specifications."""
    if not os.path.isfile(csv_path):
        print(f"[FAIL] Predictions file not found: {csv_path}")
        return False

    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if any(c.strip() for c in r)]

    if not rows:
        print(f"[FAIL] Predictions file is empty: {csv_path}")
        return False

    header = [h.strip().lower() for h in rows[0]]
    if header != ["id", "pred"]:
        print(f"[FAIL] Header must be ['id', 'pred'], got {header}")
        return False

    data_rows = rows[1:]
    if len(data_rows) != 10000:
        print(f"[FAIL] Expected exactly 10,000 predictions, found {len(data_rows)}")
        return False

    seen_ids = set()
    for i, r in enumerate(data_rows, start=2):
        if len(r) != 2:
            print(f"[FAIL] Row {i}: expected 2 columns, found {len(r)}")
            return False
        try:
            pid = int(r[0].strip())
            pred = int(r[1].strip())
            if not 0 <= pid < 10000:
                print(f"[FAIL] Row {i}: id {pid} outside 0..9999")
                return False
            if not 0 <= pred < 10:
                print(f"[FAIL] Row {i}: pred {pred} outside 0..9")
                return False
            seen_ids.add(pid)
        except ValueError as e:
            print(f"[FAIL] Row {i}: invalid integer: {e}")
            return False

    if len(seen_ids) != 10000:
        print(f"[FAIL] Duplicate or missing IDs in predictions.")
        return False

    print(f"[PASS] Predictions file valid: exactly 10,000 unique IDs in 0..9999 with valid classes 0..9.")
    return True


def analyze_memorization_start(dynamics_csv_path: str, mislabeled_target: str = "clean") -> Optional[int]:
    """Estimate the epoch at which wrong-label memorization begins.

    If mislabeled_target == 'clean':
        Memorization begins when train accuracy on mislabeled samples (against clean labels)
        peaks and begins to systematically deteriorate.
    If mislabeled_target == 'noisy':
        Memorization begins when train accuracy on mislabeled samples (against noisy labels)
        starts rising systematically above baseline chance (~10%).
    """
    if not os.path.isfile(dynamics_csv_path):
        return None

    epochs = []
    mis_accs = []
    clean_val_accs = []
    with open(dynamics_csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            mis_accs.append(float(row["train_acc_mislabeled"]))
            clean_val_accs.append(float(row["val_acc_clean"]))

    if not epochs:
        return None

    if mislabeled_target == "clean":
        # Epoch where clean accuracy on mislabeled subset reaches its peak
        best_idx = int(torch.tensor(mis_accs).argmax().item())
        return epochs[best_idx]
    else:
        # Epoch where noisy accuracy on mislabeled subset begins sustained climb above 15%
        for ep, acc in zip(epochs, mis_accs):
            if acc > 15.0:
                return ep
        return epochs[0]


def run_eval_only(
    checkpoint_path: str,
    output_csv_path: str,
    data_dir: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
):
    """Generate test predictions from an existing checkpoint without retraining."""
    set_seed(SEED)
    device = get_device()
    print(f"\n[EVAL ONLY] Generating test predictions from: {checkpoint_path}")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    _, _, test_loader = get_loaders(
        data_dir=data_dir,
        label_type="noisy",
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = get_model(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "unknown")
    val_acc = ckpt.get("val_acc", -1.0)
    print(f"Loaded checkpoint from epoch {epoch} (val acc: {val_acc:.2f}%)")

    predictions = predict_test(model, test_loader, device)
    save_predictions_csv(predictions, output_csv_path)
    print(f"Saved {len(predictions)} predictions -> {output_csv_path}")
    validate_predictions_file(output_csv_path)


def main():
    parser = argparse.ArgumentParser(description="Task 2 Training Dynamics and Early Stopping")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to data/ directory")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to save Task 2 checkpoints (default: checkpoints/task2 to protect Task 1)",
    )
    parser.add_argument("--predictions_dir", type=str, default=None, help="Path to predictions/ directory")
    parser.add_argument("--dynamics_csv", type=str, default=None, help="Custom path for t2_dynamics.csv")
    parser.add_argument("--early_stop_csv", type=str, default=None, help="Custom path for t2_early_stop.csv")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Epoch budget (max 60)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size (default 128)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--no_resume", action="store_true", help="Do not resume from existing checkpoint")
    parser.add_argument("--dry_run", action="store_true", help="Run lightweight single-batch check only")
    parser.add_argument(
        "--mislabeled_target",
        choices=["clean", "noisy"],
        default="clean",
        help="Target for mislabeled train accuracy: 'clean' (default in train_utils.py) or 'noisy'",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Generate t2_early_stop.csv from existing checkpoint without running training",
    )
    parser.add_argument(
        "--eval_checkpoint_path",
        type=str,
        default=None,
        help="Explicit checkpoint path when using --eval_only (e.g. checkpoints/task1/noisy_ce_best.pth)",
    )
    parser.add_argument(
        "--generate_t1_final_check",
        action="store_true",
        help="Also output predictions/t1_noisy_ce_reproduced.csv from epoch 60 for comparison",
    )
    parser.add_argument(
        "--confirm_new_run",
        action="store_true",
        help="Explicitly confirm intent to spend budget on a fresh 60-epoch training run",
    )
    args = parser.parse_args()

    pred_dir = get_predictions_dir(args.predictions_dir)
    os.makedirs(pred_dir, exist_ok=True)

    dynamics_csv = args.dynamics_csv or os.path.join(pred_dir, "t2_dynamics.csv")
    early_stop_csv = args.early_stop_csv or os.path.join(pred_dir, "t2_early_stop.csv")
    ckpt_dir = get_task2_checkpoint_dir(args.checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("TASK 2: TRAINING DYNAMICS & EARLY STOPPING")
    print(f"Checkpoints:     {ckpt_dir}")
    print(f"Dynamics CSV:    {dynamics_csv}")
    print(f"Early Stop CSV:  {early_stop_csv}")
    print(f"Mislabeled Tgt:  {args.mislabeled_target}")
    print("=" * 70)

    # Mode 1: Eval-only (generates early-stop predictions from existing checkpoint)
    if args.eval_only:
        eval_ckpt = args.eval_checkpoint_path
        if eval_ckpt is None:
            # Default to noisy_ce_best.pth in checkpoint_dir, or check checkpoints/task1/noisy_ce_best.pth
            cand1 = os.path.join(ckpt_dir, "noisy_ce_best.pth")
            cand2 = os.path.join(PARENT_DIR, "checkpoints", "task1", "noisy_ce_best.pth")
            if os.path.isfile(cand1):
                eval_ckpt = cand1
            elif os.path.isfile(cand2):
                eval_ckpt = cand2
            else:
                raise FileNotFoundError(
                    f"No best checkpoint found at {cand1} or {cand2}. "
                    "Please specify --eval_checkpoint_path explicitly."
                )

        run_eval_only(
            checkpoint_path=eval_ckpt,
            output_csv_path=early_stop_csv,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print("\nTask 2 eval-only finished successfully.")
        return

    # Budget Guard: Prevent automatic 60-epoch training runs without explicit user approval
    if not args.dry_run and not args.confirm_new_run:
        print("\n" + "!" * 70)
        print("[BUDGET GUARD: FULL TRAINING EXECUTION DISABLED]")
        print("!" * 70)
        print("The challenge rules impose strict budget limits:")
        print("  - At most 60 epochs per network.")
        print("  - At most 2 networks per method.")
        print()
        print("A 60-epoch run has already been completed for Task 1 noisy CE.")
        print("Starting another 60-epoch training run will expend the second network allowance")
        print("for this method. To prevent unintentional budget expenditure, full training")
        print("is disabled by default.")
        print()
        print("Options:")
        print("  1. To evaluate an existing checkpoint without retraining:")
        print("     python code/train_task2.py --eval_only --eval_checkpoint_path checkpoints/task1/noisy_ce_best.pth")
        print("  2. To verify the pipeline with a single batch without training:")
        print("     python code/train_task2.py --dry_run")
        print("  3. To explicitly authorize and run a fresh 60-epoch training run:")
        print("     python code/train_task2.py --confirm_new_run")
        print("!" * 70 + "\n")
        sys.exit(0)

    # Mode 2: Full reproducible 60-epoch training with dynamics tracking
    t1_check_csv = None
    if args.generate_t1_final_check:
        t1_check_csv = os.path.join(pred_dir, "t1_noisy_ce_reproduced.csv")

    results = train_single_model(
        model_name="noisy_ce",
        label_type="noisy",
        data_dir=args.data_dir,
        checkpoint_dir=ckpt_dir,
        output_csv_path=t1_check_csv,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resume=not args.no_resume,
        dry_run=args.dry_run,
        eval_checkpoint="final",
        track_dynamics=True,
        dynamics_csv_path=dynamics_csv,
        early_stop_csv_path=early_stop_csv,
        mislabeled_target=args.mislabeled_target,
    )

    if args.dry_run:
        print("\n[DRY RUN] Task 2 dry run completed successfully.")
        return

    # Validate output files against challenge rules
    print("\n" + "=" * 70)
    print("OUTPUT VALIDATION (check_format.py standards)")
    print("=" * 70)
    dyn_ok = validate_dynamics_file(dynamics_csv)
    pred_ok = validate_predictions_file(early_stop_csv)

    if t1_check_csv and os.path.isfile(t1_check_csv):
        validate_predictions_file(t1_check_csv)
        print(f"[INFO] Final epoch 60 test predictions saved to: {t1_check_csv}")
        # Compare with existing t1_noisy_ce.csv if present
        orig_t1 = os.path.join(pred_dir, "t1_noisy_ce.csv")
        if os.path.isfile(orig_t1):
            with open(orig_t1, encoding="utf-8") as f1, open(t1_check_csv, encoding="utf-8") as f2:
                match = f1.read() == f2.read()
            print(f"[REPRODUCIBILITY CHECK] Matches existing t1_noisy_ce.csv: {match}")

    mem_epoch = analyze_memorization_start(dynamics_csv, mislabeled_target=args.mislabeled_target)

    print("\n" + "=" * 70)
    print("TASK 2 RESULTS SUMMARY FOR results.json")
    print("=" * 70)
    print(f'  "early_stop_epoch": {results["best_epoch"]},')
    print(f'  "val_acc_noisy_labels_at_early_stop": {results["best_val_acc"]:.4f},')
    print(f'  "memorization_start_epoch": {mem_epoch}')
    print("=" * 70)

    if dyn_ok and pred_ok:
        print("\n[SUCCESS] Task 2 completed and verified successfully.")
    else:
        print("\n[WARNING] Output validation had issues. Please inspect logs above.")


if __name__ == "__main__":
    main()
