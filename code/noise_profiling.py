"""Task 1, Step 1: Noise Profiling and Confusion Matrix Analysis.

Strict Rules:
- Uses ONLY the provided data/labels.csv, data/train_idx.txt, and data/classes.txt.
- Offline exploratory analysis permitted by challenge rules.
- Does NOT touch model.py, data_utils.py, or train_utils.py.
- Does NOT train models or generate predictions.
"""

import os
import pandas as pd
import numpy as np


def run_noise_profiling(data_dir: str = "data", output_dir: str = "analysis"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load classes
    classes_path = os.path.join(data_dir, "classes.txt")
    with open(classes_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    # 2. Load labels and training indices
    labels_path = os.path.join(data_dir, "labels.csv")
    train_idx_path = os.path.join(data_dir, "train_idx.txt")

    df_labels = pd.read_csv(labels_path)
    with open(train_idx_path, "r", encoding="utf-8") as f:
        train_idx = [int(line.strip()) for line in f if line.strip()]

    train_df = df_labels.iloc[train_idx].copy()
    n_train = len(train_df)

    # 3. Calculate noise rate
    mislabeled_mask = train_df["noisy_label"] != train_df["clean_label"]
    n_mislabeled = int(mislabeled_mask.sum())
    noise_rate_pct = (n_mislabeled / n_train) * 100.0

    # 4. Compute 10x10 confusion matrix
    # Standard orientation: Rows = True Clean Label, Columns = Observed Noisy Label
    cm_matrix = np.zeros((10, 10), dtype=np.int64)
    for clean_lbl, noisy_lbl in zip(train_df["clean_label"], train_df["noisy_label"]):
        cm_matrix[clean_lbl, noisy_lbl] += 1

    cm_df = pd.DataFrame(cm_matrix, index=classes, columns=classes)
    cm_df.index.name = "clean_label (row)"

    # Save CSV
    cm_csv_path = os.path.join(output_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path)

    # 5. Extract top off-diagonal confusion pairs
    off_diag = []
    for i in range(10):
        for j in range(10):
            if i != j:
                count = cm_matrix[i, j]
                pct_of_true_class = (count / cm_matrix[i].sum()) * 100.0
                off_diag.append({
                    "clean_class": classes[i],
                    "clean_id": i,
                    "noisy_class": classes[j],
                    "noisy_id": j,
                    "count": count,
                    "pct_of_clean": pct_of_true_class,
                })

    off_diag_df = pd.DataFrame(off_diag).sort_values(by="count", ascending=False).reset_index(drop=True)

    # 6. Generate human-readable summary
    summary_path = os.path.join(output_dir, "noise_profile_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("CIFAR-10N (WORST SPLIT) - TASK 1 NOISE PROFILING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total training samples:         {n_train:,}\n")
        f.write(f"Mislabeled training samples:    {n_mislabeled:,}\n")
        f.write(f"Correctly labeled samples:      {n_train - n_mislabeled:,}\n")
        f.write(f"Training noise rate:            {noise_rate_pct:.4f}%\n\n")

        f.write("-" * 70 + "\n")
        f.write("CONFUSION MATRIX CONVENTION\n")
        f.write("-" * 70 + "\n")
        f.write("Rows    = True Ground-Truth Label (clean_label)\n")
        f.write("Columns = Observed Crowd-Worker Label (noisy_label)\n")
        f.write("Diagonal = Human annotations that match ground truth\n")
        f.write("Off-diagonal = Human annotation errors\n\n")

        f.write("-" * 70 + "\n")
        f.write("TOP 15 MOST FREQUENT CONFUSION PAIRS (Clean -> Noisy)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Clean (True)':<15} -> {'Noisy (Observed)':<16} {'Count':>8} {'% of True Class':>16}\n")
        f.write("-" * 70 + "\n")
        for _, row in off_diag_df.head(15).iterrows():
            f.write(f"{row['clean_class']:<15} -> {row['noisy_class']:<16} {row['count']:>8} {row['pct_of_clean']:>15.2f}%\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("PER-CLASS CLEAN ACCURACY (Human Label Agreement with Clean)\n")
        f.write("-" * 70 + "\n")
        for i, c in enumerate(classes):
            correct = cm_matrix[i, i]
            total = cm_matrix[i].sum()
            agree_pct = (correct / total) * 100.0
            noise_pct = 100.0 - agree_pct
            f.write(f"Class {i} ({c:<11}): {agree_pct:6.2f}% agreement ({noise_pct:6.2f}% noise, {total - correct:,}/{total:,} mislabeled)\n")

    return {
        "n_train": n_train,
        "n_mislabeled": n_mislabeled,
        "noise_rate_pct": noise_rate_pct,
        "cm_csv_path": os.path.abspath(cm_csv_path),
        "summary_path": os.path.abspath(summary_path),
        "top_pairs": off_diag_df.head(10).to_dict(orient="records"),
    }


if __name__ == "__main__":
    res = run_noise_profiling()
    print("Done. Noise rate:", f"{res['noise_rate_pct']:.4f}%")
