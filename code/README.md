# Code and Reproduction

Experiments used the provided ResNet-18 architecture, trained from scratch.

## Configuration
- Seed: 42
- Optimizer: SGD, momentum 0.9, weight decay 0.0005
- Schedule: cosine annealing over 60 epochs
- Batch size: 128
- Augmentation: random crop (padding 4) and horizontal flip

## Scripts
- `noise_profiling.py`: noise profiling
- `train_task1.py`: Task 1 training
- `train_task2.py`: Task 2 training dynamics and early stopping
- `data_utils.py`: dataset utilities
- `train_utils.py`: training utilities

## Example commands
Run from the project root:

    python code/noise_profiling.py
    python code/train_task2.py --dry_run
    python code/train_task2.py --eval_only

Task 1 predictions and Task 2 early-stop predictions are available. Task 2 per-epoch dynamics and Tasks 3–5 outputs were not completed.