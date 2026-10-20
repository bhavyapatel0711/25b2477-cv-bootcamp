"""Check that your CV submission has the right files and formats BEFORE you submit.

Usage (run from this folder, so it can find data/):
    python check_format.py path/to/<rollno>_cv_bootcamp.zip
    python check_format.py path/to/your_submission_folder

It only checks names and formats. It does NOT score your predictions.
We use the same checks when grading, so a file that fails here gets 0 marks.
"""
import csv
import json
import math
import os
import re
import shutil
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

N_TEST = 10_000
N_TRAIN = 45_000
MAX_EPOCHS = 60
NUM_CLASSES = 10

# id,pred files with a fixed name
PRED_FILES = ["t1_clean_ref.csv", "t1_noisy_ce.csv", "t2_early_stop.csv",
              "t5_baseline_synth.csv", "t5_best_synth.csv"]
DYNAMICS_FILE = "t2_dynamics.csv"
DYNAMICS_HEADER = ["epoch", "train_acc_correct", "train_acc_mislabeled", "val_acc_noisy", "val_acc_clean"]
SCORES_FILE = "t3_noise_scores.csv"
T4_PATTERN = re.compile(r"^t4_.+\.csv$")
MIN_T4, MAX_T4 = 1, 3

TASKS = {
    "task1": ["t1_clean_ref.csv", "t1_noisy_ce.csv"],
    "task2": [DYNAMICS_FILE, "t2_early_stop.csv"],
    "task3": [SCORES_FILE],
    "task4": None,  # 1-3 files matching t4_<method>.csv, handled separately
    "task5": ["t5_baseline_synth.csv", "t5_best_synth.csv"],
}


# ---------------------------------------------------------------- helpers
def locate_root(path):
    """Return (root_folder, temp_dir_or_None). root_folder contains predictions/."""
    tmp = None
    if os.path.isfile(path) and path.lower().endswith(".zip"):
        tmp = tempfile.mkdtemp()
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        path = tmp
    for dirpath, dirnames, _ in os.walk(path):
        if "__MACOSX" in dirpath:
            continue
        if "predictions" in dirnames:
            return dirpath, tmp
    return None, tmp


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if any(c.strip() for c in r)]
    if not rows:
        raise ValueError("file is empty")
    return [h.lower() for h in rows[0]], rows[1:]


def to_int(s, what):
    if not re.fullmatch(r"[+-]?\d+", s):
        raise ValueError(f"{what} must be an integer, got '{s}'")
    return int(s)


def to_float(s, what):
    try:
        v = float(s)
    except ValueError:
        raise ValueError(f"{what} must be a number, got '{s}'") from None
    if not math.isfinite(v):
        raise ValueError(f"{what} must be finite, got '{s}'")
    return v


def load_train_idx(data_dir=DATA_DIR):
    path = os.path.join(data_dir, "train_idx.txt")
    with open(path, encoding="utf-8") as f:
        return {int(line) for line in f if line.strip()}


# ---------------------------------------------------------------- per-file checks
def check_pred_file(path, warnings=None):
    """id,pred with exactly 10,000 rows, ids 0..9999 once each, pred in 0..9. Returns {id: pred}."""
    header, rows = read_csv(path)
    if header != ["id", "pred"]:
        raise ValueError("header must be 'id,pred'")
    preds = {}
    for i, r in enumerate(rows, 2):
        if len(r) != 2:
            raise ValueError(f"row {i}: need exactly 2 columns")
        pid = to_int(r[0], f"row {i}: id")
        pred = to_int(r[1], f"row {i}: pred")
        if not 0 <= pid < N_TEST:
            raise ValueError(f"row {i}: id {pid} outside 0..{N_TEST - 1}")
        if not 0 <= pred < NUM_CLASSES:
            raise ValueError(f"row {i}: pred {pred} outside 0..{NUM_CLASSES - 1}")
        if pid in preds:
            raise ValueError(f"row {i}: id {pid} appears twice")
        preds[pid] = pred
    if len(preds) != N_TEST:
        missing = sorted(set(range(N_TEST)) - set(preds))
        raise ValueError(f"need {N_TEST} rows (ids 0..{N_TEST - 1}), found {len(preds)}; "
                         f"missing e.g. {missing[:3]}")
    if warnings is not None and len(set(preds.values())) == 1:
        warnings.append(f"{os.path.basename(path)}: every prediction is the same class")
    return preds


def check_dynamics(path, warnings=None):
    """Per-epoch log. Returns list of rows (floats)."""
    header, rows = read_csv(path)
    if header != DYNAMICS_HEADER:
        raise ValueError("header must be '" + ",".join(DYNAMICS_HEADER) + "'")
    if not rows:
        raise ValueError("no epoch rows")
    if len(rows) > MAX_EPOCHS:
        raise ValueError(f"{len(rows)} epochs logged, but the budget is at most {MAX_EPOCHS}")
    epochs, values = [], []
    for i, r in enumerate(rows, 2):
        if len(r) != 5:
            raise ValueError(f"row {i}: need exactly 5 columns")
        epochs.append(to_int(r[0], f"row {i}: epoch"))
        vals = [to_float(c, f"row {i}: {DYNAMICS_HEADER[k + 1]}") for k, c in enumerate(r[1:])]
        if any(v < 0 or v > 100 for v in vals):
            raise ValueError(f"row {i}: accuracies must be between 0 and 1 (or 0 and 100)")
        values.append(vals)
    if epochs not in (list(range(len(rows))), list(range(1, len(rows) + 1))):
        raise ValueError("epoch must count up by 1 per row, starting at 0 or 1, with no gaps")
    if warnings is not None and len(rows) < MAX_EPOCHS:
        warnings.append(f"{DYNAMICS_FILE}: only {len(rows)} epochs logged "
                        f"(Task 1 asks for the full {MAX_EPOCHS}-epoch budget)")
    return values


def check_scores(path, train_idx, warnings=None):
    """index,score for every training index exactly once. Returns {index: score}."""
    header, rows = read_csv(path)
    if header != ["index", "score"]:
        raise ValueError("header must be 'index,score'")
    scores = {}
    for i, r in enumerate(rows, 2):
        if len(r) != 2:
            raise ValueError(f"row {i}: need exactly 2 columns")
        idx = to_int(r[0], f"row {i}: index")
        if idx not in train_idx:
            raise ValueError(f"row {i}: index {idx} is not in train_idx.txt")
        if idx in scores:
            raise ValueError(f"row {i}: index {idx} appears twice")
        scores[idx] = to_float(r[1], f"row {i}: score")
    missing = train_idx - set(scores)
    if missing:
        raise ValueError(f"{len(missing)} training indices missing (e.g. {sorted(missing)[:3]}); "
                         f"need all {len(train_idx)}")
    if warnings is not None and len(set(scores.values())) == 1:
        warnings.append(f"{SCORES_FILE}: all scores are identical, so the ranking carries no information")
    return scores


# ---------------------------------------------------------------- results.json / report.md
def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def check_results(results, t4_files, issues):
    """Appends human-readable problems to `issues`."""
    if not isinstance(results, dict):
        issues.append("results.json must contain a JSON object")
        return
    for key in ["name", "roll_no"]:
        if not isinstance(results.get(key), str) or not results.get(key).strip():
            issues.append(f"results.json: '{key}' is not filled in")
    if not isinstance(results.get("seed"), int) or isinstance(results.get("seed"), bool):
        issues.append("results.json: 'seed' must be an integer")

    setup = results.get("training_setup")
    if not isinstance(setup, dict):
        issues.append("results.json: 'training_setup' is missing")
    else:
        for key in ["optimizer", "learning_rate_schedule", "augmentation"]:
            if not isinstance(setup.get(key), str) or not setup.get(key).strip():
                issues.append(f"results.json: training_setup.{key} is not filled in")
        for key in ["batch_size", "epochs"]:
            v = setup.get(key)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                issues.append(f"results.json: training_setup.{key} must be a positive integer")
        if isinstance(setup.get("epochs"), int) and setup["epochs"] > MAX_EPOCHS:
            issues.append(f"results.json: training_setup.epochs is {setup['epochs']}, budget is {MAX_EPOCHS}")

    numeric_fields = [
        ("task1", "noise_rate"),
        ("task1", "val_acc_noisy_labels__clean_ref_model"),
        ("task1", "val_acc_noisy_labels__noisy_ce_model"),
        ("task2", "memorization_start_epoch"),
        ("task2", "early_stop_epoch"),
        ("task2", "val_acc_noisy_labels_at_early_stop"),
        ("task5", "val_acc_synthetic_labels__baseline"),
        ("task5", "val_acc_synthetic_labels__best_method"),
    ]
    for path in numeric_fields:
        v = _get(results, *path)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            issues.append(f"results.json: {'.'.join(path)} is not filled in (needs a number)")
    nr = _get(results, "task1", "noise_rate")
    if isinstance(nr, (int, float)) and not isinstance(nr, bool) and not 0 <= nr <= 100:
        issues.append("results.json: task1.noise_rate looks out of range")

    if not isinstance(_get(results, "task3", "method"), str) or not _get(results, "task3", "method").strip():
        issues.append("results.json: task3.method is not filled in")

    methods = _get(results, "task4", "methods")
    if not isinstance(methods, list) or not (MIN_T4 <= len(methods) <= MAX_T4):
        issues.append(f"results.json: task4.methods must list {MIN_T4} to {MAX_T4} methods")
        methods = []
    names, files = [], []
    for k, m in enumerate(methods):
        if not isinstance(m, dict):
            issues.append(f"results.json: task4.methods[{k}] must be an object")
            continue
        name, fname, acc = m.get("name"), m.get("file"), m.get("val_acc_noisy_labels")
        if not isinstance(name, str) or not name.strip():
            issues.append(f"results.json: task4.methods[{k}].name is not filled in")
        else:
            names.append(name)
        if not isinstance(fname, str) or fname not in t4_files:
            issues.append(f"results.json: task4.methods[{k}].file '{fname}' is not a t4_*.csv in predictions/")
        else:
            files.append(fname)
        if isinstance(acc, bool) or not isinstance(acc, (int, float)):
            issues.append(f"results.json: task4.methods[{k}].val_acc_noisy_labels is not filled in")
    if len(set(files)) != len(files):
        issues.append("results.json: two task4 methods point to the same file")
    for f in t4_files:
        if f not in files:
            issues.append(f"predictions/{f} is not listed in results.json task4.methods")
    best = _get(results, "task4", "best_method")
    if not isinstance(best, str) or not best.strip():
        issues.append("results.json: task4.best_method is not filled in")
    elif names and best not in names:
        issues.append(f"results.json: task4.best_method '{best}' does not match any method name in task4.methods")


def check_report(path, issues, warnings):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    marks = list(re.finditer(r"(?im)^[\s#>*_\-]*\**\s*Q([1-5])\b", text))
    found = {int(m.group(1)) for m in marks}
    for q in range(1, 6):
        if q not in found:
            issues.append(f"report.md: no answer found for Q{q} (start each answer on a line beginning 'Q{q}')")
    for k, m in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(text)
        body = re.sub(r"^[\s#>*_\-]*\**\s*Q\d\b[.:)\]*\s-]*", "", text[m.start():end].strip())
        words = len(body.split())
        if words > 100:
            warnings.append(f"report.md: Q{m.group(1)} is about {words} words (limit is 100)")


# ---------------------------------------------------------------- whole submission
def check_submission(root, train_idx):
    pred_dir = os.path.join(root, "predictions")
    present = set(os.listdir(pred_dir)) if os.path.isdir(pred_dir) else set()
    status, general, warnings = {}, [], []

    def run(name, fn):
        if name not in present:
            status[name] = "MISSING"
            return
        try:
            fn(os.path.join(pred_dir, name))
            status[name] = "OK"
        except Exception as e:  # noqa: BLE001
            status[name] = f"INVALID: {e}"

    for name in PRED_FILES:
        run(name, lambda p: check_pred_file(p, warnings))
    run(DYNAMICS_FILE, lambda p: check_dynamics(p, warnings))
    run(SCORES_FILE, lambda p: check_scores(p, train_idx, warnings))

    t4_files = sorted(f for f in present if T4_PATTERN.match(f))
    for name in t4_files:
        run(name, lambda p: check_pred_file(p, warnings))
    t4_ok = [f for f in t4_files if status[f] == "OK"]
    if not t4_files:
        general.append("no t4_<method>.csv found in predictions/ (need 1 to 3)")
    elif len(t4_files) > MAX_T4:
        general.append(f"{len(t4_files)} t4_*.csv files found, but at most {MAX_T4} are allowed")

    completion = {}
    for t, fs in TASKS.items():
        if fs is None:
            completion[t] = 1.0 if (MIN_T4 <= len(t4_files) <= MAX_T4 and len(t4_ok) == len(t4_files)) else 0.0
        else:
            completion[t] = sum(status.get(f) == "OK" for f in fs) / len(fs)

    known = set(PRED_FILES) | {DYNAMICS_FILE, SCORES_FILE} | set(t4_files)
    extra = sorted(present - known)
    if extra:
        warnings.append(f"Unexpected files in predictions/ (ignored): {', '.join(extra)}")

    rj = os.path.join(root, "results.json")
    if not os.path.isfile(rj):
        general.append("results.json is MISSING")
    else:
        try:
            with open(rj, encoding="utf-8") as f:
                results = json.load(f)
            check_results(results, t4_files, general)
        except Exception as e:  # noqa: BLE001
            general.append(f"results.json is not valid JSON: {e}")

    rp = os.path.join(root, "report.md")
    if not os.path.isfile(rp):
        general.append("report.md is MISSING")
    else:
        check_report(rp, general, warnings)
    if not os.path.isfile(os.path.join(root, "code", "README.md")):
        general.append("code/README.md is MISSING (must say how to reproduce each task)")

    return {"status": status, "t4_files": t4_files, "completion": completion,
            "general": general, "warnings": warnings}


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    target = sys.argv[1]
    root, tmp = locate_root(target)
    try:
        if root is None:
            print("ERROR: could not find a 'predictions/' folder in your submission.")
            sys.exit(1)
        res = check_submission(root, load_train_idx())
        warnings = list(res["warnings"])
        if os.path.isfile(target) and not re.fullmatch(r".+_cv_bootcamp\.zip", os.path.basename(target)):
            warnings.append("zip name should be <rollno>_cv_bootcamp.zip")

        names = PRED_FILES[:2] + [DYNAMICS_FILE, "t2_early_stop.csv", SCORES_FILE] + res["t4_files"] \
            + PRED_FILES[3:]
        print("\nFile check")
        print("-" * 60)
        for name in names:
            print(f"  {name:26s} {res['status'][name]}")
        if not res["t4_files"]:
            print(f"  {'t4_<method>.csv':26s} MISSING")

        print("\nTask completeness (1.0 = all files valid)")
        print("-" * 60)
        for t, v in res["completion"].items():
            print(f"  {t}: {v:.2f}")

        if res["general"]:
            print("\nOther issues")
            print("-" * 60)
            for m in res["general"]:
                print("  - " + m)
        if warnings:
            print("\nWarnings (do not block submission)")
            print("-" * 60)
            for m in warnings:
                print("  - " + m)

        ok = all(v == 1.0 for v in res["completion"].values()) and not res["general"]
        print("\n" + ("All good." if ok else "Fix the issues above (or submit what you have)."))
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
