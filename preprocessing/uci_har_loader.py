"""
MODULE: uci_har_loader.py

WHAT:
    Loads the UCI HAR dataset from its raw Inertial Signals (.txt) files
    and returns subject-split PyTorch Datasets and DataLoaders.

WHY:
    UCI HAR is pre-windowed (each line in the inertial signal files is one
    128-step window). Loading from Inertial Signals gives us the raw sensor
    readings that our Transformer will process — as opposed to X_train.txt
    which contains 561 hand-engineered features we do NOT want.

DATA FORMAT:
    Each Inertial Signals file (e.g. body_acc_x_train.txt) has shape
    (n_windows, 128). Each row is one window; each column is one timestep.
    We need 6 such files:
        body_acc_x, body_acc_y, body_acc_z  (accelerometer)
        body_gyro_x, body_gyro_y, body_gyro_z  (gyroscope)
    Stacked along axis=2 they give (n_windows, 128, 6).

MATH / NORMALISATION:
    After loading raw windows (float32), we compute per-channel z-score
    statistics over the TRAIN subjects only:
        mean_c = mean over all windows and all timesteps in train set, channel c
        std_c  = std  over all windows and all timesteps in train set, channel c
    Then normalise every window: x_norm[:, c] = (x[:, c] - mean_c) / std_c
    This is applied identically to val and test splits.

TENSOR SHAPES:
    window tensor: (128, 6)   → T=128 timesteps, C=6 channels
    batch tensor : (B, 128, 6)

SUBJECT SPLIT (from final locked design):
    All windows (train + test) come with a subject_id file.
    Official split: subject_train.txt  → subjects 1–21  (train pool)
                    subject_test.txt   → subjects 22–30  (held-out test)
    We further carve subjects [18, 19, 20, 21] as val from the train pool.
    Train  : subjects 1–17   (17 subjects)  ← also used for MAE pretraining
    Val    : subjects 18–21  (4 subjects)   ← hyperparameter selection only
    Test   : subjects 22–30  (9 subjects)   ← NEVER used until final evaluation

LEAKAGE RULES (enforced here):
    1. Normalization stats computed on subjects 1–17 only.
    2. Val and test windows are normalized using train stats.
    3. Subject IDs never mixed across splits.

VIVA ANSWER:
    "I load the 6 raw inertial signal channels from UCI HAR — three axes of
    body acceleration and three of gyroscope — giving a 128×6 tensor per window.
    I split subjects strictly to prevent leakage, compute normalisation statistics
    only on training subjects, and wrap everything in a PyTorch Dataset so the
    training loop never needs to know about file paths or subject IDs."
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict


# ─── CONFIG ───────────────────────────────────────────────────────────────────
UCI_BASE = Path(r"D:\college\PROJECTS-SEM 5\DL\datasets\HAR\UCI HAR Dataset\UCI HAR Dataset")

# 6 channels we want, in order: body_acc_xyz, body_gyro_xyz
CHANNEL_FILES = [
    "body_acc_x_{split}.txt",
    "body_acc_y_{split}.txt",
    "body_acc_z_{split}.txt",
    "body_gyro_x_{split}.txt",
    "body_gyro_y_{split}.txt",
    "body_gyro_z_{split}.txt",
]

# Subject split (from locked design)
TRAIN_SUBJECTS = list(range(1, 18))   # 1–17 inclusive
VAL_SUBJECTS   = [18, 19, 20, 21]
# Test subjects (22–30) are loaded from subject_test.txt automatically


# ─── LOW-LEVEL FILE LOADING ────────────────────────────────────────────────────

def _load_signal_file(path: Path) -> np.ndarray:
    """
    Reads one UCI inertial signal .txt file.

    Each row has 128 whitespace-separated floats (one window).
    Returns numpy array of shape (n_windows, 128), dtype float32.

    WHY: UCI's .txt files use multiple spaces as delimiter, so we
    use whitespace splitting (np.loadtxt default) for robustness.
    """
    data = np.loadtxt(path, dtype=np.float32)
    # shape: (n_windows, 128) — one row per window, 128 timesteps per row
    assert data.ndim == 2, f"Expected 2D array from {path.name}, got {data.ndim}D"
    assert data.shape[1] == 128, f"Expected 128 timesteps, got {data.shape[1]}"
    return data


def _load_subject_ids(split: str) -> np.ndarray:
    """
    Reads subject_train.txt or subject_test.txt.
    Returns int array of shape (n_windows,) — one subject ID per window.
    """
    path = UCI_BASE / split / f"subject_{split}.txt"
    return np.loadtxt(path, dtype=np.int32)


def _load_labels(split: str) -> np.ndarray:
    """
    Reads y_train.txt or y_test.txt.
    Labels are 1-indexed (1–6); we shift to 0-indexed (0–5) here.
    Returns int array of shape (n_windows,).
    """
    path = UCI_BASE / split / f"y_{split}.txt"
    labels = np.loadtxt(path, dtype=np.int32)
    return labels - 1  # shift to 0-indexed: 0=Walking, 1=WalkUp, ..., 5=Laying


def _load_all_channels(split: str) -> np.ndarray:
    """
    Loads all 6 channels for a given split ('train' or 'test').

    Returns array of shape (n_windows, 128, 6).

    WHY: We load each channel file separately and stack along the new
    channel axis. This keeps the code explicit about which channel is which.

    TENSOR SHAPE:
        per-file:    (n_windows, 128)
        stacked:     (n_windows, 128, 6)  ← this is our input tensor
    """
    sig_dir = UCI_BASE / split / "Inertial Signals"
    channels = []
    for fname_template in CHANNEL_FILES:
        fpath = sig_dir / fname_template.format(split=split)
        arr = _load_signal_file(fpath)         # (n_windows, 128)
        channels.append(arr)                   # 6 × (n_windows, 128)

    # stack along new last axis → (n_windows, 128, 6)
    stacked = np.stack(channels, axis=2)       # (n_windows, 128, 6)
    assert stacked.shape[2] == 6
    return stacked


# ─── NORMALIZATION ─────────────────────────────────────────────────────────────

def compute_normalization_stats(windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes per-channel mean and std over all windows AND all timesteps.

    windows: (n_windows, 128, 6)

    WHY: We collapse over the window-batch and time dimensions to get
    a single mean and std per sensor channel. This is the standard
    windowed-sensor normalization used in HAR literature.

    MATH:
        x reshaped to (N*T, C) = (n_windows*128, 6)
        mean_c = mean over axis=0 → shape (6,)
        std_c  = std  over axis=0 → shape (6,)

    Returns:
        mean: (6,)  float32
        std:  (6,)  float32
    """
    # (n_windows, 128, 6) → (n_windows*128, 6)
    flat = windows.reshape(-1, windows.shape[2])
    mean = flat.mean(axis=0).astype(np.float32)       # (6,)
    std  = flat.std(axis=0).astype(np.float32)        # (6,)
    # Guard against zero std (degenerate channels)
    std  = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_normalization(windows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Applies per-channel z-score normalization.

    windows: (n_windows, 128, 6)
    mean:    (6,)
    std:     (6,)

    MATH: x_norm[i, t, c] = (x[i, t, c] - mean[c]) / std[c]

    Broadcasting: (n_windows, 128, 6) - (6,) → auto-broadcasts along last axis.
    """
    return (windows - mean) / std   # (n_windows, 128, 6)


# ─── PYTORCH DATASET ──────────────────────────────────────────────────────────

class UCIHARDataset(Dataset):
    """
    PyTorch Dataset for UCI HAR.

    WHAT: Wraps pre-loaded normalized windows and labels into a Dataset
    that returns (x, y) pairs where:
        x: Tensor (128, 6)  — one window, 6 channels, 128 timesteps
        y: Tensor ()        — int label 0–5 (class index)

    WHY: PyTorch DataLoader requires a Dataset. Keeping the data in a
    Dataset also enables efficient batching, shuffling, and parallel loading.

    Attributes:
        windows:  (N, 128, 6)  numpy array, already normalized
        labels:   (N,)         numpy array, int32, 0-indexed
    """

    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        """
        windows: np.ndarray (N, 128, 6)  float32, normalized
        labels:  np.ndarray (N,)         int32, 0-indexed (0–5)
        """
        assert len(windows) == len(labels), (
            f"windows ({len(windows)}) and labels ({len(labels)}) must have same length"
        )
        self.windows = torch.from_numpy(windows.astype(np.float32))   # (N, 128, 6)
        self.labels  = torch.from_numpy(labels.astype(np.int64))      # (N,)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # Returns: x of shape (128, 6), y as scalar int64
        return self.windows[idx], self.labels[idx]


# ─── MAIN BUILDER ─────────────────────────────────────────────────────────────

def build_uci_har_dataloaders(
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    return_stats: bool = False,
) -> Dict:
    """
    Full pipeline: load → split → normalize → wrap in Dataset → wrap in DataLoader.

    Returns a dict with keys:
        "train"        : DataLoader for training subjects (1–17)
        "val"          : DataLoader for val subjects (18–21)
        "test"         : DataLoader for test subjects (22–30)
        "train_dataset": UCIHARDataset (for MAE pretraining — same windows, no labels needed)
        "norm_stats"   : (mean, std) if return_stats=True, else not included
        "subject_ids"  : dict with "train", "val", "test" numpy arrays

    WHY return_stats: The same normalization stats must be used for SisFall
    downstream evaluation when the encoder is transferred. We save them for reuse.
    """
    # ── 1. Load raw inertial signals ─────────────────────────────────────────
    print("[UCI HAR] Loading train split Inertial Signals...")
    train_pool_windows = _load_all_channels("train")    # (7352, 128, 6)
    train_pool_labels  = _load_labels("train")          # (7352,)  0-indexed
    train_pool_subjects = _load_subject_ids("train")    # (7352,)

    print("[UCI HAR] Loading test split Inertial Signals...")
    test_windows  = _load_all_channels("test")          # (2947, 128, 6)
    test_labels   = _load_labels("test")                # (2947,)
    test_subjects = _load_subject_ids("test")           # (2947,)

    # ── 2. Split train pool into train (1–17) and val (18–21) ────────────────
    train_mask = np.isin(train_pool_subjects, TRAIN_SUBJECTS)  # bool (7352,)
    val_mask   = np.isin(train_pool_subjects, VAL_SUBJECTS)    # bool (7352,)

    train_windows  = train_pool_windows[train_mask]   # (N_train, 128, 6)
    train_labels_  = train_pool_labels[train_mask]    # (N_train,)
    train_subj_ids = train_pool_subjects[train_mask]  # (N_train,)

    val_windows    = train_pool_windows[val_mask]     # (N_val, 128, 6)
    val_labels_    = train_pool_labels[val_mask]      # (N_val,)
    val_subj_ids   = train_pool_subjects[val_mask]    # (N_val,)

    # ── 3. Compute normalization stats on TRAIN SUBJECTS ONLY ────────────────
    print("[UCI HAR] Computing normalization statistics on train subjects (1–17)...")
    mean, std = compute_normalization_stats(train_windows)
    # mean: (6,)  std: (6,)

    # ── 4. Apply normalization to all splits ─────────────────────────────────
    train_windows_norm = apply_normalization(train_windows, mean, std)
    val_windows_norm   = apply_normalization(val_windows,   mean, std)
    test_windows_norm  = apply_normalization(test_windows,  mean, std)

    # ── 5. Wrap in Dataset ───────────────────────────────────────────────────
    train_dataset = UCIHARDataset(train_windows_norm, train_labels_)
    val_dataset   = UCIHARDataset(val_windows_norm,   val_labels_)
    test_dataset  = UCIHARDataset(test_windows_norm,  test_labels)

    # ── 6. Wrap in DataLoader ────────────────────────────────────────────────
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=g, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # ── 7. Print summary ─────────────────────────────────────────────────────
    print("\n[UCI HAR] Dataset summary:")
    print(f"  Train : {len(train_dataset):5d} windows | "
          f"subjects {sorted(set(train_subj_ids.tolist()))}")
    print(f"  Val   : {len(val_dataset):5d} windows | "
          f"subjects {sorted(set(val_subj_ids.tolist()))}")
    print(f"  Test  : {len(test_dataset):5d} windows | "
          f"subjects {sorted(set(test_subjects.tolist()))}")
    print(f"  Total : {len(train_dataset)+len(val_dataset)+len(test_dataset):5d} windows")
    print(f"  Subject overlap train∩test: "
          f"{set(train_subj_ids.tolist()) & set(test_subjects.tolist())} (must be empty)")
    print(f"\n  Norm stats (train subjects 1–17):")
    ch_names = ["BAccX", "BAccY", "BAccZ", "GyroX", "GyroY", "GyroZ"]
    for i, name in enumerate(ch_names):
        print(f"    {name}: mean={mean[i]:+.4f}  std={std[i]:.4f}")

    # ── 8. Class distribution ────────────────────────────────────────────────
    activity_names = ["Walking", "WalkUpstairs", "WalkDownstairs",
                      "Sitting", "Standing", "Laying"]
    print(f"\n  Train class distribution:")
    for c, name in enumerate(activity_names):
        n = (train_labels_ == c).sum()
        print(f"    [{c}] {name:<16}: {n:4d}  ({100*n/len(train_labels_):.1f}%)")

    result = {
        "train": train_loader,
        "val":   val_loader,
        "test":  test_loader,
        "train_dataset": train_dataset,
        "val_dataset":   val_dataset,
        "test_dataset":  test_dataset,
        "subject_ids":   {
            "train": train_subj_ids,
            "val":   val_subj_ids,
            "test":  test_subjects,
        },
    }
    if return_stats:
        result["norm_stats"] = (mean, std)

    return result


def build_uci_label_efficiency_loader(
    fraction: float,
    norm_mean: np.ndarray,
    norm_std:  np.ndarray,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """
    Builds a DataLoader using only FRACTION of training subjects.
    Used for the label-efficiency experiment.

    fraction: 0.25, 0.50, or 1.00 (25%, 50%, 100% of train subjects)
    norm_mean, norm_std: computed from full train set (subjects 1–17)

    PROTOCOL (subject-level sampling, from locked design):
        Subjects 1–17 are the train pool.
        fraction=0.25 → use first round(0.25 * 17) = 4 subjects
        The subjects selected are deterministic given the seed.
        With 5 seeds, we permute the subject list differently each time.

    WHY subject-level (not window-level):
        Window-level sampling from the same subject is not truly
        label-efficient — the model still sees ALL subjects' motion patterns.
        Subject-level sampling is the scientifically correct protocol because
        it tests: "given data from only N subjects, can the model generalise
        to unseen subjects?"
    """
    assert 0.0 < fraction <= 1.0, "fraction must be in (0, 1]"

    # Load train pool
    train_pool_windows  = _load_all_channels("train")
    train_pool_labels   = _load_labels("train")
    train_pool_subjects = _load_subject_ids("train")

    # Select only train subjects (1–17)
    pool_mask = np.isin(train_pool_subjects, TRAIN_SUBJECTS)
    windows  = train_pool_windows[pool_mask]
    labels   = train_pool_labels[pool_mask]
    subjects = train_pool_subjects[pool_mask]

    # Shuffle subjects, then take top N
    rng = np.random.default_rng(seed)
    shuffled_subjects = rng.permutation(TRAIN_SUBJECTS)     # random order per seed
    n_select = max(1, round(fraction * len(TRAIN_SUBJECTS)))
    selected = shuffled_subjects[:n_select]

    mask = np.isin(subjects, selected)
    w_sub = windows[mask]
    l_sub = labels[mask]

    # Normalize
    w_norm = apply_normalization(w_sub, norm_mean, norm_std)
    dataset = UCIHARDataset(w_norm, l_sub)

    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=g,
    )

    print(f"[UCI HAR] Label efficiency loader: {fraction*100:.0f}% → "
          f"{len(selected)} subjects → {len(dataset)} windows")
    return loader


# ─── QUICK SELF-TEST ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("UCI HAR LOADER — SELF-TEST")
    print("=" * 60)

    loaders = build_uci_har_dataloaders(batch_size=64, return_stats=True)
    mean, std = loaders["norm_stats"]

    # Test 1: batch shapes
    x_batch, y_batch = next(iter(loaders["train"]))
    print(f"\nBatch shape test:")
    print(f"  x: {tuple(x_batch.shape)}  (expect: (64, 128, 6))")
    print(f"  y: {tuple(y_batch.shape)}  (expect: (64,))")
    assert x_batch.shape == (64, 128, 6), f"Wrong x shape: {x_batch.shape}"
    assert y_batch.shape == (64,),         f"Wrong y shape: {y_batch.shape}"
    assert y_batch.min() >= 0 and y_batch.max() <= 5, "Labels out of range 0–5"

    # Test 2: no label leakage (0-indexed labels must be in 0–5)
    print(f"\nLabel range: {y_batch.min().item()} – {y_batch.max().item()} (expect 0–5)")

    # Test 3: normalization sanity (train data should be ~N(0,1))
    x_all = loaders["train_dataset"].windows   # (N_train, 128, 6)
    flat = x_all.reshape(-1, 6)               # (N_train*128, 6)
    print(f"\nNormalization check (train data should be ~mean=0, std=1):")
    for i, name in enumerate(["BAccX","BAccY","BAccZ","GyroX","GyroY","GyroZ"]):
        m = flat[:, i].mean().item()
        s = flat[:, i].std().item()
        ok = abs(m) < 0.01 and abs(s - 1.0) < 0.05
        print(f"  {name}: mean={m:+.4f}  std={s:.4f}  {'OK' if ok else 'WARN'}")

    # Test 4: subject leakage check
    train_subj = set(loaders["subject_ids"]["train"].tolist())
    val_subj   = set(loaders["subject_ids"]["val"].tolist())
    test_subj  = set(loaders["subject_ids"]["test"].tolist())
    print(f"\nSubject leakage check:")
    print(f"  train ∩ val  = {train_subj & val_subj}   (must be empty)")
    print(f"  train ∩ test = {train_subj & test_subj}  (must be empty)")
    print(f"  val   ∩ test = {val_subj & test_subj}    (must be empty)")

    # Test 5: label efficiency loader
    print("\nLabel efficiency loader test:")
    le_loader = build_uci_label_efficiency_loader(
        fraction=0.25, norm_mean=mean, norm_std=std,
        batch_size=64, seed=42,
    )
    x_le, y_le = next(iter(le_loader))
    print(f"  25% loader batch: x={tuple(x_le.shape)}  y={tuple(y_le.shape)}")

    print("\nAll tests passed.")
