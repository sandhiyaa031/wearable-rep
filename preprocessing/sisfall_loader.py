"""
MODULE: sisfall_loader.py

WHAT:
    Loads the SisFall dataset from raw .txt files, applies 200Hz -> 100Hz
    anti-aliasing downsampling, applies sliding windowing, applies
    recording-level labeling (Option A), and returns PyTorch DataLoaders.

WHY:
    Unlike UCI HAR, SisFall is not pre-windowed and the sampling rate
    (200Hz) is unnecessarily high for human motion, and would result in
    very long sequences (400 items per 2s window) that scale quadratically
    in Transformer self-attention. We downsample to 100Hz (200 sequence length).
    The visual inspection audit confirmed that peak-centered labeling is
    unreliable without manual per-file validation, so we use the robust
    Recording-level labeling (Option A).

MATH / NORMALISATION:
    Resampling: scipy.signal.resample (or decimate) with anti-aliasing.
    Windowing: 2s window @ 100Hz = 200 samples. 50% overlap = 100 sample stride.
    Normalisation: per-channel z-score fit on TRAIN subjects only.

TENSOR SHAPES:
    Raw file     : (N_raw_samples, 9) @ 200 Hz
    Downsampled  : (N_raw_samples/2, 9) @ 100 Hz
    Windowed     : (N_windows, 200, 9)
    Output batch : (B, 200, 9)

SUBJECT SPLIT (from final locked design):
    Train (25): SA01–SA16, SE01–SE09
    Val (7)   : SA17–SA20, SE10–SE12
    Test (6)  : SA21–SA23, SE13–SE15

LEAKAGE RULES (enforced here):
    1. Normalization stats computed on Train subjects only.
    2. Overlapping windows never cross recordings.
    3. No subject overlap across splits.

VIVA ANSWER:
    "For SisFall, I load the raw 200Hz recordings, downsample them to 100Hz
    using an anti-aliasing filter to reduce sequence length, and extract
    2-second windows with 50% overlap. Because my visual inspection audit proved
    that reliable fall onset cannot be systematically derived from peak acceleration
    alone across the whole dataset, I use recording-level labels where all
    windows from a fall recording are treated as fall class, creating a robust,
    noise-tolerant fine-tuning target."
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.signal import decimate

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SISFALL_DIR = Path(r"D:\college\PROJECTS-SEM 5\DL\datasets\SisFall_dataset")

# Fixed subject splits
TRAIN_SUBJECTS = (
    [f"SA{i:02d}" for i in range(1, 17)] +
    [f"SE{i:02d}" for i in range(1, 10)]
)
VAL_SUBJECTS = (
    [f"SA{i:02d}" for i in range(17, 21)] +
    [f"SE{i:02d}" for i in range(10, 13)]
)
TEST_SUBJECTS = (
    [f"SA{i:02d}" for i in range(21, 24)] +
    [f"SE{i:02d}" for i in range(13, 16)]
)

FS_RAW       = 200
FS_TARGET    = 100
WINDOW_SEC   = 2.0
OVERLAP_FRAC = 0.5
WINDOW_LEN   = int(WINDOW_SEC * FS_TARGET)               # 200
STRIDE       = int(WINDOW_LEN * (1.0 - OVERLAP_FRAC))    # 100


# ─── PYTORCH DATASET ──────────────────────────────────────────────────────────

class SisFallDataset(Dataset):
    """
    Returns:
        x: (200, 9) float32, normalized
        y: () int64  (0 = ADL, 1 = Fall)
    """
    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        assert len(windows) == len(labels)
        self.windows = torch.from_numpy(windows.astype(np.float32))
        self.labels  = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.windows[idx], self.labels[idx]


# ─── PREPROCESSING FUNCTIONS ──────────────────────────────────────────────────

def load_sisfall_file(path: Path) -> np.ndarray:
    """Reads raw .txt, returns (N, 9) int32."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().rstrip(";")
            if not line: continue
            vals = [v for v in line.split(",") if v]
            if len(vals) != 9: continue
            rows.append([int(v) for v in vals])
    return np.array(rows, dtype=np.int32)


def process_subject_folder(subj_dir: Path):
    """
    Reads all D*.txt and F*.txt in a subject directory.
    Downsamples to 100Hz, applies sliding window.
    Returns:
        windows: list of (200, 9) arrays
        labels:  list of ints (0 or 1)
    """
    all_windows = []
    all_labels  = []

    for fpath in sorted(subj_dir.glob("[DF]*.txt")):
        code = fpath.name.split("_")[0]
        label = 1 if code.startswith("F") else 0  # OPTION A labeling

        raw_data = load_sisfall_file(fpath)
        if len(raw_data) == 0:
            continue

        # Downsample 200Hz -> 100Hz (factor of 2)
        # Using decimate gives anti-aliasing Chebyshev filter by default
        # downsample along time axis (axis 0)
        # We process each channel to avoid large memory spikes
        ds_cols = []
        for c in range(9):
            ds_cols.append(decimate(raw_data[:, c], q=2, zero_phase=True))
        data_100hz = np.column_stack(ds_cols)  # (N/2, 9)

        # Sliding window
        n_samples = data_100hz.shape[0]
        for start_idx in range(0, n_samples - WINDOW_LEN + 1, STRIDE):
            window = data_100hz[start_idx : start_idx + WINDOW_LEN, :]
            all_windows.append(window)
            all_labels.append(label)

    return all_windows, all_labels


def _load_split_data(subjects: list):
    """Loads all subjects in a split. Returns (N, 200, 9) and (N,)."""
    windows, labels = [], []
    for subj in subjects:
        subj_dir = SISFALL_DIR / subj
        if not subj_dir.exists():
            print(f"[SisFall Loader] WARNING: Directory not found: {subj_dir}")
            continue
        w, l = process_subject_folder(subj_dir)
        windows.extend(w)
        labels.extend(l)
    
    if not windows:
        return np.empty((0, WINDOW_LEN, 9)), np.empty((0,))
        
    return np.stack(windows, axis=0), np.array(labels, dtype=np.int32)


def compute_normalization_stats(windows: np.ndarray):
    """
    windows: (N, 200, 9)
    Returns mean (9,), std (9,)
    """
    flat = windows.reshape(-1, windows.shape[2])
    mean = flat.mean(axis=0).astype(np.float32)
    std  = flat.std(axis=0).astype(np.float32)
    std  = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_normalization(windows: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Applies z-score normalisation."""
    return (windows - mean) / std


# ─── MAIN BUILDER ─────────────────────────────────────────────────────────────

def build_sisfall_dataloaders(
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42
):
    print("[SisFall] Loading train split...")
    train_windows, train_labels = _load_split_data(TRAIN_SUBJECTS)
    print(f"          -> {len(train_windows)} windows")

    print("[SisFall] Loading val split...")
    val_windows, val_labels = _load_split_data(VAL_SUBJECTS)
    print(f"          -> {len(val_windows)} windows")

    print("[SisFall] Loading test split...")
    test_windows, test_labels = _load_split_data(TEST_SUBJECTS)
    print(f"          -> {len(test_windows)} windows")

    print("[SisFall] Computing training normalisation stats...")
    mean, std = compute_normalization_stats(train_windows)

    train_windows_norm = apply_normalization(train_windows, mean, std)
    val_windows_norm   = apply_normalization(val_windows, mean, std)
    test_windows_norm  = apply_normalization(test_windows, mean, std)

    train_dataset = SisFallDataset(train_windows_norm, train_labels)
    val_dataset   = SisFallDataset(val_windows_norm, val_labels)
    test_dataset  = SisFallDataset(test_windows_norm, test_labels)

    g = torch.Generator()
    g.manual_seed(seed)

    # Class imbalance in training? Yes (ADL heavy).
    # Since Option A marks all fall recording windows as 1, the imbalance
    # is less severe than peak labeling, but still exists. We just shuffle.
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=g, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print("\n[SisFall] Summary:")
    print(f"  Train : {len(train_dataset):6d} windows | ADL: {(train_labels==0).sum()} | Fall: {(train_labels==1).sum()}")
    print(f"  Val   : {len(val_dataset):6d} windows | ADL: {(val_labels==0).sum()} | Fall: {(val_labels==1).sum()}")
    print(f"  Test  : {len(test_dataset):6d} windows | ADL: {(test_labels==0).sum()} | Fall: {(test_labels==1).sum()}")
    print(f"  Subject overlap check passed? (implicit by fixed lists: {len(set(TRAIN_SUBJECTS)&set(TEST_SUBJECTS)) == 0})")
    print(f"  Norm stats (Train only):")
    ch_names = ["ADXL_X","ADXL_Y","ADXL_Z", "ITG_X","ITG_Y","ITG_Z", "MMA_X","MMA_Y","MMA_Z"]
    for i, name in enumerate(ch_names):
        print(f"    {name:<6}: mean={mean[i]:+.3f}  std={std[i]:.3f}")

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "norm_stats": (mean, std),
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset
    }


# ─── QUICK SELF-TEST ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SISFALL LOADER — SELF-TEST")
    print("=" * 60)
    
    # We will test on a tiny subset so the script doesn't take 5 minutes
    # overriding the sets temporarily
    old_tr, old_va, old_te = TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS
    TRAIN_SUBJECTS = ["SA01"]
    VAL_SUBJECTS   = ["SA02"]
    TEST_SUBJECTS  = ["SA03"]

    loaders = build_sisfall_dataloaders(batch_size=32)

    # Revert
    TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS = old_tr, old_va, old_te

    # 1. Check shapes
    x, y = next(iter(loaders["train"]))
    print(f"\nBatch shape test:")
    print(f"  x: {tuple(x.shape)} (expect (32, 200, 9))")
    print(f"  y: {tuple(y.shape)} (expect (32,))")
    assert x.shape == (32, 200, 9)
    assert y.shape == (32,)

    # 2. Check normalization
    m, s = loaders["norm_stats"]
    assert m.shape == (9,)
    assert s.shape == (9,)
    
    # Check that x is actually normalized
    x_np = loaders["train_dataset"].windows
    flat = x_np.reshape(-1, 9)
    mx = flat.mean(axis=0)
    sx = flat.std(axis=0)
    print(f"\nTrain window norm sanity:")
    print(f"  mean = {mx.mean():.4f} (expect ~0.0)")
    print(f"  std  = {sx.mean():.4f} (expect ~1.0)")
    
    print("\nAll tests passed (on subset).")
