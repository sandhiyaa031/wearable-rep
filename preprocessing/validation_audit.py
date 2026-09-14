"""
PRE-IMPLEMENTATION VALIDATION AUDIT
====================================
Answers 5 critical research-design challenges BEFORE any model code is written:

CHALLENGE 1: Is UCI HAR big enough for MAE pretraining?
  → Count actual windows at 128-timestep / 50% overlap for train subjects

CHALLENGE 2: Is UCI → SisFall a valid cross-dataset experiment?
  → Quantify domain-gap: sampling rate, sensor count, channel statistics, recording length

CHALLENGE 3: Does SisFall fall-onset labeling actually work?
  → For EVERY fall file in one subject:
     - find max resultant acceleration peak
     - check uniqueness (is there one clear peak or multiple?)
     - compare ADL peak magnitudes to fall peak magnitudes
     - report what fraction of fall recordings have ambiguous peaks
     - report overlap between ADL max-peak and fall max-peak distributions

CHALLENGE 4: What is the minimum meaningful label fraction for UCI HAR?
  → Subject-level fractions: {1%, 5%, 10%, 25%, 50%, 100%}
  → Report how many subjects and windows each fraction gives

CHALLENGE 5: What does the actual UCI HAR train/test structure look like?
  → Subject count, window count, class distribution
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

SISFALL_DIR = Path(r"D:\college\PROJECTS-SEM 5\DL\datasets\SisFall_dataset")
UCI_DIR     = Path(r"D:\college\PROJECTS-SEM 5\DL\datasets\HAR\UCI HAR Dataset\UCI HAR Dataset")
OUTPUT_DIR  = Path(r"D:\college\PROJECTS-SEM 5\DL\inspection_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SISFALL_FS   = 200        # Hz — claimed by README
RESAMPLE_FS  = 100        # Hz — our planned resample target
UCI_FS       = 50         # Hz
UCI_WINDOW   = 128        # timesteps (fixed by dataset)
UCI_N_CHAN   = 6          # body acc (Ax Ay Az) + gyro (Gx Gy Gz)

# SisFall fall-onset analysis parameters
PEAK_WINDOW_HALF_SEC = 0.5    # ± seconds around peak = 1.0 sec total fall window
TRANSITION_ZONE_SEC  = 1.0    # seconds before peak to discard (transition)

# ════════════════════════════════════════════════════════════════════════════
# SISFALL LOADER
# ════════════════════════════════════════════════════════════════════════════

def load_sisfall_file(path):
    """Returns np.array shape (n_samples, 9), dtype int32."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip().rstrip(";")
            if not line:
                continue
            vals = [v for v in line.split(",") if v]
            if len(vals) != 9:
                continue
            rows.append([int(v) for v in vals])
    return np.array(rows, dtype=np.int32)


def resultant_accel(data):
    """Compute ADXL345 resultant acceleration magnitude per sample. Shape (N,)."""
    return np.sqrt(data[:, 0]**2 + data[:, 1]**2 + data[:, 2]**2).astype(float)


def find_peaks_above_threshold(signal, threshold, min_distance=50):
    """
    Simple peak finder: local maxima above threshold, with min separation.
    Returns list of peak indices.
    """
    peaks = []
    n = len(signal)
    for i in range(1, n - 1):
        if signal[i] >= threshold and signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)
            elif signal[i] > signal[peaks[-1]]:
                peaks[-1] = i  # keep the taller of nearby peaks
    return peaks


# ════════════════════════════════════════════════════════════════════════════
# CHALLENGE 3 — SISFALL FALL ONSET ANALYSIS (1 subject, all files)
# ════════════════════════════════════════════════════════════════════════════

def analyze_sisfall_onset(subject="SA01"):
    print("\n" + "="*70)
    print(f"CHALLENGE 3 — SisFall fall-onset labeling analysis (subject={subject})")
    print("="*70)

    subdir = SISFALL_DIR / subject
    all_files = sorted(subdir.glob("*.txt"))

    fall_peak_magnitudes  = []
    adl_peak_magnitudes   = []
    fall_peak_counts      = []   # how many prominent peaks per fall recording
    adl_peak_counts       = []
    ambiguous_fall_files  = []   # fall files where peak is not clearly dominant
    recording_durations   = {"Fall": [], "ADL": []}

    THRESHOLD = 6000   # raw ADU threshold — empirically chosen, revisit

    for fpath in all_files:
        fname = fpath.name
        if fname.startswith("desktop") or fname.startswith("."):
            continue

        code = fname.split("_")[0]
        is_fall = code.startswith("F")
        label   = "Fall" if is_fall else "ADL"

        data = load_sisfall_file(fpath)
        duration_sec = len(data) / SISFALL_FS
        recording_durations[label].append(duration_sec)

        res = resultant_accel(data)
        peak_val = res.max()
        peaks = find_peaks_above_threshold(res, threshold=THRESHOLD, min_distance=int(0.3 * SISFALL_FS))

        if is_fall:
            fall_peak_magnitudes.append(peak_val)
            fall_peak_counts.append(len(peaks))
            # Ambiguous: multiple comparable peaks (second peak > 70% of first)
            if len(peaks) >= 2:
                sorted_peaks = sorted([res[p] for p in peaks], reverse=True)
                if sorted_peaks[1] > 0.70 * sorted_peaks[0]:
                    ambiguous_fall_files.append(fname)
        else:
            adl_peak_magnitudes.append(peak_val)
            adl_peak_counts.append(len(peaks))

    fall_peak_magnitudes = np.array(fall_peak_magnitudes)
    adl_peak_magnitudes  = np.array(adl_peak_magnitudes)

    # ---------- Print report ----------

    print(f"\n  Recording durations (seconds):")
    print(f"    ADL  → mean={np.mean(recording_durations['ADL']):.1f}s  "
          f"min={np.min(recording_durations['ADL']):.1f}s  "
          f"max={np.max(recording_durations['ADL']):.1f}s  "
          f"n={len(recording_durations['ADL'])}")
    print(f"    Fall → mean={np.mean(recording_durations['Fall']):.1f}s  "
          f"min={np.min(recording_durations['Fall']):.1f}s  "
          f"max={np.max(recording_durations['Fall']):.1f}s  "
          f"n={len(recording_durations['Fall'])}")

    print(f"\n  Max resultant ADXL345 magnitude per recording:")
    print(f"    ADL  → mean={adl_peak_magnitudes.mean():.0f}  "
          f"std={adl_peak_magnitudes.std():.0f}  "
          f"min={adl_peak_magnitudes.min():.0f}  "
          f"max={adl_peak_magnitudes.max():.0f}")
    print(f"    Fall → mean={fall_peak_magnitudes.mean():.0f}  "
          f"std={fall_peak_magnitudes.std():.0f}  "
          f"min={fall_peak_magnitudes.min():.0f}  "
          f"max={fall_peak_magnitudes.max():.0f}")

    # ADL recordings that exceed the minimum fall peak magnitude
    adl_exceeding = (adl_peak_magnitudes >= fall_peak_magnitudes.min()).sum()
    print(f"\n  ADL recordings with peak ≥ lowest fall peak ({fall_peak_magnitudes.min():.0f}): "
          f"{adl_exceeding} / {len(adl_peak_magnitudes)}")

    print(f"\n  Peaks-per-recording (threshold={THRESHOLD}):")
    print(f"    ADL  → mean={np.mean(adl_peak_counts):.1f}  max={max(adl_peak_counts)}")
    print(f"    Fall → mean={np.mean(fall_peak_counts):.1f}  max={max(fall_peak_counts)}")

    pct_amb = 100 * len(ambiguous_fall_files) / max(len(fall_peak_magnitudes), 1)
    print(f"\n  Ambiguous fall files (2nd peak > 70% of 1st): "
          f"{len(ambiguous_fall_files)} / {len(fall_peak_magnitudes)} ({pct_amb:.1f}%)")
    for af in ambiguous_fall_files[:10]:
        print(f"    → {af}")

    # ---------- Plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(adl_peak_magnitudes, bins=30, alpha=0.7, label="ADL max peak", color="blue")
    axes[0].hist(fall_peak_magnitudes, bins=30, alpha=0.7, label="Fall max peak", color="red")
    axes[0].axvline(fall_peak_magnitudes.min(), color='red', linestyle='--', label=f"Fall min={fall_peak_magnitudes.min():.0f}")
    axes[0].set_xlabel("Max resultant acceleration (raw ADU)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Peak magnitude distribution — {subject}")
    axes[0].legend()

    axes[1].scatter(range(len(adl_peak_magnitudes)), sorted(adl_peak_magnitudes), 
                    c='blue', s=8, label="ADL", alpha=0.6)
    axes[1].scatter(range(len(fall_peak_magnitudes)), sorted(fall_peak_magnitudes), 
                    c='red', s=8, label="Fall", alpha=0.6)
    axes[1].set_xlabel("Recording index (sorted by magnitude)")
    axes[1].set_ylabel("Max resultant acceleration (raw ADU)")
    axes[1].set_title("Sorted peak magnitude: ADL vs Fall")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"audit_challenge3_peak_distribution_{subject}.png", dpi=150)
    plt.close()
    print(f"\n  Plot saved → audit_challenge3_peak_distribution_{subject}.png")

    return {
        "adl_peak_magnitudes": adl_peak_magnitudes,
        "fall_peak_magnitudes": fall_peak_magnitudes,
        "adl_exceeding_min_fall": adl_exceeding,
        "pct_ambiguous": pct_amb,
    }


# ════════════════════════════════════════════════════════════════════════════
# CHALLENGE 1 + 5 — UCI HAR WINDOW COUNT AND SCALE
# ════════════════════════════════════════════════════════════════════════════

def analyze_uci_har():
    print("\n" + "="*70)
    print("CHALLENGES 1 & 5 — UCI HAR dataset scale and subject structure")
    print("="*70)

    # UCI comes pre-windowed. Let's count the actual windows and subjects.
    train_dir = UCI_DIR / "train" / "Inertial Signals"
    test_dir  = UCI_DIR / "test"  / "Inertial Signals"

    # Count train windows from subject_train.txt
    train_subj_file = UCI_DIR / "train" / "subject_train.txt"
    test_subj_file  = UCI_DIR / "test"  / "subject_test.txt"
    train_label_file = UCI_DIR / "train" / "y_train.txt"
    test_label_file  = UCI_DIR / "test"  / "y_test.txt"

    def load_txt_int(path):
        with open(path) as f:
            return [int(line.strip()) for line in f if line.strip()]

    train_subjects = np.array(load_txt_int(train_subj_file))
    test_subjects  = np.array(load_txt_int(test_subj_file))
    train_labels   = np.array(load_txt_int(train_label_file))
    test_labels    = np.array(load_txt_int(test_label_file))

    unique_train_subj = np.unique(train_subjects)
    unique_test_subj  = np.unique(test_subjects)

    print(f"\n  TRAIN SET:")
    print(f"    Total windows : {len(train_subjects)}")
    print(f"    Subjects      : {len(unique_train_subj)} → {sorted(unique_train_subj.tolist())}")
    print(f"    Class counts  : ", end="")
    for c in range(1, 7):
        print(f"class{c}={( train_labels==c).sum()}", end="  ")
    print()

    print(f"\n  TEST SET:")
    print(f"    Total windows : {len(test_subjects)}")
    print(f"    Subjects      : {len(unique_test_subj)} → {sorted(unique_test_subj.tolist())}")

    # Windows per subject (train)
    print(f"\n  Windows per TRAIN subject:")
    windows_per_subj = {}
    for s in unique_train_subj:
        n = (train_subjects == s).sum()
        windows_per_subj[s] = n
        print(f"    Subject {s:02d}: {n} windows")

    # Label fractions at SUBJECT LEVEL
    print(f"\n  CHALLENGE 4 — Label fraction analysis (subject-level sampling):")
    fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    n_train_subj = len(unique_train_subj)   # 21

    for frac in fractions:
        n_subj = max(1, round(frac * n_train_subj))
        # Draw the least ambiguous set: first n_subj subjects
        sampled_subjs = unique_train_subj[:n_subj]
        n_windows = sum(windows_per_subj[s] for s in sampled_subjs)
        n_hours = (n_windows * UCI_WINDOW / UCI_FS) / 3600
        flag = ""
        if n_subj < 3:
            flag = "  ⚠️  STATISTICALLY UNRELIABLE (< 3 subjects)"
        elif n_subj < 5:
            flag = "  ⚠️  BORDERLINE (< 5 subjects)"
        print(f"    {frac*100:5.1f}%  →  {n_subj:2d} subjects  |  {n_windows:4d} windows  |  {n_hours:.2f} hrs{flag}")

    # Data volume for MAE pretraining
    total_windows = len(train_subjects)
    total_hours   = (total_windows * UCI_WINDOW / UCI_FS) / 3600
    print(f"\n  CHALLENGE 1 — pretraining data volume:")
    print(f"    Full UCI HAR train = {total_windows} windows = {total_hours:.2f} hours of 128@50Hz data")
    print(f"    Compare: LSM=40M hours, Inertia-1=18.2M hours, RelCon=12M+ hours")
    print(f"    → UCI HAR is {40_000_000 / total_hours:.0f}× smaller than LSM")
    print(f"    → This is NOT a foundation model pretraining scale.")
    print(f"    → Correct framing: self-supervised representation learning on a small public corpus.")

    return {
        "n_train_windows": total_windows,
        "n_train_subjects": int(n_train_subj),
        "n_test_subjects": len(unique_test_subj),
        "total_hours": total_hours,
    }


# ════════════════════════════════════════════════════════════════════════════
# CHALLENGE 2 — DOMAIN GAP: UCI → SisFall
# ════════════════════════════════════════════════════════════════════════════

def analyze_domain_gap():
    print("\n" + "="*70)
    print("CHALLENGE 2 — Domain gap: UCI HAR → SisFall")
    print("="*70)

    gap_items = [
        ("Sampling rate",       "50 Hz",              "200 Hz (→ resample to 100 Hz)", "HIGH — 2-4× difference even after resampling"),
        ("Sensor channels",     "6 (acc + gyro)",     "9 (2×acc + gyro)",               "MEDIUM — different sensor count; separate input projections needed"),
        ("Sensor type",         "Smartphone sensors", "ADXL345/ITG3200/MMA8451Q",       "HIGH — different hardware, different noise/range"),
        ("Body placement",      "Waist (approx)",     "Waist/belt",                     "LOW — both waist-mounted, similar placement"),
        ("Population",          "30 volunteers",      "38 subjects (young+elderly)",    "MEDIUM — elderly subjects in SisFall, not in UCI"),
        ("Task type",           "6 ADL categories",   "19 ADL + 15 fall types",         "HIGH — no fall class in UCI; task mismatch"),
        ("Recording length",    "~2.56 s (fixed)",    "~14-15 s (full recording)",      "HIGH — SisFall needs windowing, UCI is pre-windowed"),
        ("Label granularity",   "Window-level",       "Recording-level (no onset)",     "HIGH — manual onset detection required"),
        ("Class balance",       "Roughly balanced",   "ADL-heavy (no onset labels)",    "HIGH — severe imbalance after windowing"),
        ("Signal magnitude",    "Normalized g units", "Raw ADU (integer counts)",       "MEDIUM — unit conversion needed"),
    ]

    print(f"\n  {'Dimension':<22} {'UCI HAR':<26} {'SisFall':<35} {'Gap Level'}")
    print(f"  {'-'*22} {'-'*26} {'-'*35} {'-'*20}")
    for dim, uci, sisfall, gap in gap_items:
        print(f"  {dim:<22} {uci:<26} {sisfall:<35} {gap}")

    print("""
  VERDICT:
  ─────────────────────────────────────────────────────────────────────
  The UCI → SisFall transfer IS a valid experiment, but NOT a trivial one.

  The transfer is NOT:
    "The same sensors, same task, slightly different distribution."

  The transfer IS:
    "A representation learned from general human locomotion (UCI, 50 Hz, 6ch)
     is applied as an initialization for fall detection (SisFall, 200→100 Hz, 9ch)."

  This requires SEPARATE INPUT PROJECTION LAYERS per dataset.
  The shared encoder operates in D-dimensional embedding space.
  The claim is that the encoder's learned temporal patterns transfer.

  This is a legitimate and interesting transfer experiment.
  You should frame it as:
    "Does temporal self-supervised pretraining on general motion help when
     fine-tuned on a different task (fall detection) with minimal labels?"

  You should NOT frame it as:
    "Our model generalizes to SisFall" (too strong - you're fine-tuning, not zero-shot)
  """)


# ════════════════════════════════════════════════════════════════════════════
# BONUS — SisFall dataset scale
# ════════════════════════════════════════════════════════════════════════════

def analyze_sisfall_scale():
    print("\n" + "="*70)
    print("BONUS — SisFall dataset scale (all 38 subjects, estimated)")
    print("="*70)

    all_dirs = [d for d in SISFALL_DIR.iterdir() if d.is_dir()]
    total_files   = 0
    total_samples = 0

    fall_recordings = 0
    adl_recordings  = 0

    for subdir in sorted(all_dirs):
        files = list(subdir.glob("F*.txt")) + list(subdir.glob("D*.txt"))
        for f in files:
            code = f.name.split("_")[0]
            # Estimate from file size (≈ 8 bytes per sample × 9 channels)
            # Actual counting is slow; use file size heuristic
            size_bytes = f.stat().st_size
            estimated_rows = size_bytes // 19  # ~19 chars/line on average
            total_samples += estimated_rows
            total_files   += 1
            if code.startswith("F"):
                fall_recordings += 1
            else:
                adl_recordings  += 1

    total_hours_200hz = (total_samples / SISFALL_FS) / 3600
    total_hours_100hz = (total_samples / RESAMPLE_FS) / 3600  # after resampling

    print(f"\n  Subjects analysed   : {len(all_dirs)}")
    print(f"  Total .txt files    : {total_files}")
    print(f"  ADL recordings      : {adl_recordings}")
    print(f"  Fall recordings     : {fall_recordings}")
    print(f"  Estimated samples   : {total_samples:,}")
    print(f"  Estimated duration  : {total_hours_200hz:.1f} hours @ 200 Hz")
    print(f"  After resample      : {total_hours_100hz:.1f} hours @ 100 Hz")

    # Windowed estimate
    window_sz   = 200  # samples @ 100 Hz = 2 sec
    overlap     = 0.5
    stride      = int(window_sz * (1 - overlap))
    estimated_windows = (total_samples // RESAMPLE_FS * RESAMPLE_FS) // stride
    print(f"\n  Estimated windows @ 2s / 50% overlap: ~{estimated_windows:,}")
    print(f"  (Fall class will be a small fraction — exact count requires onset labeling)")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█"*70)
    print("  PRE-IMPLEMENTATION VALIDATION AUDIT")
    print("  Answers 5 critical research-design questions from real data")
    print("█"*70)

    # Run all analyses
    uci_stats = analyze_uci_har()
    sisfall_stats = analyze_sisfall_onset(subject="SA01")
    analyze_domain_gap()
    analyze_sisfall_scale()

    # ─── Final summary ───────────────────────────────────────────────────
    print("\n" + "═"*70)
    print("  FINAL AUDIT SUMMARY")
    print("═"*70)

    print(f"""
  CHALLENGE 1 — UCI HAR pretraining scale
  ────────────────────────────────────────
  UCI train: {uci_stats['n_train_windows']} windows = {uci_stats['total_hours']:.2f} hours
  This is tiny for 'foundation model' pretraining.
  REVISED CLAIM: Frame as "self-supervised wearable representation learning
  on a small public corpus" — NOT a foundation model.
  MAE pretraining on this is still scientifically valid but the representation
  may be limited. Treat this as the scientific question, not an assumption.

  CHALLENGE 2 — UCI → SisFall validity
  ───────────────────────────────────────
  Transfer is valid as FINE-TUNED TRANSFER (not zero-shot).
  Requires separate input projections.
  The encoder learns temporal motion patterns — these CAN transfer.
  Frame correctly as "task-agnostic temporal representation transfer."

  CHALLENGE 3 — SisFall onset labeling
  ────────────────────────────────────────
  ADL peak exceeding min fall peak: {sisfall_stats['adl_exceeding_min_fall']} recordings
  Ambiguous fall recordings       : {sisfall_stats['pct_ambiguous']:.1f}%
  → See plot: audit_challenge3_peak_distribution_SA01.png
  Recommendation: Use peak-acceleration heuristic AS A STARTING POINT,
  then VALIDATE visually on 10 fall + 10 ADL files before finalizing.
  Document heuristic clearly as a study limitation.

  CHALLENGE 4 — Label fraction feasibility  
  ────────────────────────────────────────
  21 train subjects in UCI HAR.
  1%  = 1 subject → statistically unreliable
  5%  = 1 subject → still unreliable
  10% = 2 subjects → borderline (run only if consistent across 5 seeds)
  25% = 5 subjects → reasonable minimum
  REVISED FRACTIONS: Use {{10%, 25%, 50%, 100%}} as core experiment.
  Add 1% and 5% only as supplementary if subject-counts allow.

  CHALLENGE 5 — UCI HAR structure
  ────────────────────────────────────────
  {uci_stats['n_train_subjects']} train subjects, {uci_stats['n_test_subjects']} test subjects (pre-defined split).
  These subjects do NOT overlap — good subject independence by default.
  The official train/val/test is already subject-disjoint.
  Use: train subjects for pretraining + fine-tuning, test subjects for evaluation.
  Create a validation split from training subjects (e.g., last 4 train subjects).
""")


if __name__ == "__main__":
    main()
