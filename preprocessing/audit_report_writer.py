"""
Audit report writer — ASCII-safe, writes directly to a .txt file.
Runs all 5 challenge analyses and produces inspection_results/AUDIT_REPORT.txt
"""

import numpy as np
import sys
import io
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SISFALL_DIR = Path(r"D:\college\PROJECTS-SEM 5\DL\datasets\SisFall_dataset")
UCI_DIR     = Path(r"D:\college\PROJECTS-SEM 5\DL\datasets\HAR\UCI HAR Dataset\UCI HAR Dataset")
OUTPUT_DIR  = Path(r"D:\college\PROJECTS-SEM 5\DL\inspection_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUTPUT_DIR / "AUDIT_REPORT.txt"

SISFALL_FS  = 200
RESAMPLE_FS = 100
UCI_FS      = 50
UCI_WINDOW  = 128


def w(lines, msg=""):
    """Write a line to the report buffer."""
    lines.append(msg)


def load_sisfall_file(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
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
    return np.sqrt(data[:, 0]**2 + data[:, 1]**2 + data[:, 2]**2).astype(float)


def find_dominant_peaks(signal, min_distance=50):
    """Find local maxima with minimum separation."""
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            if not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)
            elif signal[i] > signal[peaks[-1]]:
                peaks[-1] = i
    return peaks


def analyze_sisfall_onset(lines, subject="SA01"):
    w(lines)
    w(lines, "=" * 70)
    w(lines, f"CHALLENGE 3 -- SisFall fall-onset labeling (subject={subject})")
    w(lines, "=" * 70)

    subdir = SISFALL_DIR / subject
    all_files = sorted(subdir.glob("[DF]*.txt"))

    fall_peaks, adl_peaks = [], []
    fall_multi_peak, adl_multi_peak = 0, 0
    ambiguous_files = []
    fall_durations, adl_durations = [], []
    per_fall_detail = []

    for fpath in all_files:
        code = fpath.name.split("_")[0]
        is_fall = code.startswith("F")  
        data = load_sisfall_file(fpath)
        dur  = len(data) / SISFALL_FS
        res  = resultant_accel(data)
        peak_val = float(res.max())
        peaks = find_dominant_peaks(res, min_distance=int(0.25 * SISFALL_FS))

        if is_fall:
            fall_peaks.append(peak_val)
            fall_durations.append(dur)
            if len(peaks) >= 2:
                fall_multi_peak += 1
                top2 = sorted([res[p] for p in peaks], reverse=True)[:2]
                if top2[1] > 0.70 * top2[0]:
                    ambiguous_files.append(fpath.name)
            per_fall_detail.append((fpath.name, peak_val, len(peaks), dur))
        else:
            adl_peaks.append(peak_val)
            adl_durations.append(dur)
            if len(peaks) >= 2:
                adl_multi_peak += 1

    fall_peaks = np.array(fall_peaks)
    adl_peaks  = np.array(adl_peaks)

    w(lines, f"  ADL  recordings : {len(adl_peaks)}")
    w(lines, f"  Fall recordings : {len(fall_peaks)}")
    w(lines)
    w(lines, "  Recording durations (seconds):")
    w(lines, f"    ADL   mean={np.mean(adl_durations):.1f}  min={np.min(adl_durations):.1f}  max={np.max(adl_durations):.1f}")
    w(lines, f"    Fall  mean={np.mean(fall_durations):.1f}  min={np.min(fall_durations):.1f}  max={np.max(fall_durations):.1f}")
    w(lines)
    w(lines, "  Max resultant ADXL345 acceleration per recording (raw ADU):")
    w(lines, f"    ADL   mean={adl_peaks.mean():.0f}  std={adl_peaks.std():.0f}  "
             f"min={adl_peaks.min():.0f}  max={adl_peaks.max():.0f}")
    w(lines, f"    Fall  mean={fall_peaks.mean():.0f}  std={fall_peaks.std():.0f}  "
             f"min={fall_peaks.min():.0f}  max={fall_peaks.max():.0f}")
    w(lines)

    adl_overlap = int((adl_peaks >= fall_peaks.min()).sum())
    fall_separable = int((fall_peaks > adl_peaks.max()).sum())
    w(lines, f"  ADL recordings with peak >= minimum fall peak "
             f"({fall_peaks.min():.0f}): {adl_overlap} / {len(adl_peaks)}")
    w(lines, f"  Fall recordings with peak > maximum ADL peak "
             f"({adl_peaks.max():.0f}): {fall_separable} / {len(fall_peaks)}")
    w(lines)
    w(lines, f"  Multi-peak recordings (>= 2 prominent peaks at 0.25s min gap):")
    w(lines, f"    ADL  : {adl_multi_peak} / {len(adl_peaks)} "
             f"({100*adl_multi_peak/max(len(adl_peaks),1):.0f}%)")
    w(lines, f"    Fall : {fall_multi_peak} / {len(fall_peaks)} "
             f"({100*fall_multi_peak/max(len(fall_peaks),1):.0f}%)")
    w(lines)
    pct_amb = 100 * len(ambiguous_files) / max(len(fall_peaks), 1)
    w(lines, f"  Ambiguous fall files (2nd peak > 70% of max): "
             f"{len(ambiguous_files)} / {len(fall_peaks)} ({pct_amb:.1f}%)")
    for af in ambiguous_files[:8]:
        w(lines, f"    -> {af}")
    w(lines)
    w(lines, "  Per-fall-type summary (first 15 fall types):")
    w(lines, f"    {'File':<22} {'MaxPeak':>9} {'N_peaks':>8} {'Dur(s)':>8}")
    for fname, pv, np_, dur in sorted(per_fall_detail, key=lambda x: x[0])[:15]:
        w(lines, f"    {fname:<22} {pv:>9.0f} {np_:>8d} {dur:>8.1f}")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(adl_peaks, bins=25, alpha=0.65, color="steelblue", label=f"ADL (n={len(adl_peaks)})")
    ax.hist(fall_peaks, bins=25, alpha=0.65, color="tomato",  label=f"Fall (n={len(fall_peaks)})")
    ax.axvline(fall_peaks.min(), color="red",  linestyle="--", lw=1.5, label=f"Fall min={fall_peaks.min():.0f}")
    ax.axvline(adl_peaks.max(),  color="blue", linestyle="--", lw=1.5, label=f"ADL max={adl_peaks.max():.0f}")
    ax.set_xlabel("Max resultant acceleration (raw ADU, ADXL345)")
    ax.set_ylabel("Count (recordings)")
    ax.set_title(f"Challenge 3: ADL vs Fall peak magnitude distribution -- {subject}")
    ax.legend()
    plt.tight_layout()
    out = OUTPUT_DIR / f"audit_ch3_peak_distribution_{subject}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    w(lines, f"  Plot saved -> {out.name}")

    return {
        "adl_exceeding_minfall": adl_overlap,
        "fall_separable": fall_separable,
        "pct_ambiguous": pct_amb,
        "adl_max": float(adl_peaks.max()),
        "fall_min": float(fall_peaks.min()),
        "fall_mean": float(fall_peaks.mean()),
        "n_fall_recordings": len(fall_peaks),
        "n_adl_recordings": len(adl_peaks),
    }


def load_uci_int_file(path):
    vals = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s:
                vals.append(int(s))
    return np.array(vals, dtype=np.int32)


def analyze_uci(lines):
    w(lines)
    w(lines, "=" * 70)
    w(lines, "CHALLENGES 1 + 4 + 5 -- UCI HAR dataset scale, subjects, label fractions")
    w(lines, "=" * 70)

    train_subj   = load_uci_int_file(UCI_DIR / "train" / "subject_train.txt")
    test_subj    = load_uci_int_file(UCI_DIR / "test"  / "subject_test.txt")
    train_labels = load_uci_int_file(UCI_DIR / "train" / "y_train.txt")
    test_labels  = load_uci_int_file(UCI_DIR / "test"  / "y_test.txt")

    uniq_train = np.unique(train_subj)
    uniq_test  = np.unique(test_subj)
    class_names = {1:"Walking", 2:"WalkUp", 3:"WalkDown", 4:"Sitting", 5:"Standing", 6:"Laying"}

    w(lines)
    w(lines, f"  TRAIN: {len(train_subj)} windows across {len(uniq_train)} subjects")
    w(lines, f"  TEST : {len(test_subj)}  windows across {len(uniq_test)} subjects")
    w(lines, f"  Subject overlap: {set(uniq_train.tolist()) & set(uniq_test.tolist())} (must be empty)")
    w(lines)
    w(lines, "  Class distribution (train):")
    for c, name in class_names.items():
        n = int((train_labels == c).sum())
        w(lines, f"    class {c} {name:<12}: {n:4d} windows ({100*n/len(train_labels):.1f}%)")
    w(lines)
    w(lines, "  Windows per train subject:")
    wpsubj = {}
    for s in uniq_train:
        n = int((train_subj == s).sum())
        wpsubj[s] = n
        w(lines, f"    Subject {s:02d}: {n} windows")

    total_hours = (len(train_subj) * UCI_WINDOW / UCI_FS) / 3600
    w(lines)
    w(lines, f"  CHALLENGE 1 -- Pretraining data volume:")
    w(lines, f"    Total train windows : {len(train_subj)}")
    w(lines, f"    Total hours         : {total_hours:.3f} h  (= {len(train_subj)} x 2.56s / 3600)")
    w(lines, f"    Compare to LSM      : ~{40_000_000 / total_hours:,.0f}x smaller than 40M-hour corpus")
    w(lines, f"    Compare to Inertia-1: ~{18_200_000 / total_hours:,.0f}x smaller than 18.2M-hour corpus")
    w(lines, f"    VERDICT: This is NOT a foundation-model-scale corpus.")
    w(lines, f"    Correct framing: 'self-supervised wearable representation learning")
    w(lines, f"    on a small public benchmark; the scientific value lies in the")
    w(lines, f"    evaluation protocol, not pretraining scale.'")

    w(lines)
    w(lines, "  CHALLENGE 4 -- Label fraction feasibility (subject-level sampling):")
    w(lines, f"    Total train subjects: {len(uniq_train)}")
    w(lines, f"    {'Fraction':>10} {'N_Subj':>8} {'N_Windows':>11} {'Hours':>8}  {'Verdict'}")
    w(lines, f"    {'-'*10} {'-'*8} {'-'*11} {'-'*8}  {'-'*30}")
    fracs = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    for frac in fracs:
        n_subj = max(1, round(frac * len(uniq_train)))
        sampled = uniq_train[:n_subj]
        n_win   = sum(wpsubj[s] for s in sampled)
        hrs     = (n_win * UCI_WINDOW / UCI_FS) / 3600
        if n_subj < 3:
            verdict = "UNRELIABLE  (< 3 subjects)"
        elif n_subj < 5:
            verdict = "BORDERLINE  (< 5 subjects, only with 5+ seeds)"
        else:
            verdict = "OK"
        w(lines, f"    {frac*100:>9.1f}% {n_subj:>8d} {n_win:>11d} {hrs:>8.3f}  {verdict}")

    w(lines)
    w(lines, "  REVISED LABEL FRACTIONS (recommended):")
    w(lines, "    Core experiment : {10%, 25%, 50%, 100%}  (n_subj = 2, 5, 10, 21)")
    w(lines, "    Note: 10% = 2 subjects, borderline, but run with >= 5 seeds")
    w(lines, "    Do NOT use 1% or 5% as primary results -- too few subjects")
    w(lines, "    Optionally report 1%/5% in supplementary as extreme edge case")

    return {
        "n_train_windows": len(train_subj),
        "n_train_subjects": len(uniq_train),
        "n_test_subjects": len(uniq_test),
        "total_hours": total_hours,
        "uniq_train": uniq_train.tolist(),
        "uniq_test":  uniq_test.tolist(),
    }


def analyze_domain_gap(lines):
    w(lines)
    w(lines, "=" * 70)
    w(lines, "CHALLENGE 2 -- Domain gap: UCI HAR -> SisFall transfer validity")
    w(lines, "=" * 70)
    rows = [
        ("Sampling rate",  "50 Hz",              "200 Hz (->100 Hz resamp)",    "HIGH -- 2x diff even after resamp"),
        ("Input channels", "6 (acc+gyro)",        "9 (2xacc+gyro)",              "MEDIUM -- sep. input projections needed"),
        ("Sensor hardware","Smartphone sensors",  "ADXL345/ITG3200/MMA8451Q",    "HIGH -- diff noise/range/units"),
        ("Body placement", "Waist (approx)",      "Belt/waist",                  "LOW -- both waist, acceptable"),
        ("Population",     "30 volunteers",       "38 (young+elderly)",          "MEDIUM -- elderly not in UCI"),
        ("Task",           "6 ADL categories",    "19 ADL + 15 fall types",      "HIGH -- no fall class in UCI"),
        ("Rec. length",    "2.56s (fixed)",       "~14-15s (full rec)",          "HIGH -- SisFall needs windowing"),
        ("Label type",     "Window-level given",  "Recording-level (no onset)",  "HIGH -- manual onset required"),
        ("Class balance",  "~balanced",           "ADL-heavy (after windowing)", "HIGH -- use weighted loss"),
        ("Signal units",   "Normalized g-units",  "Raw integer ADU",             "MEDIUM -- conversion needed"),
    ]
    w(lines)
    w(lines, f"  {'Dimension':<20}  {'UCI HAR':<24}  {'SisFall':<28}  {'Gap'}")
    w(lines, f"  {'-'*20}  {'-'*24}  {'-'*28}  {'-'*30}")
    for dim, uci, sf, gap in rows:
        w(lines, f"  {dim:<20}  {uci:<24}  {sf:<28}  {gap}")

    w(lines)
    w(lines, "  VERDICT:")
    w(lines, "  UCI -> SisFall IS a valid transfer experiment, but the gap is large.")
    w(lines, "  This is NOT a same-distribution generalization test.")
    w(lines, "  The correct framing is:")
    w(lines, "    'We fine-tune a UCI-pretrained encoder on SisFall with labels.'")
    w(lines, "    NOT: 'We generalize to SisFall' (too strong -- you're fine-tuning)")
    w(lines, "    NOT: 'Zero-shot transfer' (there is supervised fine-tuning)")
    w(lines, "  The encoder shares weights; separate LINEAR INPUT PROJECTIONS per dataset.")
    w(lines, "  The scientific question: does UCI pretraining give a better init for SisFall?")
    w(lines, "  This is meaningful and defensible even if the result is a marginal gain.")


def main():
    lines = []
    w(lines, "#" * 70)
    w(lines, "  PRE-IMPLEMENTATION VALIDATION AUDIT")
    w(lines, "  Answers 5 critical design challenges from REAL data")
    w(lines, "#" * 70)

    uci_stats = analyze_uci(lines)
    sf_stats  = analyze_sisfall_onset(lines, subject="SA01")
    analyze_domain_gap(lines)

    # SisFall scale estimate
    w(lines)
    w(lines, "=" * 70)
    w(lines, "BONUS -- SisFall dataset scale estimate")
    w(lines, "=" * 70)
    all_dirs   = [d for d in SISFALL_DIR.iterdir() if d.is_dir()]
    total_size = 0
    n_fall_files = 0
    n_adl_files  = 0
    for sd in all_dirs:
        for f in sd.glob("[DF]*.txt"):
            sz = f.stat().st_size
            total_size += sz
            if f.name.startswith("F"):
                n_fall_files += 1
            else:
                n_adl_files  += 1
    # ~19 chars/line (9 ints + commas + semicolon + newline), 9 int channels
    est_samples = total_size // 19
    est_hrs_200 = est_samples / SISFALL_FS / 3600
    est_hrs_100 = est_samples / RESAMPLE_FS / 3600
    w(lines, f"  Subjects        : {len(all_dirs)}")
    w(lines, f"  ADL files       : {n_adl_files}")
    w(lines, f"  Fall files      : {n_fall_files}")
    w(lines, f"  Est. samples    : {est_samples:,}")
    w(lines, f"  Est. duration   : {est_hrs_200:.1f} h @ 200 Hz  |  {est_hrs_100:.1f} h @ 100 Hz")
    # windowed estimate
    W_SZ, STRIDE = 200, 100   # 2s window @ 100 Hz, 50% overlap
    est_wins = est_samples // 2 // STRIDE   # rough
    w(lines, f"  Rough window est: ~{est_wins:,} windows (2s/50%overlap, before onset labeling)")
    w(lines, f"  Fall windows will be a small fraction -- exact count requires onset labeling")

    # ---- FINAL SUMMARY ----
    w(lines)
    w(lines, "#" * 70)
    w(lines, "  FINAL AUDIT SUMMARY -- REVISED DESIGN DECISIONS")
    w(lines, "#" * 70)
    w(lines, f"""
  CHALLENGE 1 -- UCI HAR pretraining scale
  ------------------------------------------
  {uci_stats['n_train_windows']} windows = {uci_stats['total_hours']:.3f} hours total.
  This is ~{40_000_000/uci_stats['total_hours']:,.0f}x smaller than LSM (40M hours).
  DECISION: Frame as 'self-supervised representation learning at public-dataset scale.'
  Do NOT claim 'foundation model' anywhere in the paper or viva.
  Accept small-corpus as a study variable: does SSL still help even when
  pretraining data is tiny? This is itself an interesting research question.

  CHALLENGE 2 -- UCI -> SisFall transfer validity
  --------------------------------------------------
  Gap is LARGE (sampling rate, sensors, task, population, annotation).
  Transfer is valid as fine-tuned transfer, NOT zero-shot.
  Use separate linear input-projection layers per dataset.
  Shared encoder processes D-dim embeddings -- temporal patterns can transfer.
  Frame: 'Does UCI-pretrained encoder provide a better init for SisFall fine-tuning?'

  CHALLENGE 3 -- SisFall fall-onset labeling reliability
  --------------------------------------------------------
  ADL recordings exceeding min fall peak : {sf_stats['adl_exceeding_minfall']} / {sf_stats['n_adl_recordings']}
  Fall recordings separable from all ADLs: {sf_stats['fall_separable']} / {sf_stats['n_fall_recordings']}
  Ambiguous fall files (dual peaks)      : {sf_stats['pct_ambiguous']:.1f}%
  DECISIONS:
  (a) Use peak-acceleration heuristic as STARTING POINT only.
  (b) Validate visually on 10 fall + 10 ADL files before finalizing.
  (c) Consider a 'transition zone' discard strategy around ambiguous peaks.
  (d) Document labeling method as a limitation in the paper/report.
  (e) Consider including both pre-fall + fall windows as 'fall class'
      following Musci et al. -- but only if enhanced labels are available.

  CHALLENGE 4 -- Label fraction feasibility
  ------------------------------------------
  21 train subjects in UCI HAR.
  1%  (1 subj) -- UNRELIABLE. Remove as primary result.
  5%  (1 subj) -- UNRELIABLE. Remove as primary result.
  10% (2 subj) -- BORDERLINE. Use only if >= 5 seeds; report std.
  25% (5 subj) -- MINIMUM RELIABLE. Use as lowest primary fraction.
  REVISED CORE FRACTIONS: {{25%, 50%, 100%}}
  SUPPLEMENTARY FRACTIONS: {{10%}} with >= 5 seeds, explicit caveat.

  CHALLENGE 5 -- UCI HAR structure
  ----------------------------------
  {uci_stats['n_train_subjects']} train subjects, {uci_stats['n_test_subjects']} test subjects.
  Official split is already SUBJECT-DISJOINT -- no leakage in default split.
  Carve validation set from train subjects (e.g. last 4 of 21 train subjects).
  Pretraining uses train-subject windows only (no labels, no test subjects).
""")

    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
