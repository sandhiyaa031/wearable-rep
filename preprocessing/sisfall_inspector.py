"""
MODULE: sisfall_inspector.py

WHAT:
    Visual inspection tool for SisFall raw recordings.
    Plots 10 fall and 10 ADL recordings, computes per-recording statistics,
    and determines whether peak-centered windowing is viable.

WHY (from the audit):
    The pre-implementation audit revealed that 49% of ADL recordings in SA01
    exceed the minimum peak magnitude of fall recordings, and 100% of all
    recordings have multiple prominent peaks. Before writing ANY labeling code,
    we must visually confirm whether:
      (a) Fall events are visually distinct and localizable
      (b) ADL peaks are truly comparable in magnitude to fall peaks
      (c) The peak location within fall recordings is predictable
      (d) A transition zone / discard zone is feasible

WHAT THIS SCRIPT PRODUCES:
    inspection_results/sisfall_inspection/
        fall_plots/   → one plot per inspected fall recording (4-panel)
        adl_plots/    → one plot per inspected ADL recording (4-panel)
        INSPECTION_REPORT.txt → statistics + labeling decision

DECISION POINT:
    After running this script, examine the plots and the report.
    Choose ONE of:
      OPTION A: Recording-level labeling (F*.txt → all windows = FALL)
                Safe but noisy — pre-fall ADL windows get "FALL" label
      OPTION B: Peak-centered labeling (window ± 1s around peak = FALL,
                baseline before peak = ADL, transition zone discarded)
                More precise but requires validated onset detection

MATH (peak-centered option):
    peak_idx = argmax(sqrt(ADXL_X^2 + ADXL_Y^2 + ADXL_Z^2))
    fall_start = peak_idx - FS_raw        (1 second before peak @ 200Hz)
    fall_end   = peak_idx + FS_raw        (1 second after  peak @ 200Hz)
    adl_end    = peak_idx - 3 * FS_raw   (baseline: >3 seconds before peak)
    transition = [adl_end, fall_start]   (discarded)

TENSOR SHAPES:
    raw file : (N_samples, 9)   integer raw ADU values
    plot data: (N_samples,)     resultant acceleration magnitude

VIVA ANSWER:
    "Before writing any preprocessing code I visually inspected 10 fall and
     10 ADL recordings to validate the labeling strategy. I confirmed whether
     fall events produce a single dominant acceleration peak that is reliably
     distinct from ADL motion, and used this to justify the recording-level
     vs peak-centered labeling decision documented in INSPECTION_REPORT.txt."
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SISFALL_DIR  = Path(r"D:\college\PROJECTS-SEM 5\DL\datasets\SisFall_dataset")
OUTPUT_DIR   = Path(r"D:\college\PROJECTS-SEM 5\DL\inspection_results\sisfall_inspection")
FS_RAW       = 200   # Hz (original SisFall rate)
SUBJECTS_TO_INSPECT = ["SA01", "SA02"]   # inspect multiple subjects for robustness

# Fall types to inspect (representative sample covering different fall directions)
FALL_TYPES_TO_INSPECT = ["F01", "F05", "F07", "F10", "F15"]   # 5 types from 2 subjects = 10
ADL_TYPES_TO_INSPECT  = ["D01", "D05", "D07", "D10", "D17"]   # 5 types from 2 subjects = 10


# ─── FILE I/O ─────────────────────────────────────────────────────────────────

def load_sisfall_file(path: Path) -> np.ndarray:
    """
    Reads one SisFall .txt file.
    Returns np.ndarray of shape (N_samples, 9), dtype int32.

    Format per line: v1,v2,...,v9;
    """
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


def resultant_adxl(data: np.ndarray) -> np.ndarray:
    """
    Computes ADXL345 resultant acceleration per sample.
    data: (N, 9) raw int
    Returns: (N,) float
    """
    return np.sqrt(data[:, 0]**2 + data[:, 1]**2 + data[:, 2]**2).astype(float)


# ─── PEAK ANALYSIS PER RECORDING ──────────────────────────────────────────────

def analyze_recording(data: np.ndarray, is_fall: bool) -> dict:
    """
    Computes inspection statistics for one recording.
    data: (N, 9)

    Returns dict with:
        n_samples       : int
        duration_sec    : float
        res_peak        : float  (max resultant ADXL magnitude)
        peak_idx        : int    (sample index of peak)
        peak_time_frac  : float  (where in recording peak occurs: 0=start, 1=end)
        fall_onset_idx  : int    (if is_fall: where peak-centered fall window starts)
        adl_baseline_end: int    (if is_fall: end of safe ADL baseline before transition)
        n_secondary_peaks: int   (peaks within 70% of max peak, min 0.5s gap)
        adxl_raw        : (N,) resultant magnitudes for plotting
    """
    res = resultant_adxl(data)
    peak_idx = int(np.argmax(res))
    peak_val = float(res[peak_idx])
    n = len(res)

    # Count secondary peaks (local maxima > 70% of max, at least 0.5s = 100 samples gap)
    min_gap = int(0.5 * FS_RAW)
    peaks = []
    for i in range(1, n - 1):
        if res[i] > res[i-1] and res[i] > res[i+1]:
            if res[i] > 0.70 * peak_val:
                if not peaks or (i - peaks[-1]) >= min_gap:
                    peaks.append(i)
                elif res[i] > res[peaks[-1]]:
                    peaks[-1] = i

    fall_onset_idx   = max(0, peak_idx - FS_RAW)      # 1s before peak
    adl_baseline_end = max(0, peak_idx - 3 * FS_RAW)  # 3s before peak

    return {
        "n_samples":         n,
        "duration_sec":      n / FS_RAW,
        "res_peak":          peak_val,
        "peak_idx":          peak_idx,
        "peak_time_frac":    peak_idx / max(n - 1, 1),
        "fall_onset_idx":    fall_onset_idx,
        "adl_baseline_end":  adl_baseline_end,
        "n_secondary_peaks": len(peaks) - 1 if peaks else 0,  # subtract dominant peak
        "adxl_raw":          res,
    }


# ─── PLOTTING ─────────────────────────────────────────────────────────────────

def plot_recording(fname: str, data: np.ndarray, stats: dict,
                   label_type: str, out_dir: Path):
    """
    4-panel plot per recording:
    Panel 1: ADXL345 raw channels (3 axes)
    Panel 2: Resultant ADXL345 magnitude with peak marker
    Panel 3: ITG3200 gyroscope raw channels
    Panel 4: MMA8451Q accelerometer raw channels

    Saves to out_dir/<fname>.png
    """
    n = data.shape[0]
    t = np.arange(n) / FS_RAW   # (N,) time in seconds

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    color = "tomato" if label_type == "Fall" else "steelblue"

    # Panel 1: ADXL345 raw
    for c, ch_name in enumerate(["ADXL_X", "ADXL_Y", "ADXL_Z"]):
        axes[0].plot(t, data[:, c], label=ch_name, linewidth=0.7, alpha=0.8)
    axes[0].set_title(f"ADXL345 accelerometer — {fname} [{label_type}]", color=color)
    axes[0].set_ylabel("raw ADU")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Resultant ADXL magnitude (key diagnostic panel)
    res = stats["adxl_raw"]
    peak_t = stats["peak_idx"] / FS_RAW
    axes[1].plot(t, res, color=color, linewidth=0.9, label="||ADXL||")
    axes[1].axvline(peak_t, color="black", linestyle="--", linewidth=1.5,
                    label=f"Peak @{peak_t:.2f}s ({stats['res_peak']:.0f})")
    if label_type == "Fall":
        # Mark the proposed fall window and ADL baseline
        fall_start_t = stats["fall_onset_idx"] / FS_RAW
        adl_end_t    = stats["adl_baseline_end"] / FS_RAW
        axes[1].axvspan(fall_start_t, min(peak_t + 1.0, t[-1]),
                        alpha=0.15, color="red",  label="Proposed fall window (±1s)")
        axes[1].axvspan(0, adl_end_t,
                        alpha=0.15, color="blue", label="ADL baseline (<3s before peak)")
        axes[1].axvspan(adl_end_t, fall_start_t,
                        alpha=0.10, color="gray", label="Transition (discard)")
    axes[1].set_title(f"Resultant ADXL magnitude | peak fraction: "
                      f"{stats['peak_time_frac']:.2f} | "
                      f"secondary peaks: {stats['n_secondary_peaks']}")
    axes[1].set_ylabel("||ADXL|| (raw ADU)")
    axes[1].legend(loc="upper right", fontsize=7)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Gyroscope
    for c, ch_name in enumerate(["ITG_X", "ITG_Y", "ITG_Z"]):
        axes[2].plot(t, data[:, c + 3], label=ch_name, linewidth=0.7, alpha=0.8)
    axes[2].set_title("ITG3200 gyroscope (angular velocity)")
    axes[2].set_ylabel("raw ADU")
    axes[2].legend(loc="upper right", fontsize=7)
    axes[2].grid(True, alpha=0.3)

    # Panel 4: MMA8451Q
    for c, ch_name in enumerate(["MMA_X", "MMA_Y", "MMA_Z"]):
        axes[3].plot(t, data[:, c + 6], label=ch_name, linewidth=0.7, alpha=0.8)
    axes[3].set_title("MMA8451Q accelerometer")
    axes[3].set_ylabel("raw ADU")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].legend(loc="upper right", fontsize=7)
    axes[3].grid(True, alpha=0.3)

    plt.suptitle(f"{fname}  |  Duration: {stats['duration_sec']:.1f}s  |  "
                 f"Peak: {stats['res_peak']:.0f} ADU @ {peak_t:.2f}s",
                 fontsize=10)
    plt.tight_layout()
    out_path = out_dir / f"{fname.replace('.txt','')}.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    fall_out = OUTPUT_DIR / "fall_plots"
    adl_out  = OUTPUT_DIR / "adl_plots"
    fall_out.mkdir(parents=True, exist_ok=True)
    adl_out.mkdir(parents=True, exist_ok=True)

    report_lines = []
    def r(msg=""): report_lines.append(msg)

    r("#" * 70)
    r("  SISFALL VISUAL INSPECTION REPORT")
    r("  Determines labeling strategy before any preprocessing code is written")
    r("#" * 70)

    fall_stats_all = []
    adl_stats_all  = []

    # ── Inspect fall recordings ────────────────────────────────────────────
    r()
    r("=" * 70)
    r("FALL RECORDINGS (10 files across 2 subjects x 5 fall types)")
    r("=" * 70)
    r(f"  {'File':<26} {'Dur(s)':>6} {'Peak(ADU)':>10} {'PeakFrac':>9} {'SecPeaks':>9}")
    r(f"  {'-'*26} {'-'*6} {'-'*10} {'-'*9} {'-'*9}")

    inspected = 0
    for subj in SUBJECTS_TO_INSPECT:
        for fall_type in FALL_TYPES_TO_INSPECT:
            fname = f"{fall_type}_{subj}_R01.txt"
            fpath = SISFALL_DIR / subj / fname
            if not fpath.exists():
                continue
            data  = load_sisfall_file(fpath)
            stats = analyze_recording(data, is_fall=True)
            out   = plot_recording(fname, data, stats, "Fall", fall_out)
            fall_stats_all.append(stats)
            r(f"  {fname:<26} {stats['duration_sec']:>6.1f} "
              f"{stats['res_peak']:>10.0f} {stats['peak_time_frac']:>9.3f} "
              f"{stats['n_secondary_peaks']:>9d}")
            inspected += 1
            if inspected >= 10:
                break
        if inspected >= 10:
            break

    # ── Inspect ADL recordings ─────────────────────────────────────────────
    r()
    r("=" * 70)
    r("ADL RECORDINGS (10 files across 2 subjects x 5 ADL types)")
    r("=" * 70)
    r(f"  {'File':<26} {'Dur(s)':>6} {'Peak(ADU)':>10} {'PeakFrac':>9} {'SecPeaks':>9}")
    r(f"  {'-'*26} {'-'*6} {'-'*10} {'-'*9} {'-'*9}")

    inspected = 0
    for subj in SUBJECTS_TO_INSPECT:
        for adl_type in ADL_TYPES_TO_INSPECT:
            fname = f"{adl_type}_{subj}_R01.txt"
            fpath = SISFALL_DIR / subj / fname
            if not fpath.exists():
                continue
            data  = load_sisfall_file(fpath)
            stats = analyze_recording(data, is_fall=False)
            out   = plot_recording(fname, data, stats, "ADL", adl_out)
            adl_stats_all.append(stats)
            r(f"  {fname:<26} {stats['duration_sec']:>6.1f} "
              f"{stats['res_peak']:>10.0f} {stats['peak_time_frac']:>9.3f} "
              f"{stats['n_secondary_peaks']:>9d}")
            inspected += 1
            if inspected >= 10:
                break
        if inspected >= 10:
            break

    # ── Comparative statistics ─────────────────────────────────────────────
    fall_peaks = np.array([s["res_peak"] for s in fall_stats_all])
    adl_peaks  = np.array([s["res_peak"] for s in adl_stats_all])
    fall_fracs = np.array([s["peak_time_frac"] for s in fall_stats_all])
    adl_fracs  = np.array([s["peak_time_frac"] for s in adl_stats_all])

    r()
    r("=" * 70)
    r("COMPARATIVE STATISTICS")
    r("=" * 70)
    r(f"  Fall peak magnitude : mean={fall_peaks.mean():.0f}  std={fall_peaks.std():.0f}  "
      f"min={fall_peaks.min():.0f}  max={fall_peaks.max():.0f}")
    r(f"  ADL  peak magnitude : mean={adl_peaks.mean():.0f}  std={adl_peaks.std():.0f}  "
      f"min={adl_peaks.min():.0f}  max={adl_peaks.max():.0f}")
    r()
    r(f"  Fall peak time fraction: mean={fall_fracs.mean():.3f}  std={fall_fracs.std():.3f}")
    r(f"  ADL  peak time fraction: mean={adl_fracs.mean():.3f}  std={adl_fracs.std():.3f}")
    r()
    overlap = (adl_peaks >= fall_peaks.min()).sum()
    r(f"  ADL recordings overlapping with min fall peak: {overlap}/{len(adl_peaks)}")
    r(f"  Fall recordings with peak in SECOND HALF of recording "
      f"(frac > 0.5): {(fall_fracs > 0.5).sum()}/{len(fall_fracs)}")
    r()
    r("  If fall peaks reliably occur in the second half of recording ->")
    r("  peak-centered windowing is viable. Check plots to confirm.")

    r()
    r("=" * 70)
    r("LABELING DECISION (fill in after reviewing plots)")
    r("=" * 70)
    r("""
  OPTION A (Recording-level, safest):
    All windows from F*.txt -> label = 1 (FALL)
    All windows from D*.txt -> label = 0 (ADL)
    PRO: No onset detection required. Zero risk of mislabeling.
    CON: Pre-fall ADL portions of fall recordings are labeled 1.
         Post-fall recovery also labeled 1.
    WHEN TO USE: If peaks are NOT reliably in the 2nd half of recording,
                 OR if visual inspection shows high peak overlap with ADL.

  OPTION B (Peak-centered, more precise):
    For each F*.txt:
      peak_idx = argmax(||ADXL||)
      fall_window = [peak_idx - FS_raw, peak_idx + FS_raw]  (2 seconds, centered)
      adl_baseline = [0, peak_idx - 3*FS_raw]  (baseline, if long enough)
      transition  = [peak_idx - 3*FS_raw, peak_idx - FS_raw]  (discard)
    For each D*.txt:
      all windows -> label = 0 (ADL)
    PRO: Clean, clinically meaningful fall labels.
    CON: Requires the onset detection to be reliable AND validated.
    WHEN TO USE: If fall peaks are visually clear, mostly in 2nd half,
                 AND visually distinct from ADL peak shapes.

  CHOSEN STRATEGY: [TO BE FILLED AFTER VIEWING PLOTS]
  JUSTIFICATION:   [TO BE FILLED AFTER VIEWING PLOTS]
  """)

    # ── Save report ───────────────────────────────────────────────────────
    report_path = OUTPUT_DIR / "INSPECTION_REPORT.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Report written to: {report_path}")
    print(f"Fall plots: {fall_out}  ({len(fall_stats_all)} files)")
    print(f"ADL plots : {adl_out}  ({len(adl_stats_all)} files)")
    print(f"\nREVIEW the plots, then fill in CHOSEN STRATEGY in INSPECTION_REPORT.txt")
    print(f"before proceeding to sisfall_loader.py implementation.")


if __name__ == "__main__":
    main()
