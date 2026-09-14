"""
SisFall raw dataset inspection script.

Purpose ONLY: understand and visually inspect a small number of RAW
recordings. No preprocessing, no normalization, no filtering, no
windowing, no resampling, no model training happens here.

Reads files ONE AT A TIME (not the whole dataset into memory).
Does not modify, rename, or move any original dataset files.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # save PNGs directly, no GUI window needed
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------------------

DATASET_DIR = Path(r"D:\college\PROJECTS-SEM 5\DL\datasets\SisFall_dataset")
OUTPUT_DIR  = Path(r"D:\college\PROJECTS-SEM 5\DL\inspection_results")

FS = 200  # sampling rate claimed by the SisFall README (Hz)

# The 9 raw channels, in the exact column order given in the SisFall README
CHANNEL_NAMES = [
    "ADXL345_X", "ADXL345_Y", "ADXL345_Z",    # columns 1-3
    "ITG3200_X", "ITG3200_Y", "ITG3200_Z",    # columns 4-6
    "MMA8451Q_X", "MMA8451Q_Y", "MMA8451Q_Z"  # columns 7-9
]

# The specific files we want to inspect.
# Subject folder is derived automatically from the filename (SA01, etc.),
# so you only need to list the filenames here.
FILES_TO_INSPECT = [
    "D01_SA01_R01.txt",   # normal ADL (e.g. walking)
    "D18_SA01_R01.txt",   # fall-like / sudden ADL (e.g. sitting down quickly)
    "F01_SA01_R01.txt",   # fall type 1
    "F05_SA01_R01.txt",   # fall type 5 (a different fall pattern)
]


# ----------------------------------------------------------------------
# 2. FILENAME PARSING
# ----------------------------------------------------------------------

def parse_filename(filename):
    """
    Parses a SisFall filename like 'D01_SA01_R01.txt' or 'F05_SE02_R03.txt'
    into its meaning, using plain string splitting (no regex needed).

    Returns a dict:
        code          -> 'D01' or 'F05'          (raw activity/fall code)
        activity_type -> 'ADL' or 'Fall'
        activity_num  -> 1, 5, 18, ...            (integer)
        subject_id    -> 'SA01' (SA = young adult, SE = elderly)
        trial         -> 'R01'
    """
    stem = filename.replace(".txt", "")          # 'D01_SA01_R01'
    parts = stem.split("_")                      # ['D01', 'SA01', 'R01']
    code, subject_id, trial = parts[0], parts[1], parts[2]

    activity_type = "ADL" if code.startswith("D") else "Fall"
    activity_num = int(code[1:])                 # 'D01' -> 1

    return {
        "code": code,
        "activity_type": activity_type,
        "activity_num": activity_num,
        "subject_id": subject_id,
        "trial": trial,
    }


# ----------------------------------------------------------------------
# 3. RAW FILE LOADER (reads ONE file at a time -> low memory use)
# ----------------------------------------------------------------------

def load_raw_file(filepath):
    """
    Reads a single SisFall raw .txt file and returns a NumPy array
    of shape (n_rows, 9): one row per sample, 9 raw integer channels.

    File format (per line):
        val1,val2,val3,val4,val5,val6,val7,val8,val9;
    i.e. 9 comma-separated integers ending with a semicolon.
    Blank lines (sometimes present at the end of a file) are skipped.
    """
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.rstrip(";")
            values = [v for v in line.split(",") if v != ""]
            if len(values) != 9:
                continue  # skip any malformed line rather than crash
            rows.append([int(v) for v in values])
    return np.array(rows, dtype=np.int32)


# ----------------------------------------------------------------------
# 4. STATISTICS
# ----------------------------------------------------------------------

def compute_channel_stats(data):
    """
    data: NumPy array of shape (n_rows, 9)
    Returns per-channel min, max, mean, std as four arrays of length 9,
    computed along axis=0 (down each column / channel).
    """
    ch_min = data.min(axis=0)
    ch_max = data.max(axis=0)
    ch_mean = data.mean(axis=0)
    ch_std = data.std(axis=0)
    return ch_min, ch_max, ch_mean, ch_std


# ----------------------------------------------------------------------
# 5. REPORT PRINTING
# ----------------------------------------------------------------------

def print_file_report(filename, meta, data):
    """
    Prints filename, subject, activity code, shape, duration,
    and per-channel min/max/mean/std for one file.
    Returns a small dict used later for the summary table.
    """
    n_rows, n_channels = data.shape
    duration_sec = n_rows / FS
    ch_min, ch_max, ch_mean, ch_std = compute_channel_stats(data)

    print("=" * 70)
    print(f"File           : {filename}")
    print(f"Subject ID     : {meta['subject_id']}")
    print(f"Activity code  : {meta['code']} ({meta['activity_type']} #{meta['activity_num']})")
    print(f"Rows (samples) : {n_rows}")
    print(f"Channels       : {n_channels}")
    print(f"Assumed Fs     : {FS} Hz")
    print(f"Duration       : {duration_sec:.2f} s  (= n_rows / Fs)")
    print(f"{'Channel':<12}{'Min':>10}{'Max':>10}{'Mean':>12}{'Std':>12}")
    for i, name in enumerate(CHANNEL_NAMES):
        print(f"{name:<12}{ch_min[i]:>10}{ch_max[i]:>10}{ch_mean[i]:>12.2f}{ch_std[i]:>12.2f}")
    print("=" * 70 + "\n")

    return {
        "filename": filename,
        "subject_id": meta["subject_id"],
        "code": meta["code"],
        "activity_type": meta["activity_type"],
        "n_rows": n_rows,
        "duration_sec": duration_sec,
    }


# ----------------------------------------------------------------------
# 6. PLOTTING (raw values, no unit conversion, no filtering)
# ----------------------------------------------------------------------

def plot_file(filename, meta, data, output_dir):
    """
    Plots the 9 raw channels against time, grouped by sensor, as 3
    stacked subplots in one figure:
        subplot 1 -> ADXL345 accelerometer (X, Y, Z)
        subplot 2 -> ITG3200 gyroscope (X, Y, Z)
        subplot 3 -> MMA8451Q accelerometer (X, Y, Z)
    The full recording is plotted (no windowing), so for fall files
    the entire event is visible. Saves the figure as a PNG.
    """
    n_rows = data.shape[0]
    time = np.arange(n_rows) / FS   # seconds, shape (n_rows,)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    sensor_groups = [
        ("ADXL345 accelerometer (raw counts)",  [0, 1, 2], axes[0]),
        ("ITG3200 gyroscope (raw counts)",      [3, 4, 5], axes[1]),
        ("MMA8451Q accelerometer (raw counts)", [6, 7, 8], axes[2]),
    ]

    for title, cols, ax in sensor_groups:
        for c in cols:
            ax.plot(time, data[:, c], label=CHANNEL_NAMES[c], linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("raw sensor value")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{filename}  |  subject={meta['subject_id']}  "
                 f"code={meta['code']} ({meta['activity_type']})")
    fig.tight_layout()

    out_path = output_dir / f"{filename.replace('.txt', '')}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {out_path}")


# ----------------------------------------------------------------------
# 7. MAIN
# ----------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for filename in FILES_TO_INSPECT:
        meta = parse_filename(filename)
        filepath = DATASET_DIR / meta["subject_id"] / filename

        if not filepath.exists():
            print(f"[WARNING] File not found, skipping: {filepath}")
            continue

        data = load_raw_file(filepath)          # shape (n_rows, 9)

        row = print_file_report(filename, meta, data)
        plot_file(filename, meta, data, OUTPUT_DIR)
        summary_rows.append(row)

    # ---- final summary table ----
    print("\nSUMMARY TABLE")
    print(f"{'Filename':<20}{'Subject':<10}{'Code':<8}{'Type':<8}{'Rows':>8}{'Duration(s)':>14}")
    for r in summary_rows:
        print(f"{r['filename']:<20}{r['subject_id']:<10}{r['code']:<8}"
              f"{r['activity_type']:<8}{r['n_rows']:>8}{r['duration_sec']:>14.2f}")


if __name__ == "__main__":
    main()