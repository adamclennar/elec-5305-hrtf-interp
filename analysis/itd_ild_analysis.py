
"""analysis/itd_ild_analysis.py
Computes ITD and ILD vs azimuth for one or more SOFA HRIR datasets.
Usage:
    python analysis/itd_ild_analysis.py --datasets data/CIPIC_subject003.sofa CIPIC data/HRIR_KEMAR.sofa KEMAR
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from src.sofa_loader import load_hrir_data, select_elevation

def compute_itd_ild(hrir: np.ndarray, fs: float):
    """Return (itd_ms, ild_db) for each measurement in hrir[M,2,N]."""
    M = hrir.shape[0]
    itd_ms = np.zeros(M, dtype=float)
    ild_db = np.zeros(M, dtype=float)
    for i in range(M):
        left, right = hrir[i, 0], hrir[i, 1]
        # ITD via cross-correlation
        corr = correlate(left, right, mode='full')
        lags = np.arange(-len(left) + 1, len(left))
        lag_s = lags[int(np.argmax(corr))] / fs
        itd_ms[i] = lag_s * 1000.0
        # ILD via RMS ratio
        rms_l = float(np.sqrt(np.mean(left**2) + 1e-12))
        rms_r = float(np.sqrt(np.mean(right**2) + 1e-12))
        ild_db[i] = 20.0 * np.log10(rms_r / rms_l + 1e-12)
    return itd_ms, ild_db

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs='+', required=True,
                    help="Pairs of <sofa_path> <label> ... e.g., data/CIPIC_subject003.sofa CIPIC data/HRIR_KEMAR.sofa KEMAR")
    ap.add_argument("--elev", type=float, default=0.0, help="Elevation to analyse (deg)")
    ap.add_argument("--tol", type=float, default=5.0, help="Elevation tolerance (deg)")
    args = ap.parse_args()

    if len(args.datasets) % 2 != 0:
        raise SystemExit("--datasets expects pairs: <path> <label> ...")

    pairs = [(args.datasets[i], args.datasets[i+1]) for i in range(0, len(args.datasets), 2)]

    # Prepare plots: one for ITD, one for ILD
    plt.figure(figsize=(8, 4))
    for path, label in pairs:
        hrir, src_pos, fs = load_hrir_data(path)
        h_sub, p_sub = select_elevation(hrir, src_pos, args.elev, tol=args.tol)
        if h_sub.shape[0] == 0:
            print(f"[WARN] {label}: no HRIRs near {args.elev}° (±{args.tol}°). Skipping.")
            continue
        az = p_sub[:, 0]
        itd_ms, ild_db = compute_itd_ild(h_sub, fs)
        order = np.argsort(az)
        az_sorted = az[order]
        itd_sorted = itd_ms[order]
        plt.plot(az_sorted, itd_sorted, marker='o', label=label)
    plt.xlabel("Azimuth (°)")
    plt.ylabel("ITD (ms)")
    plt.title("ITD vs Azimuth") 
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("/mnt/data/itd_vs_azimuth.png", dpi=150)

    plt.figure(figsize=(8, 4))
    for path, label in pairs:
        hrir, src_pos, fs = load_hrir_data(path)
        h_sub, p_sub = select_elevation(hrir, src_pos, args.elev, tol=args.tol)
        if h_sub.shape[0] == 0:
            continue
        az = p_sub[:, 0]
        itd_ms, ild_db = compute_itd_ild(h_sub, fs)
        order = np.argsort(az)
        az_sorted = az[order]
        ild_sorted = ild_db[order]
        plt.plot(az_sorted, ild_sorted, marker='s', label=label)
    plt.xlabel("Azimuth (°)")
    plt.ylabel("ILD (dB)")
    plt.title("ILD vs Azimuth")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("/mnt/data/ild_vs_azimuth.png", dpi=150)

    print("Saved plots:") 
    print(" - /mnt/data/itd_vs_azimuth.png") 
    print(" - /mnt/data/ild_vs_azimuth.png") 

if __name__ == "__main__":
    main()
