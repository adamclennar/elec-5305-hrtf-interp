"""
analysis/itd_ild_analysis.py

Compare HRIR datasets by ITD/ILD vs azimuth.
All non-reference datasets are aligned to the first dataset's azimuth grid,
then ΔILD/ΔITD are plotted with MAE/RMSE/MaxAbs summaries.

Example:
PYTHONPATH=. python analysis/itd_ild_analysis.py \
  --datasets "data/hrtf_M_normal pinna.sofa" KEMAR_5deg \
             "data/hrtf_M_normal pinna resolution 0.5 deg.sofa" KEMAR_0p5deg \
  --elev 0 --tol 5 --outdir results --show
"""
from __future__ import annotations
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from src.sofa_loader import load_hrir_data, select_elevation


# ---------------------- Utilities ----------------------

def normalize_az_deg(a: np.ndarray) -> np.ndarray:
    """Map [0..360) to [-180..180) for nicer sorting/overlay."""
    return (a + 180.0) % 360.0 - 180.0

def align_by_nearest(az_ref: np.ndarray, az_other: np.ndarray, vals_other: np.ndarray, tol_deg: float = 0.6):
    """
    Align vals_other (sampled at az_other) onto az_ref by nearest neighbour.
    Returns:
        aligned_vals: np.ndarray (len(az_ref)) with NaN where no match within tol
        ok_mask: boolean mask of matches
    """
    aligned = np.full(len(az_ref), np.nan, dtype=float)
    ok = np.zeros(len(az_ref), dtype=bool)
    for i, ar in enumerate(az_ref):
        j = int(np.argmin(np.abs(az_other - ar)))
        if abs(az_other[j] - ar) <= tol_deg:
            aligned[i] = vals_other[j]
            ok[i] = True
    return aligned, ok

def compute_itd_ild(hrir: np.ndarray, fs: float):
    """
    Compute ITD (ms) and ILD (dB) for HRIR[M,2,N].
    ITD via cross-correlation peak lag; ILD via RMS ratio (R/L).
    """
    M = hrir.shape[0]
    itd_ms = np.zeros(M, dtype=float)
    ild_db = np.zeros(M, dtype=float)
    for i in range(M):
        left, right = hrir[i, 0], hrir[i, 1]

        # ITD
        corr = correlate(left, right, mode='full')
        lags = np.arange(-len(left) + 1, len(left))
        lag_s = lags[int(np.argmax(corr))] / fs
        itd_ms[i] = lag_s * 1000.0

        # ILD
        rms_l = float(np.sqrt(np.mean(left**2) + 1e-12))
        rms_r = float(np.sqrt(np.mean(right**2) + 1e-12))
        ild_db[i] = 20.0 * np.log10((rms_r + 1e-12) / (rms_l + 1e-12))
    return itd_ms, ild_db


# ---------------------- Main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs='+', required=True,
                    help='Pairs: <sofa_path> <label> ... e.g., data/A.sofa A data/B.sofa B')
    ap.add_argument("--elev", type=float, default=0.0, help="Elevation to analyse (deg)")
    ap.add_argument("--tol", type=float, default=5.0, help="Elevation tolerance (deg)")
    ap.add_argument("--align_tol", type=float, default=0.6, help="Azimuth matching tolerance for alignment (deg)")
    ap.add_argument("--outdir", type=str, default="results", help="Directory to save plots")
    ap.add_argument("--show", action="store_true", help="Also display the figures")
    args = ap.parse_args()

    if len(args.datasets) % 2 != 0:
        raise SystemExit("--datasets expects pairs: <path> <label> ...")

    pairs = [(args.datasets[i], args.datasets[i+1]) for i in range(0, len(args.datasets), 2)]
    os.makedirs(args.outdir, exist_ok=True)

    # Load all datasets and compute ITD/ILD at target elevation
    datasets_data = []  # list of dicts: label, az, itd, ild
    for path, label in pairs:
        hrir, src_pos, fs = load_hrir_data(path)
        h_sub, p_sub = select_elevation(hrir, src_pos, args.elev, tol=args.tol)
        if h_sub.shape[0] == 0:
            print(f"[WARN] {label}: no HRIRs near {args.elev}° (±{args.tol}°). Skipping.")
            continue
        az = p_sub[:, 0]
        itd_ms, ild_db = compute_itd_ild(h_sub, fs)
        order = np.argsort(az)
        datasets_data.append({
            "label": label,
            "az": normalize_az_deg(az[order]),
            "itd": itd_ms[order],
            "ild": ild_db[order],
        })

    if len(datasets_data) < 1:
        print("No datasets available after filtering — nothing to plot.")
        return
    if len(datasets_data) == 1:
        print("Only one dataset loaded; plotting its raw ITD/ILD vs azimuth.")
        d = datasets_data[0]
        az = np.sort(d["az"])
        idx = np.argsort(d["az"])
        # Simple one-dataset plots
        for y, name, ylabel, fname in [
            (d["itd"][idx], "ITD vs Azimuth", "ITD (ms)", "itd_vs_azimuth.png"),
            (d["ild"][idx], "ILD vs Azimuth", "ILD (dB)", "ild_vs_azimuth.png"),
        ]:
            plt.figure(figsize=(9, 4))
            plt.plot(az, y, marker='o', label=d["label"])
            plt.xlabel("Azimuth (°)"); plt.ylabel(ylabel); plt.title(name); plt.grid(True); plt.legend()
            plt.tight_layout(); plt.savefig(os.path.join(args.outdir, fname), dpi=150)
        if args.show: plt.show()
        return

    # Use the FIRST dataset as reference grid
    ref = datasets_data[0]
    order_ref = np.argsort(ref["az"])
    az_ref = ref["az"][order_ref]
    itd_ref = ref["itd"][order_ref]
    ild_ref = ref["ild"][order_ref]

    # -------- ILD aligned plot --------
    plt.figure(figsize=(10, 4.2))
    plt.plot(az_ref, ild_ref, marker='o', lw=1.5, label=ref["label"])
    for d in datasets_data[1:]:
        ild_aligned, ok = align_by_nearest(az_ref, d["az"], d["ild"], tol_deg=args.align_tol)
        plt.plot(az_ref[ok], ild_aligned[ok], marker='s', lw=1.0, label=d["label"])
    plt.xlabel("Azimuth (°)"); plt.ylabel("ILD (dB)")
    plt.title("ILD vs Azimuth (aligned to reference grid)")
    plt.grid(True); plt.legend()
    ild_aligned_path = os.path.join(args.outdir, "ild_vs_azimuth_aligned.png")
    plt.tight_layout(); plt.savefig(ild_aligned_path, dpi=150)

    # ΔILD per dataset
    for d in datasets_data[1:]:
        ild_aligned, ok = align_by_nearest(az_ref, d["az"], d["ild"], tol_deg=args.align_tol)
        delta = ild_aligned[ok] - ild_ref[ok]
        mae = float(np.nanmean(np.abs(delta)))
        rmse = float(np.sqrt(np.nanmean(delta**2)))
        mx = float(np.nanmax(np.abs(delta)))
        plt.figure(figsize=(10, 3.8))
        plt.plot(az_ref[ok], delta, marker='.', lw=1.0)
        plt.axhline(0, ls='--', color='k')
        plt.xlabel("Azimuth (°)"); plt.ylabel("ΔILD (dB)")
        plt.title(f"ΔILD: {d['label']} − {ref['label']}  |  MAE={mae:.2f} dB, RMSE={rmse:.2f} dB, MaxAbs={mx:.2f} dB")
        plt.grid(True)
        pth = os.path.join(args.outdir, f"delta_ild_{d['label']}_minus_{ref['label']}.png")
        plt.tight_layout(); plt.savefig(pth, dpi=150)

    # -------- ITD aligned plot --------
    plt.figure(figsize=(10, 4.2))
    plt.plot(az_ref, itd_ref, marker='o', lw=1.5, label=ref["label"])
    for d in datasets_data[1:]:
        itd_aligned, ok = align_by_nearest(az_ref, d["az"], d["itd"], tol_deg=args.align_tol)
        plt.plot(az_ref[ok], itd_aligned[ok], marker='s', lw=1.0, label=d["label"])
    plt.xlabel("Azimuth (°)"); plt.ylabel("ITD (ms)")
    plt.title("ITD vs Azimuth (aligned to reference grid)")
    plt.grid(True); plt.legend()
    itd_aligned_path = os.path.join(args.outdir, "itd_vs_azimuth_aligned.png")
    plt.tight_layout(); plt.savefig(itd_aligned_path, dpi=150)

    # ΔITD per dataset
    for d in datasets_data[1:]:
        itd_aligned, ok = align_by_nearest(az_ref, d["az"], d["itd"], tol_deg=args.align_tol)
        delta = itd_aligned[ok] - itd_ref[ok]
        mae = float(np.nanmean(np.abs(delta)))
        rmse = float(np.sqrt(np.nanmean(delta**2)))
        mx = float(np.nanmax(np.abs(delta)))
        plt.figure(figsize=(10, 3.8))
        plt.plot(az_ref[ok], delta, marker='.', lw=1.0)
        plt.axhline(0, ls='--', color='k')
        plt.xlabel("Azimuth (°)"); plt.ylabel("ΔITD (ms)")
        plt.title(f"ΔITD: {d['label']} − {ref['label']}  |  MAE={mae:.3f} ms, RMSE={rmse:.3f} ms, MaxAbs={mx:.3f} ms")
        plt.grid(True)
        pth = os.path.join(args.outdir, f"delta_itd_{d['label']}_minus_{ref['label']}.png")
        plt.tight_layout(); plt.savefig(pth, dpi=150)

    print("Saved aligned & delta plots to:", args.outdir)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
