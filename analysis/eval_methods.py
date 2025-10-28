#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from scipy.interpolate import CubicSpline

# ---------------------------
# helpers
# ---------------------------
def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0

def make_mask(az_deg: np.ndarray, step_deg: float, offset_deg: float = 0.0) -> np.ndarray:
    """Keep angles every `step_deg` starting from offset."""
    a = wrap_deg(az_deg - offset_deg)
    # snap to grid; tolerate small numeric jitter
    keep = (np.abs((a / step_deg) - np.round(a / step_deg)) < 1e-6)
    return keep

def band_idx(freqs_hz: np.ndarray, lo=1000.0, hi=8000.0):
    return (freqs_hz >= lo) & (freqs_hz <= hi)

def lsd_db(pred_db: np.ndarray, true_db: np.ndarray, band: np.ndarray) -> float:
    """Log-spectral distance: RMSE in dB over freq band (averaged over angles)."""
    # pred/true: (A,F)
    D = (pred_db[:, band] - true_db[:, band])
    rmse = np.sqrt(np.mean(D**2))
    return float(rmse)

def ild_db(magR_db: np.ndarray, magL_db: np.ndarray) -> np.ndarray:
    """ILD per angle per frequency (R-L in dB)."""
    return (magR_db - magL_db)

# ---------------------------
# baselines
# ---------------------------
def predict_nn(az, keep_mask, magL_db, magR_db, itd_ms):
    """Nearest-neighbour along azimuth (circular)."""
    A = len(az)
    predL = magL_db.copy()
    predR = magR_db.copy()
    preditd = itd_ms.copy()

    kept_angles = az[keep_mask]
    if kept_angles.size == 0:
        raise RuntimeError("Empty keep set for NN.")

    for i in range(A):
        if keep_mask[i]:
            continue
        # circular nearest
        d = np.minimum(np.abs(az[i] - kept_angles), 360.0 - np.abs(az[i] - kept_angles))
        j = np.where(keep_mask)[0][np.argmin(d)]
        predL[i] = magL_db[j]
        predR[i] = magR_db[j]
        preditd[i] = itd_ms[j]
    return predL, predR, preditd


def _uniq_mean(x, Y):
    """Deduplicate x (1D) by averaging rows in Y for identical x values."""
    x = np.asarray(x)
    order = np.argsort(x)
    x = x[order]
    Y = Y[order]
    uniq_x, idx_start, counts = np.unique(x, return_index=True, return_counts=True)
    # average blocks
    outY = []
    for s, c in zip(idx_start, counts):
        outY.append(Y[s:s+c].mean(axis=0))
    return uniq_x, np.vstack(outY)

def predict_spline_periodic(az, keep_mask, magL_db, magR_db, itd_ms):
    """
    Periodic cubic spline across azimuth for each frequency bin (ear-wise) and for ITD (1D).
    Robust to duplicate kept angles; enforces periodicity by appending (x0+360, y0).
    """
    A, F = magL_db.shape
    predL = magL_db.copy()
    predR = magR_db.copy()

    # Sort once
    x_all = az.astype(float)
    order_all = np.argsort(x_all)
    x_all = x_all[order_all]
    L_all = magL_db[order_all]
    R_all = magR_db[order_all]
    T_all = itd_ms[order_all]
    kept = keep_mask[order_all]

    # Kept (training) angles
    xk = x_all[kept]
    if xk.size < 4:
        raise RuntimeError("Too few kept angles for periodic spline; increase density or use NN.")

    # Deduplicate kept angles by averaging magnitudes/ITD at identical az
    Lk = L_all[kept]   # (K,F)
    Rk = R_all[kept]
    Tk = T_all[kept]   # (K,)

    xk_u, Lk_u = _uniq_mean(xk, Lk)
    _,     Rk_u = _uniq_mean(xk, Rk)
    _,     Tk_u = _uniq_mean(xk, Tk.reshape(-1,1)); Tk_u = Tk_u[:,0]

    # Enforce periodicity by appending (x0+360, y0)
    x0 = xk_u[0]
    xk_per = np.concatenate([xk_u, xk_u[:1] + 360.0])  # strictly increasing
    # build query angles in same [x0, x0+360) range
    xq = x_all.copy()
    xq[xq < x0] += 360.0

    # Magnitudes per frequency
    for f in range(F):
        yL = np.concatenate([Lk_u[:, f], Lk_u[:1, f]])
        yR = np.concatenate([Rk_u[:, f], Rk_u[:1, f]])
        csL = CubicSpline(xk_per, yL, bc_type='periodic')
        csR = CubicSpline(xk_per, yR, bc_type='periodic')
        predL[:, f] = csL(xq)
        predR[:, f] = csR(xq)

    # ITD 1D spline
    yT = np.concatenate([Tk_u, Tk_u[:1]])
    csT = CubicSpline(xk_per, yT, bc_type='periodic')
    preditd = csT(xq)

    # Map predictions back to original az order
    inv = np.empty_like(order_all)
    inv[order_all] = np.arange(A)
    return predL[inv], predR[inv], preditd[inv]
# ---------------------------
# evaluation on one subject
# ---------------------------
def evaluate_subject(npz_path: str, step_deg: int, method: str, outdir: str,
                     band_lo=1000.0, band_hi=8000.0) -> Dict[str, float]:
    os.makedirs(outdir, exist_ok=True)
    d = np.load(npz_path)
    az = d["az_deg"].astype(float)
    freqs = d["freqs_hz"].astype(float)
    L = d["magL_db"].astype(float)
    R = d["magR_db"].astype(float)
    ITD = d["itd_ms"].astype(float)

    # sort by azimuth in [-180,180)
    az = wrap_deg(az)
    order = np.argsort(az)
    az, L, R, ITD = az[order], L[order], R[order], ITD[order]

    keep = make_mask(az, step_deg, offset_deg=0.0)
    hide = ~keep

    if method.upper() == "NN":
        predL, predR, preditd = predict_nn(az, keep, L, R, ITD)
    elif method.upper() in ("RBF", "SPLINE"):
        predL, predR, preditd = predict_spline_periodic(az, keep, L, R, ITD)
    else:
        raise ValueError(f"Unknown method {method}. Use NN or RBF.")

    # metrics only on hidden angles
    b = band_idx(freqs, band_lo, band_hi)
    lsdL = lsd_db(predL[hide], L[hide], b)
    lsdR = lsd_db(predR[hide], R[hide], b)

    ild_true = ild_db(R, L)[:, b].mean(axis=1)        # (A,)
    ild_pred = ild_db(predR, predL)[:, b].mean(axis=1)
    ild_mae = float(np.mean(np.abs(ild_pred[hide] - ild_true[hide])))

    itd_mae = float(np.mean(np.abs(preditd[hide] - ITD[hide])))

    # save a quick figure (LSD bars)
    fig, ax = plt.subplots(figsize=(4.2,3))
    ax.bar([0,1], [lsdL, lsdR], width=0.6)
    ax.set_xticks([0,1], ["LSD_L", "LSD_R"])
    ax.set_ylabel("dB")
    ax.set_title(f"{os.path.basename(npz_path)} | {method} | {step_deg}° keep")
    fig.tight_layout()
    png = os.path.join(outdir, f"{os.path.splitext(os.path.basename(npz_path))[0]}__{method}__{step_deg}deg_metrics.png")
    fig.savefig(png, dpi=150); plt.close(fig)

    # write a small CSV row (append)
    row = {
        "subject": os.path.splitext(os.path.basename(npz_path))[0],
        "method": method.upper(),
        "sparsity_deg": step_deg,
        "lsd_L_db": lsdL,
        "lsd_R_db": lsdR,
        "itd_MAE_ms": itd_mae,
        "ild_MAE_db": ild_mae,
        "png": png,
    }
    return row

# ---------------------------
# main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", help="Path to one .npz subject file")
    ap.add_argument("--test_glob", help="Glob for many .npz (alternative to --subject)")
    ap.add_argument("--sparsity", nargs="+", type=int, required=True, help="Keep every N degrees (e.g., 30 15 10)")
    ap.add_argument("--methods", nargs="+", required=True, help="NN and/or RBF")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--band_lo", type=float, default=1000.0)
    ap.add_argument("--band_hi", type=float, default=8000.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # collect subjects
    subjects = []
    if args.subject:
        subjects = [args.subject]
    elif args.test_glob:
        import glob
        subjects = sorted(glob.glob(args.test_glob))
    else:
        raise SystemExit("Provide --subject or --test_glob")

    # eval
    rows = []
    for npz_path in subjects:
        for step in args.sparsity:
            for m in args.methods:
                try:
                    row = evaluate_subject(npz_path, step, m, args.outdir,
                                           band_lo=args.band_lo, band_hi=args.band_hi)
                    rows.append(row)
                    print(f"[OK] {os.path.basename(npz_path)} | {m} | {step}° -> LSD_L={row['lsd_L_db']:.3f} dB, LSD_R={row['lsd_R_db']:.3f} dB, ITD_MAE={row['itd_MAE_ms']:.3f} ms, ILD_MAE={row['ild_MAE_db']:.3f} dB")
                except Exception as e:
                    print(f"[WARN] skip {os.path.basename(npz_path)} | {m} | {step}°: {e}")

    # write CSV
    if rows:
        import csv
        csv_path = os.path.join(args.outdir, "interp_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("Saved metrics:", csv_path)
    else:
        print("No results produced.")

if __name__ == "__main__":
    main()
