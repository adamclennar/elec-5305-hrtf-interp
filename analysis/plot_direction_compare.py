#!/usr/bin/env python3
"""
Visualise HRTF interpolation at a single direction for multiple methods.

For a given subject, sparsity, and target azimuth:
  - hide that azimuth from the "measured" set (if possible),
  - run RBF, SH, NEURAL_FIELD (residual NF), etc,
  - plot true vs predicted magnitude responses at that azimuth.

Usage example:

PYTHONPATH=. python analysis/plot_direction_compare.py \
  --npz data_npz/test/NH43__hrtf_M_hrtf_B__elev0.npz \
  --sparsity 30 \
  --az_deg 40 \
  --methods RBF SH NEURAL_FIELD \
  --outdir results/interp_viz

"""

import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from analysis.eval_methods import (
    wrap_deg,
    make_mask,
    predict_nn,
    predict_spline_periodic,
    predict_neural_field_residual_banded,
)
from src.sh_interp import predict_sh_banded_cv

def pick_hidden_index(az, keep_mask, target_deg):
    """
    Pick index of azimuth closest to target_deg that is HIDDEN (not kept).
    If the closest azimuth is kept, look for nearest hidden one.
    """
    target_deg = float(target_deg)
    # distance on circle
    d = np.minimum(
        np.abs(az - target_deg),
        360.0 - np.abs(az - target_deg)
    )
    order = np.argsort(d)

    # prefer hidden angles
    for idx in order:
        if not keep_mask[idx]:
            return int(idx)

    # fallback: if all kept (degenerate), just use closest
    return int(order[0])


def run_methods_at_direction(
    d,           # np.load dict
    npz_path,
    sparsity,
    az_target,
    methods,
    nf_dir="results/neural_field",
    band_lo=1000.0,
    band_hi=8000.0,
):
    az = d["az_deg"].astype(float)
    freqs = d["freqs_hz"].astype(float)
    L = d["magL_db"].astype(float)
    R = d["magR_db"].astype(float)
    ITD = d["itd_ms"].astype(float)

    az = wrap_deg(az)
    order = np.argsort(az)
    az, L, R, ITD = az[order], L[order], R[order], ITD[order]

    keep = make_mask(az, sparsity, offset_deg=0.0)

    # choose hidden index nearest to requested az
    idx = pick_hidden_index(az, keep, az_target)
    az_chosen = az[idx]
    is_hidden = not keep[idx]
    print(f"Chosen azimuth {az_chosen:.1f}° "
          f"({'HIDDEN' if is_hidden else 'KEPT'} at {sparsity}°).")

    subj_id = os.path.splitext(os.path.basename(npz_path))[0]

    out = {}  # method -> (predL[idx,:], predR[idx,:])

    for m in methods:
        mu = m.upper()
        if mu == "NN":
            predL, predR, _ = predict_nn(az, keep, L, R, ITD)

        elif mu in ("RBF", "SPLINE"):
            predL, predR, _ = predict_spline_periodic(az, keep, L, R, ITD)

        elif mu == "SH":
            predL, predR, _, _info = predict_sh_banded_cv(
                az, keep, L, R, ITD, freqs,
                n_bands=48,
                L_grid=tuple(range(1,13)),
                lam_grid=(1e-1, 3e-2, 1e-2, 3e-3, 1e-3),
                band_lo=band_lo, band_hi=band_hi
            )

        elif mu in ("NEURAL_FIELD", "NF"):
            predL, predR, _ = predict_neural_field_residual_banded(
                az, keep, L, R, ITD, freqs,
                step_deg=sparsity,
                subject_id=subj_id,
                nf_dir=nf_dir,
                band_lo=band_lo,
                band_hi=band_hi,
                n_bands=48,
                width=128,
                depth=4,
                epochs=400,
                batch_size=1024,
                lr=3e-4,
                smooth_lambda=1e-4,
                smooth_output=False,
            )

        elif mu in ("NEURAL_FIELD_SMOOTH", "NF_SMOOTH"):
            predL, predR, _ = predict_neural_field_residual_banded(
                az, keep, L, R, ITD, freqs,
                step_deg=sparsity,
                subject_id=subj_id,
                nf_dir=nf_dir,
                band_lo=band_lo,
                band_hi=band_hi,
                n_bands=48,
                width=128,
                depth=4,
                epochs=400,
                batch_size=1024,
                lr=3e-4,
                smooth_lambda=1e-4,
                smooth_output=True,      # key line
                sg_window=9,
                sg_poly=3,
            )


        else:
            print(f"[WARN] Unknown method {m}, skipping.")
            continue

        out[mu] = (predL[idx, :].copy(), predR[idx, :].copy())

    # true spectra at that azimuth
    trueL = L[idx, :]
    trueR = R[idx, :]

    return freqs, az_chosen, is_hidden, trueL, trueR, out


def plot_direction(
    freqs,
    az_chosen,
    is_hidden,
    trueL,
    trueR,
    preds,
    methods,
    out_png,
    band_lo=1000.0,
    band_hi=8000.0,
):
    plt.figure(figsize=(9, 6))

    # --- Left ear ---
    ax1 = plt.subplot(2, 1, 1)
    ax1.semilogx(freqs, trueL, label="True L", linewidth=2)
    for m in methods:
        mu = m.upper()
        if mu in preds:
            ax1.semilogx(freqs, preds[mu][0], label=f"{mu} L", alpha=0.8)
    ax1.set_xlim(freqs[0], freqs[-1])
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title(
        f"Subject az={az_chosen:.1f}° "
        f"({'hidden' if is_hidden else 'kept'}) — Left ear"
    )
    ax1.axvspan(band_lo, band_hi, color="grey", alpha=0.08)
    ax1.legend(loc="best", fontsize=8)

    # --- Right ear ---
    ax2 = plt.subplot(2, 1, 2)
    ax2.semilogx(freqs, trueR, label="True R", linewidth=2)
    for m in methods:
        mu = m.upper()
        if mu in preds:
            ax2.semilogx(freqs, preds[mu][1], label=f"{mu} R", alpha=0.8)
    ax2.set_xlim(freqs[0], freqs[-1])
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_title(
        f"Subject az={az_chosen:.1f}° "
        f"({'hidden' if is_hidden else 'kept'}) — Right ear"
    )
    ax2.axvspan(band_lo, band_hi, color="grey", alpha=0.08)
    ax2.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[saved] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to subject .npz file")
    ap.add_argument("--sparsity", type=int, required=True,
                    help="Keep every N degrees (e.g., 30 or 15)")
    ap.add_argument("--az_deg", type=float, required=True,
                    help="Target azimuth in degrees (will snap to nearest)")
    ap.add_argument("--methods", nargs="+",
                    default=["RBF", "SH", "NEURAL_FIELD"],
                    help="Which methods to plot")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--nf_dir", default="results/neural_field")
    ap.add_argument("--band_lo", type=float, default=1000.0)
    ap.add_argument("--band_hi", type=float, default=8000.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    d = np.load(args.npz)
    freqs, az_chosen, is_hidden, trueL, trueR, preds = run_methods_at_direction(
        d,
        npz_path=args.npz,
        sparsity=args.sparsity,
        az_target=args.az_deg,
        methods=args.methods,
        nf_dir=args.nf_dir,
        band_lo=args.band_lo,
        band_hi=args.band_hi,
    )

    base = os.path.splitext(os.path.basename(args.npz))[0]
    out_png = os.path.join(
        args.outdir,
        f"{base}__az{int(round(az_chosen))}deg__s{args.sparsity}.png"
    )

    plot_direction(
        freqs, az_chosen, is_hidden,
        trueL, trueR,
        preds,
        methods=args.methods,
        out_png=out_png,
        band_lo=args.band_lo,
        band_hi=args.band_hi,
    )


if __name__ == "__main__":
    main()
