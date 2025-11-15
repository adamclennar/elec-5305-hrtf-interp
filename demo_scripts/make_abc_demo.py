#!/usr/bin/env python3
"""
make_abc_demo.py

A/B/C listening demo:
    A = True measured HRTF
    B = RBF interpolation
    C = NF-smoothed interpolation

Usage (example), from repo root:
    PYTHONPATH=. python demo_scripts/make_abc_demo.py \
        --npz data_npz/test/NH43__hrtf_M_hrtf_B__elev0.npz \
        --audio_in demo_sources/speech_mono.wav \
        --az_hidden 20 \
        --step_deg 30 \
        --outdir demo_audio
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

# ---------------------------------------------------------------------
# Ensure repo root is on sys.path so `analysis` is importable
# ---------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Now we can import from analysis.eval_methods
from analysis.eval_methods import (
    wrap_deg,
    make_mask,
    predict_spline_periodic,
    predict_neural_field_residual_banded,
)

# ---------------------------------------------------------------------
# Utility: minimum-phase reconstruction from magnitude
# ---------------------------------------------------------------------

def mag_to_minphase_hrir(mag_db, freqs_hz, fs, n_fft=512):
    """
    Rough minimum-phase HRIR from magnitude response.

    mag_db: (F,) magnitude in dB for positive freqs up to Nyquist
    freqs_hz: (F,) corresponding frequencies
    fs: sample rate
    n_fft: FFT size (must be >= 2*(F-1))

    Assumes mag_db corresponds to linearly spaced frequency bins.
    This is an approximation but fine for demo audio.
    """
    mag = 10.0 ** (mag_db / 20.0)

    F = mag.shape[0]
    N = max(n_fft, 2 * (F - 1))

    # Build full spectrum (real, even)
    spec = np.zeros(N, dtype=np.complex64)
    spec[:F] = mag
    spec[F:] = mag[1:-1][::-1]

    # Real cepstrum
    log_mag = np.log(np.abs(spec) + 1e-12)
    ceps = np.fft.ifft(log_mag).real

    # Minimum-phase cepstrum (keep causal part, double middle)
    ceps_min = np.zeros_like(ceps)
    ceps_min[0] = ceps[0]
    ceps_min[1:N//2] = 2 * ceps[1:N//2]

    # Back to spectrum and time domain
    log_mag_min = np.fft.fft(ceps_min)
    spec_min = np.exp(log_mag_min)
    h = np.fft.ifft(spec_min).real

    # Truncate to something reasonable (e.g. 128 taps)
    return h[:128]


def apply_integer_itd(hrir, itd_ms, fs, sign=+1):
    """
    Apply ITD as integer-sample delay (very rough but OK for demo).
    sign = +1 for delaying this ear, -1 for advancing.
    """
    samples = int(round((itd_ms * 1e-3 * fs) / 2.0))  # ±ITD/2 per ear
    shift = sign * samples
    if shift == 0:
        return hrir
    if shift > 0:
        return np.concatenate([np.zeros(shift), hrir])
    else:
        shift = -shift
        return hrir[shift:]


def make_hrirs_from_mag(magL_db, magR_db, freqs_hz, itd_ms, fs):
    """
    Build L/R HRIRs from magnitude + ITD.
    Uses min-phase HRIR and integer-sample ITD shift.
    """
    hL_min = mag_to_minphase_hrir(magL_db, freqs_hz, fs)
    hR_min = mag_to_minphase_hrir(magR_db, freqs_hz, fs)

    hL = apply_integer_itd(hL_min, itd_ms, fs, sign=-1)
    hR = apply_integer_itd(hR_min, itd_ms, fs, sign=+1)

    # Pad to same length
    N = max(len(hL), len(hR))
    hL = np.pad(hL, (0, N - len(hL)))
    hR = np.pad(hR, (0, N - len(hR)))
    return hL, hR

# ---------------------------------------------------------------------
# Helpers to get HRTF at a single azimuth
# ---------------------------------------------------------------------

def get_true_at_az(d, az_target):
    """
    Extract true magL_db, magR_db, itd_ms at nearest measured azimuth.
    d: np.load(...) dict
    """
    az = d["az_deg"].astype(float)
    freqs = d["freqs_hz"].astype(float)
    L = d["magL_db"].astype(float)
    R = d["magR_db"].astype(float)
    ITD = d["itd_ms"].astype(float)

    # wrap & nearest
    a = (az + 180.0) % 360.0 - 180.0
    az_t = (az_target + 180.0) % 360.0 - 180.0
    idx = np.argmin(np.minimum(np.abs(a - az_t), 360.0 - np.abs(a - az_t)))

    return freqs, L[idx], R[idx], ITD[idx]


def get_rbf_at_az(d, az_target, step_deg):
    """
    Use the same periodic spline/RBF interpolation as in eval_methods.py,
    with a keep mask defined by step_deg.
    """
    az = d["az_deg"].astype(float)
    freqs = d["freqs_hz"].astype(float)
    L = d["magL_db"].astype(float)
    R = d["magR_db"].astype(float)
    ITD = d["itd_ms"].astype(float)

    az = wrap_deg(az)
    order = np.argsort(az)
    az, L, R, ITD = az[order], L[order], R[order], ITD[order]

    # Use the same mask logic as evaluate_subject
    keep = make_mask(az, step_deg, offset_deg=0.0)

    predL, predR, preditd = predict_spline_periodic(az, keep, L, R, ITD)

    az_t = (az_target + 180.0) % 360.0 - 180.0
    idx = np.argmin(np.minimum(np.abs(az - az_t), 360.0 - np.abs(az - az_t)))
    return freqs, predL[idx], predR[idx], preditd[idx]


def get_nf_smooth_at_az(npz_path, d, az_target, step_deg,
                        nf_dir="results/neural_field",
                        band_lo=1000.0, band_hi=8000.0):
    """
    Use the NF-smoothed model (same settings as NEURAL_FIELD_SMOOTH in
    evaluate_subject) to get mags + ITD at a given azimuth.

    This calls predict_neural_field_residual_banded once (per subject/step),
    then selects the requested azimuth.
    """
    subj_id = os.path.splitext(os.path.basename(npz_path))[0]

    az = d["az_deg"].astype(float)
    freqs = d["freqs_hz"].astype(float)
    L = d["magL_db"].astype(float)
    R = d["magR_db"].astype(float)
    ITD = d["itd_ms"].astype(float)

    az = wrap_deg(az)
    order = np.argsort(az)
    az, L, R, ITD = az[order], L[order], R[order], ITD[order]

    # Same keep mask as in evaluate_subject
    keep = make_mask(az, step_deg, offset_deg=0.0)

    # Call NF with the same hyperparameters as NEURAL_FIELD_SMOOTH
    predL, predR, preditd = predict_neural_field_residual_banded(
        az, keep, L, R, ITD, freqs,
        step_deg=step_deg,
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
        smooth_output=True,      # APPLY SAVITZKY–GOLAY (smoothed NF)
        sg_window=9,
        sg_poly=3,
    )

    az_t = (az_target + 180.0) % 360.0 - 180.0
    idx = np.argmin(np.minimum(np.abs(az - az_t), 360.0 - np.abs(az - az_t)))
    return freqs, predL[idx], predR[idx], preditd[idx]

# ---------------------------------------------------------------------
# Main: generate A/B/C audio
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to subject .npz file")
    ap.add_argument("--audio_in", required=True, help="Path to mono WAV")
    ap.add_argument("--az_hidden", type=float, default=20.0,
                    help="Hidden/test azimuth (deg)")
    ap.add_argument("--step_deg", type=int, default=30,
                    help="Sparsity step (deg), e.g., 15 or 30")
    ap.add_argument("--fs", type=int, default=48000,
                    help="Target sample rate for output")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if not os.path.isfile(args.audio_in):
        raise FileNotFoundError(f"Input audio file not found: {args.audio_in}")

    d = np.load(args.npz)

    # Load mono source
    x, fs_in = sf.read(args.audio_in)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs_in != args.fs:
        raise RuntimeError(
            "Resampling not implemented in this script; "
            f"please provide a {args.fs} Hz file (got {fs_in} Hz)."
        )

    # --- TRUE ---
    freqs, magL_true, magR_true, itd_true = get_true_at_az(d, args.az_hidden)
    hL_true, hR_true = make_hrirs_from_mag(magL_true, magR_true, freqs, itd_true, args.fs)
    yL_true = fftconvolve(x, hL_true)
    yR_true = fftconvolve(x, hR_true)
    y_true = np.stack([yL_true, yR_true], axis=1)
    sf.write(os.path.join(args.outdir, "demo_true.wav"), y_true, args.fs)

    # --- RBF ---
    freqs, magL_rbf, magR_rbf, itd_rbf = get_rbf_at_az(d, args.az_hidden, args.step_deg)
    hL_rbf, hR_rbf = make_hrirs_from_mag(magL_rbf, magR_rbf, freqs, itd_rbf, args.fs)
    yL_rbf = fftconvolve(x, hL_rbf)
    yR_rbf = fftconvolve(x, hR_rbf)
    y_rbf = np.stack([yL_rbf, yR_rbf], axis=1)
    sf.write(os.path.join(args.outdir, "demo_rbf.wav"), y_rbf, args.fs)

    # --- NF-smoothed ---
    freqs, magL_nf, magR_nf, itd_nf = get_nf_smooth_at_az(
        args.npz, d, args.az_hidden, args.step_deg,
        nf_dir="results/neural_field",
        band_lo=1000.0, band_hi=8000.0,
    )
    hL_nf, hR_nf = make_hrirs_from_mag(magL_nf, magR_nf, freqs, itd_nf, args.fs)
    yL_nf = fftconvolve(x, hL_nf)
    yR_nf = fftconvolve(x, hR_nf)
    y_nf = np.stack([yL_nf, yR_nf], axis=1)
    sf.write(os.path.join(args.outdir, "demo_nf_smooth.wav"), y_nf, args.fs)

    print("Saved A/B/C demo WAVs to", args.outdir)


if __name__ == "__main__":
    main()
