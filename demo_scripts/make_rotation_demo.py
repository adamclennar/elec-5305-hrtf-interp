#!/usr/bin/env python3
"""
make_rotation_demo.py

Generate a 360-ish degree rotation demo for one interpolation method
(e.g. NF-smoothed). The source appears to move around the listener.

Usage (from repo root):
    PYTHONPATH=. python demo_scripts/make_rotation_demo.py \
        --npz data_npz/test/NH43__hrtf_M_hrtf_B__elev0.npz \
        --audio_in demo_sources/pinknoise_mono.wav \
        --method nf_smooth \
        --step_deg 30 \
        --out demo_audio/rotation_nf_smooth.wav
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from analysis.eval_methods import wrap_deg

# ---------------------------------------------------------------------
# Ensure repo root is on sys.path so `analysis` is importable
# ---------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from analysis.eval_methods import (
    wrap_deg,
    make_mask,
    predict_spline_periodic,
    predict_neural_field_residual_banded,
)

# ---------------------------------------------------------------------
# HRIR helpers (duplicated from make_abc_demo for simplicity)
# ---------------------------------------------------------------------

def mag_to_minphase_hrir(mag_db, freqs_hz, fs, n_fft=512):
    """
    Rough minimum-phase HRIR from magnitude response.
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
# Method helpers (per-azimuth HRTF)
# ---------------------------------------------------------------------

def get_at_az_nn(d, az_target):
    """
    NN baseline: simply pick nearest measured angle (no interpolation).
    """
    az = d["az_deg"].astype(float)
    freqs = d["freqs_hz"].astype(float)
    L = d["magL_db"].astype(float)
    R = d["magR_db"].astype(float)
    ITD = d["itd_ms"].astype(float)

    a = (az + 180.0) % 360.0 - 180.0
    az_t = (az_target + 180.0) % 360.0 - 180.0
    idx = np.argmin(np.minimum(np.abs(a - az_t), 360.0 - np.abs(a - az_t)))
    return freqs, L[idx], R[idx], ITD[idx]


def get_at_az_rbf(d, az_target, step_deg):
    """
    RBF/spline interpolation at a given azimuth,
    using same mask logic as evaluate_subject.
    """
    az = d["az_deg"].astype(float)
    freqs = d["freqs_hz"].astype(float)
    L = d["magL_db"].astype(float)
    R = d["magR_db"].astype(float)
    ITD = d["itd_ms"].astype(float)

    az = wrap_deg(az)
    order = np.argsort(az)
    az, L, R, ITD = az[order], L[order], R[order], ITD[order]

    # Same keep mask as main pipeline
    keep = make_mask(az, step_deg, offset_deg=0.0)

    predL, predR, preditd = predict_spline_periodic(az, keep, L, R, ITD)

    az_t = (az_target + 180.0) % 360.0 - 180.0
    idx = np.argmin(np.minimum(np.abs(az - az_t), 360.0 - np.abs(az - az_t)))
    return freqs, predL[idx], predR[idx], preditd[idx]


def get_at_az_nf_smooth(npz_path, d, az_target, step_deg,
                        nf_dir="results/neural_field",
                        band_lo=1000.0, band_hi=8000.0):
    """
    NF-smoothed interpolation at a given azimuth,
    mirroring the NEURAL_FIELD_SMOOTH settings in evaluate_subject.
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

    keep = make_mask(az, step_deg, offset_deg=0.0)

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
        smooth_output=True,      # APPLY SAVITZKY–GOLAY
        sg_window=9,
        sg_poly=3,
    )

    az_t = (az_target + 180.0) % 360.0 - 180.0
    idx = np.argmin(np.minimum(np.abs(az - az_t), 360.0 - np.abs(az - az_t)))
    return freqs, predL[idx], predR[idx], preditd[idx]

# ---------------------------------------------------------------------
# Main: rotation demo
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--audio_in", required=True, help="Mono WAV")
    ap.add_argument("--method", required=True, choices=["nn", "rbf", "nf_smooth"])
    ap.add_argument("--step_deg", type=int, default=30,
                    help="Sparsity step (deg), e.g. 15 or 30")
    ap.add_argument("--fs", type=int, default=48000)
    ap.add_argument("--segment_sec", type=float, default=0.1,
                    help="Duration per azimuth segment (s)")
    ap.add_argument("--out", required=True, help="Output WAV")
    args = ap.parse_args()

    d = np.load(args.npz)

    # Load mono source
    x, fs_in = sf.read(args.audio_in)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs_in != args.fs:
        raise RuntimeError("Resampling not implemented; supply {} Hz audio (got {} Hz)".format(args.fs, fs_in))

    seg_len = int(args.segment_sec * args.fs)
    if len(x) < seg_len:
        # loop or pad
        reps = int(np.ceil(seg_len / len(x)))
        x = np.tile(x, reps)
    x_seg = x[:seg_len]

    # Option A: walk through all measured azimuths (true rotation)
    use_measured_path = True  # set False to go back to synthetic path

    if use_measured_path:
        az_meas = d["az_deg"].astype(float)
        az_meas = wrap_deg(az_meas)
        az_path = np.sort(az_meas)
    else:
        # Original synthetic path: -80° -> +80° -> -80°
        path1 = np.linspace(-80, 80, 17)
        path2 = np.linspace(80, -80, 17)
        az_path = np.concatenate([path1, path2])

    # Build method dispatcher with closures to include npz_path, step_deg, etc.
    methods = {
        "nn":        lambda az: get_at_az_nn(d, az),
        "rbf":       lambda az: get_at_az_rbf(d, az, args.step_deg),
        "nf_smooth": lambda az: get_at_az_nf_smooth(args.npz, d, az, args.step_deg,
                                                    nf_dir="results/neural_field",
                                                    band_lo=1000.0, band_hi=8000.0),
    }
    get_fun = methods[args.method]

    segments_L = []
    segments_R = []

    for az in az_path:
        freqs, magL, magR, itd = get_fun(az)
        hL, hR = make_hrirs_from_mag(magL, magR, freqs, itd, args.fs)

        yL = fftconvolve(x_seg, hL)
        yR = fftconvolve(x_seg, hR)

        # Trim to same length as segment for simple concatenation
        yL = yL[:seg_len]
        yR = yR[:seg_len]

        segments_L.append(yL)
        segments_R.append(yR)

    yL_full = np.concatenate(segments_L)
    yR_full = np.concatenate(segments_R)
    y_full = np.stack([yL_full, yR_full], axis=1)

    # Ensure output dir exists
    outdir = os.path.dirname(args.out)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    sf.write(args.out, y_full, args.fs)
    print("Saved rotation demo to", args.out)
    print("Method:", args.method)
    print("Azimuth path (deg):", az_path)


if __name__ == "__main__":
    main()
