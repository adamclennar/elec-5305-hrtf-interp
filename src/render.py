
"""render.py
Offline mono->binaural rendering via HRIR convolution.
Requires: numpy, scipy, soundfile
"""
from __future__ import annotations
import os
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, butter, sosfilt

from .sofa_loader import load_hrir_data, select_elevation, pick_nearest

def loudness_normalise(x: np.ndarray, ref_rms: float = 0.1, eps: float = 1e-12) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x**2) + eps))
    if rms == 0.0:
        return x
    return x * (ref_rms / rms)

def distance_filters(x: np.ndarray, fs: float, dist_m: float) -> np.ndarray:
    """Crude distance model: 1/r gain, air absorption LP, and geometric pre-delay."""
    dist_m = max(1e-6, float(dist_m))
    gain = 1.0 / dist_m
    delay_s = dist_m / 343.0
    delay_n = int(round(delay_s * fs))
    x_del = np.concatenate([np.zeros(delay_n, dtype=x.dtype), x])
    # Simple LP where cutoff decreases with distance (very rough psychoacoustics)
    fc = max(2000.0, 18000.0 - (dist_m - 1.0) * 3000.0)
    fc = min(fc, 20000.0)
    sos = butter(1, fc / (fs * 0.5), btype='low', output='sos')
    x_f = sosfilt(sos, x_del) * gain
    return x_f

def convolve_lr(x: np.ndarray, h_l: np.ndarray, h_r: np.ndarray) -> np.ndarray:
    y_l = fftconvolve(x, h_l, mode='full')
    y_r = fftconvolve(x, h_r, mode='full')
    n = max(len(y_l), len(y_r))
    if len(y_l) < n:
        y_l = np.pad(y_l, (0, n - len(y_l)))
    if len(y_r) < n:
        y_r = np.pad(y_r, (0, n - len(y_r)))
    return np.stack([y_l, y_r], axis=0)

def spatialise_mono(x: np.ndarray, fs: float, h_lr: np.ndarray, dist_m: float = 1.5) -> np.ndarray:
    x = loudness_normalise(x)
    x_d = distance_filters(x, fs, dist_m)
    h_l, h_r = h_lr[0], h_lr[1]
    y_lr = convolve_lr(x_d, h_l, h_r)
    # Safe normalisation
    peak = float(np.max(np.abs(y_lr))) if y_lr.size else 1.0
    if peak > 0:
        y_lr = y_lr / peak
    return y_lr

def render_file(in_wav: str, out_wav: str, sofa_path: str, az_deg: float = 30.0, el_deg: float = 0.0, dist_m: float = 1.5, elev_tol: float = 5.0):
    # Load audio
    x, fs = sf.read(in_wav)
    if x.ndim > 1:
        x = np.mean(x, axis=1)  # downmix
    # Load HRIR dataset
    hrir, src_pos, fs_hrir = load_hrir_data(sofa_path)
    if int(fs) != int(fs_hrir):
        raise ValueError(f"Sample-rate mismatch: audio {fs} Hz vs HRIR {fs_hrir} Hz. Resample one to match.")
    # Subset to elevation
    h_sub, p_sub = select_elevation(hrir, src_pos, el_deg, tol=elev_tol)
    if h_sub.shape[0] == 0:
        raise ValueError(f"No HRIRs near elevation {el_deg}° (±{elev_tol}°). Available: {np.unique(np.round(src_pos[:,1],1))}")
    h_lr, _ = pick_nearest(h_sub, p_sub, az_deg, el_deg)
    y_lr = spatialise_mono(x, fs, h_lr, dist_m=dist_m)
    os.makedirs(os.path.dirname(out_wav) or '.', exist_ok=True)
    sf.write(out_wav, y_lr.T, fs)
    return out_wav

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Mono->Binaural HRIR Renderer (offline)")
    ap.add_argument("--in", dest="in_wav", required=True, help="Input mono/stereo WAV path")
    ap.add_argument("--out", dest="out_wav", required=True, help="Output binaural WAV path")
    ap.add_argument("--sofa", dest="sofa_path", required=True, help="SOFA HRIR dataset path")
    ap.add_argument("--az", type=float, default=30.0, help="Azimuth in degrees")
    ap.add_argument("--el", type=float, default=0.0, help="Elevation in degrees")
    ap.add_argument("--dist", type=float, default=1.5, help="Distance in meters (crude model)")
    ap.add_argument("--elev_tol", type=float, default=5.0, help="Elevation tolerance in degrees for matching")
    args = ap.parse_args()
    out = render_file(args.in_wav, args.out_wav, args.sofa_path, az_deg=args.az, el_deg=args.el, dist_m=args.dist, elev_tol=args.elev_tol)
    print(f"Saved: {out}")
