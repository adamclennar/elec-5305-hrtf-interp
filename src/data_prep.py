# src/data_prep.py
from __future__ import annotations
from scipy.signal import savgol_filter
import os, json, hashlib
import re
import numpy as np
from typing import Tuple
from src.sofa_loader import load_hrir_data, select_elevation
from src.signal_tools import (
    estimate_itd_ms_gcc,
    frac_delay,
    mag_spectrum_db,
    itd_ms_from_phase_slope,
)

_SUBJ_RE = re.compile(r"^NH\d{1,3}$", re.IGNORECASE)

def sofa_to_features(
    sofa_path: str,
    out_npz: str,
    elev: float = 0.0,
    tol: float = 5.0,
    nfft: int = 2048,
    lp_hz_for_itd: float = 1500.0,
    itd_window_ms: float = 5.0,
) -> str:
    """
    Convert a SOFA HRIR file into compact features for interpolation:
      - azimuths (deg) at target elevation band (elev±tol)
      - ITD per azimuth (ms) estimated via GCC-PHAT
      - per-ear magnitude spectra in dB on fixed frequency grid
      - sample rate and a bit of metadata

    Saves an .npz at `out_npz` and returns that path.
    """
    # 1) Load HRIRs & positions
    hrir, src_pos, fs = load_hrir_data(sofa_path)  # hrir[M,2,N], src_pos[M,3] in deg,deg,m
    # 2) Select elevation slice
    h_sub, p_sub = select_elevation(hrir, src_pos, elev, tol=tol)
    if h_sub.shape[0] == 0:
        raise RuntimeError(f"No HRIRs within elev {elev}±{tol} deg in {os.path.basename(sofa_path)}")

    # 3) Sort by azimuth (for consistency)
    az = p_sub[:, 0].astype(float)                 # degrees
    order = np.argsort(((az + 180.0) % 360.0) - 180.0)
    az = az[order]
    h_sub = h_sub[order]                           # [A, 2, N]
    A, _, N = h_sub.shape

    # 4) Estimate ITD per azimuth (ms) — phase-slope primary, GCC fallback
    itd_ms = np.zeros(A, dtype=np.float32)
    for i in range(A):
        # Primary: phase-slope in 200–1500 Hz
        tau = itd_ms_from_phase_slope(h_sub[i, 0], h_sub[i, 1], fs, nfft=4096, f_lo=200, f_hi=1500)
        # Sanity clamp & fallback to GCC if phase estimate is too tiny/unreliable
        if not np.isfinite(tau) or abs(tau) < 1e-3:  # < 0.001 ms suspiciously near 0
            tau = estimate_itd_ms_gcc(h_sub[i, 0], h_sub[i, 1], fs,
                                  lp_hz=1200.0, coarse_search_ms=12.0,
                                  refine_win_ms=6.0, max_abs_ms=1.0)
        # Final clamp
        tau = -float(tau) # positive ITD means R delayed vs L
        itd_ms[i] = float(np.clip(tau, -1.0, 1.0))
        
    # 5) Remove ITD: time-align ears with ±ITD/2 fractional delay
    #    (We align so residual describes purely spectral coloration.)
    h_aligned = np.zeros_like(h_sub, dtype=np.float64)
    for i in range(A):
        d_samp = (itd_ms[i] / 1000.0) * fs
        # positive ITD means R later than L -> apply -d/2 to L, +d/2 to R
        hL = frac_delay(h_sub[i, 0], -0.5 * d_samp, order=8)
        hR = frac_delay(h_sub[i, 1],  0.5 * d_samp, order=8)
        # ensure equal length (clip/pad)
        minN = min(len(hL), len(hR), N)
        h_aligned[i, 0, :minN] = hL[:minN]
        h_aligned[i, 1, :minN] = hR[:minN]

    # 6) Per-ear magnitude spectra (dB) on a fixed frequency grid
    magL_db = np.zeros((A, (nfft // 2) + 1), dtype=np.float32)
    magR_db = np.zeros_like(magL_db)
    for i in range(A):
        magL_db[i], freqs = mag_spectrum_db(h_aligned[i, 0], nfft=nfft, fs=fs, floor_db=-120.0, window=True)
        magR_db[i], _     = mag_spectrum_db(h_aligned[i, 1], nfft=nfft, fs=fs, floor_db=-120.0, window=True)

    # 7) Save compact feature file
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    meta = {
        "source": os.path.abspath(sofa_path),
        "elev_target_deg": float(elev),
        "elev_tol_deg": float(tol),
        "nfft": int(nfft),
        "lp_hz_for_itd": float(lp_hz_for_itd),
        "itd_window_ms": float(itd_window_ms),
    }
    np.savez_compressed(
        out_npz,
        az_deg=az.astype(np.float32),
        freqs_hz=freqs.astype(np.float32),
        fs=float(fs),
        itd_ms=itd_ms,
        magL_db=magL_db,
        magR_db=magR_db,
        hrir_len=int(N),
        meta=json.dumps(meta),
    )
    return out_npz

def _subject_from_path(sofa_path: str) -> str:
    """
    Derive subject tag from parent folder name (e.g., 'NH5', 'NH55', 'NH123').
    Falls back to parent folder even if it doesn't match NH\d+.
    """
    parent = os.path.basename(os.path.dirname(os.path.abspath(sofa_path)))
    return parent if _SUBJ_RE.match(parent or "") else (parent or "UNKNOWN")

def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in name)

# then:

def bulk_sofa_to_npz(list_file: str, out_dir: str, elev: float = 0.0, tol: float = 5.0, nfft: int = 2048):
    """
    Convert a newline-separated list of SOFA paths into .npz features in out_dir.
    Output filename: <SUBJECT>__<sofa-base>__elev<elev>.npz
    Example: data_npz/test/NH55__hrtf_M_hrtf B__elev0.npz
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(list_file, "r") as f:
        paths = [ln.rstrip("\n") for ln in f if ln.strip()]   # keep spaces in filenames

    if not paths:
        print(f"[WARN] No SOFA paths in {list_file}")
        return

    for p in paths:
        try:
            subj = _subject_from_path(p)                        # e.g., 'NH55'
             # e.g., 'hrtf_M_hrtf B'
            base = _sanitize(os.path.splitext(os.path.basename(p))[0])
            out_npz = os.path.join(out_dir, f"{subj}__{base}__elev{int(elev)}.npz")
            sofa_to_features(p, out_npz, elev=elev, tol=tol, nfft=nfft)
            print(f"[OK] {subj}: {os.path.basename(p)} -> {out_npz}")
        except Exception as e:
            print(f"[WARN] Skipped {p}: {e}")

    print(f"Done. NPZs written in {out_dir}")
