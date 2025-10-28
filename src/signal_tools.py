# src/signal_tools.py
from __future__ import annotations
import numpy as np
from typing import Tuple
from scipy.signal import butter, sosfilt, fftconvolve

# =========================
# Angle + window utilities
# =========================
def wrap_deg(a: np.ndarray | float) -> np.ndarray:
    """Map degrees to [-180, 180)."""
    a = np.asarray(a, dtype=float)
    return (a + 180.0) % 360.0 - 180.0

def hann_window(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(n, dtype=float)
    return np.hanning(n)

# =========================
# Filters / helpers
# =========================
def _lp_sos(fs: float, fc: float = 1200.0, order: int = 3):
    """Low-pass (for ITD LF emphasis)."""
    fc = min(fc, 0.49 * fs)
    return butter(order, fc / (fs * 0.5), btype="low", output="sos")

def _gcc_phat_centered(x: np.ndarray, y: np.ndarray, nfft: int) -> Tuple[np.ndarray, np.ndarray]:
    """Centered GCC-PHAT (zero lag at center)."""
    X = np.fft.rfft(x, nfft)
    Y = np.fft.rfft(y, nfft)
    R = X * np.conj(Y)
    R /= (np.abs(R) + 1e-12)
    r = np.fft.irfft(R, nfft)
    r = np.concatenate((r[-(nfft // 2):], r[:(nfft // 2) + 1]))
    lags = np.arange(-nfft // 2, nfft // 2 + 1, dtype=float)
    return r, lags

def _onset_cum_energy(x: np.ndarray, fs: float, search_ms: float = 12.0, frac: float = 0.06) -> int:
    """
    Onset = first index where cumulative energy within the first search_ms reaches `frac` of total.
    More robust than fixed dB thresholds when direct peak is late/quiet.
    """
    N = max(int(search_ms * 1e-3 * fs), 64)
    xw = np.asarray(x[:N], float)
    e = np.cumsum(xw * xw)
    if e[-1] <= 1e-18:
        return 0
    thr = frac * e[-1]
    return int(np.searchsorted(e, thr))

# ======================================================
# ITD estimation (robust): coarse onset + GCC refine
# ======================================================
def estimate_itd_ms_gcc(
    hrir_L: np.ndarray,
    hrir_R: np.ndarray,
    fs: float,
    lp_hz: float = 1200.0,
    coarse_search_ms: float = 12.0,
    refine_win_ms: float = 6.0,
    max_abs_ms: float = 1.0,
) -> float:
    """
    Robust ITD estimator.
      1) Low-pass to LF region.
      2) Find per-ear onsets by cumulative energy within `coarse_search_ms`.
      3) Coarse lag = iR - iL (samples), clamped to ±max_abs_ms.
      4) Extract a small window around the earlier onset for both ears.
      5) GCC-PHAT in that window; refine peak with parabolic interpolation.
      6) If the GCC peak SNR is weak, fall back to phase-slope ITD.

    Positive ITD means Right ear arrives later than Left.
    Returns ITD in milliseconds.
    """
    L = np.asarray(hrir_L, float)
    R = np.asarray(hrir_R, float)

    # 1) Emphasize LF (fine-structure ITD)
    sos = _lp_sos(fs, lp_hz, order=3)
    Lf = sosfilt(sos, L)
    Rf = sosfilt(sos, R)

    # 2) Coarse onsets (sample indices)
    iL = _onset_cum_energy(Lf, fs, search_ms=coarse_search_ms, frac=0.06)
    iR = _onset_cum_energy(Rf, fs, search_ms=coarse_search_ms, frac=0.06)
    coarse_lag = int(iR - iL)  # samples; R later => positive

    # Clamp to plausible range
    lim = int(max_abs_ms * 1e-3 * fs)
    coarse_lag = int(np.clip(coarse_lag, -lim, lim))

    # 3) Refine window centered near earlier onset (ensure direct path included)
    half = max(8, int(0.5 * refine_win_ms * 1e-3 * fs))
    center = min(iL, iR) + abs(coarse_lag) // 2
    s = max(0, center - half)
    e = min(len(Lf), center + half)
    # ensure same length for both ears
    x = Lf[s:e]
    y = Rf[s:e]
    if len(x) < 16 or len(y) < 16:
        # Fallback to first 2*half samples
        x = Lf[: 2 * half]
        y = Rf[: 2 * half]

    # Per-ear normalisation (makes GCC more robust across angles)
    x = x / (np.max(np.abs(x)) + 1e-12)
    y = y / (np.max(np.abs(y)) + 1e-12)

    # 4) GCC-PHAT refine
    nfft = 1 << (max(len(x), len(y)) - 1).bit_length()
    r, lags = _gcc_phat_centered(x, y, nfft)

    mid = np.where(lags == 0)[0][0]
    r_w = r[mid - lim : mid + lim + 1]
    l_w = lags[mid - lim : mid + lim + 1]

    k = int(np.argmax(r_w))
    lag0 = l_w[k]
    # Parabolic sub-sample refinement
    if 0 < k < len(r_w) - 1:
        y1, y2, y3 = r_w[k - 1], r_w[k], r_w[k + 1]
        denom = (y1 - 2 * y2 + y3)
        delta = 0.5 * (y1 - y3) / denom if abs(denom) > 1e-12 else 0.0
    else:
        delta = 0.0

    lag_refined = float(lag0 + delta)

    # 5) Peak quality check (simple SNR-ish heuristic)
    peak = r_w[k]
    noise = (np.mean(np.abs(r_w)) + np.std(r_w) + 1e-12)
    snr_ok = peak > 3.0 * noise

    if not snr_ok:
        # 6) Fallback: frequency-domain phase-slope on LF band
        return itd_ms_from_phase_slope(L, R, fs, nfft=max(4096, nfft))

    # Sanity clamp & return (ms)
    lag_refined = float(np.clip(lag_refined, -lim, lim))
    return lag_refined / fs * 1000.0

# ======================================================
# Phase-slope ITD fallback (LF IPD regression)
# ======================================================
def itd_ms_from_phase_slope(
    hrir_L: np.ndarray,
    hrir_R: np.ndarray,
    fs: float,
    nfft: int = 4096,
    f_lo: float = 300.0,
    f_hi: float = 1700.0,
) -> float:
    """
    Estimate ITD from interaural phase slope in LF band:
      angle(HR/HL) ≈ -2π f τ  =>  slope gives τ (seconds).
    Returns ITD in milliseconds (Right later than Left => positive).
    """
    HL = np.fft.rfft(hrir_L, nfft)
    HR = np.fft.rfft(hrir_R, nfft)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band):
        return 0.0

    # Interaural phase difference (unwrap)
    IPD = np.angle(HR[band] / (HL[band] + 1e-18))
    IPD = np.unwrap(IPD)

    # Magnitude weighting helps stability
    w = (np.abs(HR[band]) + np.abs(HL[band])) + 1e-12
    x = 2 * np.pi * freqs[band]  # rad/s

    # Weighted least squares: IPD ≈ -x * tau
    A = np.sum(w * x * x)
    b = np.sum(w * x * (-IPD))
    tau = b / (A + 1e-18)  # seconds
    return float(tau * 1000.0)

# ======================================================
# Fractional delay (Lagrange FIR)
# ======================================================
def frac_delay(sig: np.ndarray, delay_samples: float, order: int = 8) -> np.ndarray:
    """
    Apply a fractional delay to 'sig' using (order)-tap Lagrange FIR.
    Positive delay shifts the signal later in time.
    """
    x = np.asarray(sig, dtype=float)
    m = int(order)
    D = float(delay_samples)
    int_delay = int(np.floor(D))
    mu = D - np.floor(D)  # fractional part in [0,1)

    # Lagrange interpolation kernel (length m+1)
    h = np.ones(m + 1, dtype=float)
    idx = np.arange(0, m + 1)
    for i in range(m + 1):
        for j in range(m + 1):
            if i == j:
                continue
            h[i] *= (mu - j) / (i - j)

    y = fftconvolve(x, h, mode="full")
    start = m + int_delay  # account for FIR group delay + integer delay
    start = max(0, start)
    end = start + len(x)
    if end > len(y):
        y = np.pad(y, (0, end - len(y)))
    return y[start:end]

# ======================================================
# Magnitude spectrum (dB)
# ======================================================
def mag_spectrum_db(
    h: np.ndarray,
    nfft: int,
    fs: float,
    floor_db: float = -120.0,
    window: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Magnitude spectrum (in dB) and corresponding frequency grid for a real IR.
    """
    x = np.asarray(h, dtype=float)
    if window:
        x = x * hann_window(len(x))
    H = np.fft.rfft(x, nfft)
    mag = np.abs(H) + 1e-12
    mag_db = 20.0 * np.log10(mag)
    mag_db = np.maximum(mag_db, floor_db)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    return mag_db.astype(np.float32), freqs.astype(np.float32)
