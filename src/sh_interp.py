# src/sh_interp.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple

def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0

def _fourier_design(phi_rad: np.ndarray, L: int) -> np.ndarray:
    """ [1, cos φ, sin φ, ..., cos Lφ, sin Lφ] -> (N, 2L+1) """
    N = phi_rad.shape[0]
    X = np.empty((N, 2*L + 1), dtype=np.float64)
    X[:, 0] = 1.0
    for m in range(1, L+1):
        X[:, 2*m - 1] = np.cos(m * phi_rad)
        X[:, 2*m]     = np.sin(m * phi_rad)
    return X

def _ridge_fit(phi_tr: np.ndarray, Y_tr: np.ndarray, L: int, lam: float):
    """Return coefficients C for ridge fit in Fourier basis."""
    X = _fourier_design(phi_tr, L)              # (K, 2L+1)
    XtX = X.T @ X
    R   = lam * np.eye(XtX.shape[0], dtype=np.float64)
    C   = np.linalg.solve(XtX + R, X.T @ Y_tr)  # (2L+1, D)
    return C

def _ridge_predict(phi_q: np.ndarray, C: np.ndarray, L: int):
    Xq = _fourier_design(phi_q, L)              # (A, 2L+1)
    return Xq @ C                                # (A, D)

def _bands_from_freqs(freqs_hz: np.ndarray, lo=1000., hi=8000., n_bands=48) -> List[np.ndarray]:
    band = (freqs_hz >= lo) & (freqs_hz <= hi)
    idx  = np.where(band)[0]
    splits = np.array_split(idx, n_bands)
    return [np.asarray(s, int) for s in splits if len(s)>0]

def _reduce_bands(mag_db: np.ndarray, bands: List[np.ndarray]) -> np.ndarray:
    # (A,F) -> (A,B)
    return np.stack([mag_db[:, b].mean(axis=1) for b in bands], axis=1)

def _expand_bands_to_full(banded: np.ndarray, bands: List[np.ndarray], F: int) -> np.ndarray:
    # (A,B) -> (A,F) by filling each band with its band value
    A, B = banded.shape
    out = np.zeros((A, F), dtype=np.float64)
    for j, b in enumerate(bands):
        out[:, b] = banded[:, j:j+1]
    return out

def _cv_grid_search(phi_tr: np.ndarray, Y_tr: np.ndarray, L_grid: List[int], lam_grid: List[float]) -> Tuple[int,float]:
    """
    Tiny leave-one-out CV over (L, lam). Y_tr: (K, D).
    K is small (e.g., 6–24), so brute-force LOO is cheap.
    """
    K = phi_tr.shape[0]
    best = (None, None, np.inf)
    for L in L_grid:
        if 2*L+1 > K:  # not identifiable
            continue
        for lam in lam_grid:
            se = 0.0; n = 0
            for i in range(K):
                m = np.ones(K, dtype=bool); m[i] = False
                C = _ridge_fit(phi_tr[m], Y_tr[m], L, lam)
                y_hat = _ridge_predict(phi_tr[i:i+1], C, L)  # (1,D)
                d = Y_tr[i:i+1] - y_hat
                se += float((d*d).mean())  # average over D dims
                n  += 1
            rmse = np.sqrt(se / max(n,1))
            if rmse < best[2]:
                best = (L, lam, rmse)
    if best[0] is None:
        # fallback: smallest admissible L, moderate lam
        L_try = max(1, (K-1)//2)
        return L_try, 1e-2
    return best[0], best[1]

def predict_sh_banded_cv(
    az_deg: np.ndarray,
    keep_mask: np.ndarray,
    magL_db: np.ndarray,   # (A,F)
    magR_db: np.ndarray,   # (A,F)
    itd_ms: np.ndarray,    # (A,)
    freqs_hz: np.ndarray,  # (F,)
    n_bands: int = 48,
    L_grid: Tuple[int,...] = tuple(range(1, 13)),       # try up to order 12
    lam_grid: Tuple[float,...] = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3),
    band_lo: float = 1000., band_hi: float = 8000.,
):
    """
    Azimuthal SH (Fourier) interpolation with:
      - band-averaging (1–8 kHz, n_bands bands)
      - (L, λ) chosen by LOO-CV on kept angles
    Returns full-resolution predictions matching input shapes.
    """
    az  = _wrap_deg(np.asarray(az_deg, float))
    ord = np.argsort(az)
    az  = az[ord]
    Ldb = magL_db[ord].astype(np.float64)
    Rdb = magR_db[ord].astype(np.float64)
    T   = itd_ms[ord].astype(np.float64)
    keep = keep_mask[ord]
    FREQS = np.asarray(freqs_hz, float)

    # Build bands
    bands = _bands_from_freqs(FREQS, band_lo, band_hi, n_bands)
    Lb = _reduce_bands(Ldb, bands)    # (A,B)
    Rb = _reduce_bands(Rdb, bands)    # (A,B)

    # Train set (kept angles)
    phi_tr = np.deg2rad(az[keep])     # (K,)
    if phi_tr.size < 3:
        raise RuntimeError("Too few kept angles for SH (need ≥3).")

    # Stack features for joint fit across channels (Left+Right bands and ITD)
    # Y_tr shape: (K, D) with D = 2B + 1
    Y_tr = np.concatenate([Lb[keep], Rb[keep], T[keep].reshape(-1,1)], axis=1)

    # CV choose L, lam
    L_opt, lam_opt = _cv_grid_search(phi_tr, Y_tr, list(L_grid), list(lam_grid))

    # Fit on ALL kept with chosen (L, λ), then predict at all angles
    C = _ridge_fit(phi_tr, Y_tr, L_opt, lam_opt)
    Y_pred = _ridge_predict(np.deg2rad(az), C, L_opt)   # (A, 2B+1)

    B = len(bands)
    Lb_p = Y_pred[:, :B]
    Rb_p = Y_pred[:, B:2*B]
    T_p  = Y_pred[:, 2*B]

    # Expand banded back to full F
    L_full = _expand_bands_to_full(Lb_p, bands, Ldb.shape[1])
    R_full = _expand_bands_to_full(Rb_p, bands, Rdb.shape[1])

    # Unsort back to input order
    inv = np.empty_like(ord); inv[ord] = np.arange(ord.size)
    return L_full[inv], R_full[inv], T_p[inv], {"L": L_opt, "lam": lam_opt, "bands": len(bands)}
