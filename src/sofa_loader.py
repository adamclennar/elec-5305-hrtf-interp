
"""sofa_loader.py
Utilities for loading HRIR datasets from SOFA files.
Requires: pysofaconventions (pip install pysofaconventions)
"""
from __future__ import annotations
import numpy as np

try:
    import pysofaconventions as sofa
except Exception as e:
    raise ImportError(
        "pysofaconventions is required. Install with: pip install pysofaconventions"
    ) from e

def load_hrir_data(sofa_path: str):
    """Load HRIR impulse responses and source positions from a SOFA file.

    Returns
    -------
    hrir : np.ndarray
        Shape [M, 2, N], M measurements, 2 ears (L,R), N samples.
    src_pos : np.ndarray
        Shape [M, 3] with columns [azimuth_deg, elevation_deg, distance_m].
        Some datasets may not supply distances; treat as metadata if present.
    fs : float
        Sampling rate of the HRIR data.
    """
    s = sofa.SOFAFile(sofa_path, "r")
    # HRIR: [M, R, N]
    hrir = np.asarray(s.getDataIR())
    # Positions: often in degrees (az, el, r) following SOFA conventions
    try:
        # pysofaconventions stores SourcePosition as an object with .getValues()
        src_pos = np.asarray(s.SourcePosition.getValues())
    except Exception:
        # Fallback if getValues is not available
        src_pos = np.asarray(s.SourcePosition)
    # Sampling rate
    try:
        fs = float(s.Data.SamplingRate.getValues()[0])
    except Exception:
        fs = float(s.getSamplingRate())
    return hrir, src_pos, fs

def select_elevation(hrir: np.ndarray, src_pos: np.ndarray, elev_deg: float, tol: float = 5.0):
    """Filter measurements near a target elevation in degrees (±tol)."""
    mask = np.isclose(src_pos[:, 1], elev_deg, atol=tol)
    return hrir[mask], src_pos[mask]

def pick_nearest(hrir_subset: np.ndarray, srcpos_subset: np.ndarray, az_deg: float, el_deg: float):
    """Pick the nearest HRIR (2,N) at (az_deg, el_deg) from a subset.

    Returns (h_lr, meta_idx) where h_lr.shape == (2, N)."""
    az = srcpos_subset[:, 0]
    el = srcpos_subset[:, 1]
    d = (az - az_deg) ** 2 + (el - el_deg) ** 2
    idx = int(np.argmin(d))
    return hrir_subset[idx], idx

def list_unique_azimuths(src_pos: np.ndarray) -> np.ndarray:
    """Return sorted unique azimuth angles (degrees)."""
    return np.unique(np.round(src_pos[:, 0], 3))

def list_unique_elevations(src_pos: np.ndarray) -> np.ndarray:
    """Return sorted unique elevation angles (degrees)."""
    return np.unique(np.round(src_pos[:, 1], 3))
