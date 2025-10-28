
"""sofa_loader.py (robust)
Utilities for loading HRIR datasets from SOFA files.
Tries multiple backends/APIs to support different pysofaconventions versions.

pip install pysofaconventions  (preferred)
# optional fallback:
# pip install pysofa
"""
from __future__ import annotations
import numpy as np

# Try to import one of the SOFA libraries
_backend = None
try:
    import pysofaconventions as _sofa
    _backend = "pysofaconventions"

except Exception as e:
    raise ImportError(
        "No SOFA reader found. Install one of: 'pysofaconventions' or 'pysofa'."
    ) from e

def _open_sofa(path):
    return _sofa.SOFAFile(path, "r")

def _get_hrir(s):
    # Both libs provide getDataIR()
    return np.asarray(s.getDataIR())

def _get_fs(s):
    # Try common ways to fetch sampling rate
    for getter in (
        lambda: float(s.Data.SamplingRate.getValues()[0]),  # pysofaconventions
        lambda: float(s.getSamplingRate()),                 # generic
    ):
        try:
            return getter()
        except Exception:
            pass
    raise AttributeError("Could not retrieve SamplingRate from SOFA file.")

def _get_srcpos(s):
    """Return SourcePosition as np.ndarray [M,3] (az, el, r).
    Tries several access patterns across libs/versions.
    """
    # 1) Preferred: variable getter
    for getter in (
        lambda: s.getVariableValue("SourcePosition"),     # pysofaconventions
        lambda: s.getVariable("SourcePosition").getValues(),  # some versions
    ):
        try:
            v = getter()
            return np.asarray(v)
        except Exception:
            pass

    # 2) Attribute-style (older bindings)
    for name in ("SourcePosition", "sourcePosition", "Positions", "Source_Positions"):
        try:
            v = getattr(s, name)
            # some objects wrap values, try .getValues() if present
            try:
                v = v.getValues()
            except Exception:
                pass
            return np.asarray(v)
        except Exception:
            pass

    # 3) Fall back: look for any variable with 'SourcePosition' in its name
    try:
        varnames = []
        try:
            varnames = s.getVariableNames()  # pysofaconventions >= 0.3
        except Exception:
            pass
        for vn in (varnames or []):
            if "SourcePosition".lower() in str(vn).lower():
                try:
                    v = s.getVariableValue(vn)
                    return np.asarray(v)
                except Exception:
                    continue
    except Exception:
        pass

    raise AttributeError("Could not retrieve SourcePosition from SOFA file. This file may be non-standard or corrupted.")

def load_hrir_data(sofa_path: str):
    """Load HRIR impulse responses and source positions from a SOFA file.
    Returns
    -------
    hrir : np.ndarray  # [M, 2, N]
    src_pos : np.ndarray  # [M, 3] columns [az_deg, el_deg, r_m] (if r is present)
    fs : float
    """
    s = _open_sofa(sofa_path)
    hrir = _get_hrir(s)
    src_pos = _get_srcpos(s)
    fs = _get_fs(s)
    # Normalize shape (ensure 2 channels dimension is in the middle)
    if hrir.ndim != 3:
        raise ValueError(f"Unexpected HRIR shape {hrir.shape}; expected [M,2,N].")
    if hrir.shape[1] != 2 and hrir.shape[0] == 2:
        # Some files might be [2, M, N]; transpose to [M,2,N]
        hrir = np.transpose(hrir, (1, 0, 2))
    if hrir.shape[1] != 2:
        raise ValueError(f"HRIR does not appear to have 2 ears. Got shape {hrir.shape}.")
    return hrir, src_pos, fs

def select_elevation(hrir: np.ndarray, src_pos: np.ndarray, elev_deg: float, tol: float = 5.0):
    """Filter measurements near a target elevation in degrees (±tol)."""
    if src_pos.shape[1] < 2:
        raise ValueError("SourcePosition has fewer than 2 columns; cannot filter by elevation.")
    mask = np.isclose(src_pos[:, 1], elev_deg, atol=tol)
    return hrir[mask], src_pos[mask]

def pick_nearest(hrir_subset: np.ndarray, srcpos_subset: np.ndarray, az_deg: float, el_deg: float):
    """Pick the nearest HRIR (2,N) at (az_deg, el_deg) from a subset."""
    az = srcpos_subset[:, 0]
    el = srcpos_subset[:, 1] if srcpos_subset.shape[1] > 1 else np.zeros_like(az)
    d = (az - az_deg) ** 2 + (el - el_deg) ** 2
    idx = int(np.argmin(d))
    return hrir_subset[idx], idx
