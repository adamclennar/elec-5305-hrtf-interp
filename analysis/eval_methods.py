#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, csv, glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from scipy.interpolate import CubicSpline
import torch
from scipy.signal import savgol_filter
from src.models.cnn_interp import CNNInterp1D
from src.sh_interp import predict_sh_banded_cv
from src.models.neural_field import ResidualNeuralFieldBanded, GlobalResidualNeuralFieldBanded




# ---------------------------
# helpers
# ---------------------------
def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0

def make_mask(az_deg: np.ndarray, step_deg: float, offset_deg: float = 0.0) -> np.ndarray:
    a = wrap_deg(az_deg - offset_deg)
    keep = (np.abs((a / step_deg) - np.round(a / step_deg)) < 1e-6)
    if not keep.any():  # guarantee at least one kept angle
        q = a / step_deg
        j = int(np.argmin(np.abs(q - np.round(q))))
        keep[j] = True
    return keep

def band_idx(freqs_hz: np.ndarray, lo=1000.0, hi=8000.0):
    return (freqs_hz >= lo) & (freqs_hz <= hi)

def lsd_db(pred_db: np.ndarray, true_db: np.ndarray, band: np.ndarray) -> float:
    D = (pred_db[:, band] - true_db[:, band])
    return float(np.sqrt(np.mean(D**2)))

def lsd_db_banded(pred_db, true_db, freqs, lo=1000.0, hi=8000.0, n_bands=48, mask=None):
    band = (freqs >= lo) & (freqs <= hi)
    idx  = np.where(band)[0]
    splits = np.array_split(idx, n_bands)
    def reduce(x):  # (A,F) -> (A,B)
        return np.stack([x[:, s].mean(axis=1) for s in splits if len(s) > 0], axis=1)
    Pb = reduce(pred_db); Tb = reduce(true_db)
    if mask is not None:
        Pb = Pb[mask]; Tb = Tb[mask]
    D = Pb - Tb
    return float(np.sqrt(np.mean(D**2)))

def ild_db(magR_db: np.ndarray, magL_db: np.ndarray) -> np.ndarray:
    return (magR_db - magL_db)

def _uniq_mean(x, Y):
    """Deduplicate x (1D) by averaging rows in Y for identical x values."""
    x = np.asarray(x)
    order = np.argsort(x)
    x = x[order]; Y = Y[order]
    ux, idx_start, counts = np.unique(x, return_index=True, return_counts=True)
    outY = []
    for s, c in zip(idx_start, counts):
        outY.append(Y[s:s+c].mean(axis=0))
    return ux, np.vstack(outY)

# ---------------------------
# baselines
# ---------------------------
def predict_nn(az, keep_mask, magL_db, magR_db, itd_ms):
    A = len(az)
    predL = magL_db.copy(); predR = magR_db.copy(); preditd = itd_ms.copy()
    kept_angles = az[keep_mask]
    if kept_angles.size == 0:
        j0 = 0
        predL[:] = magL_db[j0]; predR[:] = magR_db[j0]; preditd[:] = itd_ms[j0]
        return predL, predR, preditd
    kept_idx = np.where(keep_mask)[0]
    for i in range(A):
        if keep_mask[i]:
            continue
        d = np.minimum(np.abs(az[i] - kept_angles), 360.0 - np.abs(az[i] - kept_angles))
        j = kept_idx[np.argmin(d)]
        predL[i] = magL_db[j]; predR[i] = magR_db[j]; preditd[i] = itd_ms[j]
    return predL, predR, preditd

def _band_indices(freqs, f_lo=1000., f_hi=8000., n_bands=48):
    band = (freqs >= f_lo) & (freqs <= f_hi)
    idx = np.where(band)[0]
    splits = np.array_split(idx, n_bands)
    return [np.array(s, dtype=int) for s in splits if len(s) > 0]

def _reduce_bands(mag_db, bands):
    return np.stack([mag_db[:, b].mean(axis=1) for b in bands], axis=1)

def _expand_bands_to_full(banded, bands, F):
    A, B = banded.shape
    out = np.zeros((A, F), dtype=np.float32)
    for j, b in enumerate(bands):
        out[:, b] = banded[:, j:j+1]
    return out

def predict_cnn(az, keep_mask, magL_db, magR_db, itd_ms, freqs, ckpt_path, n_bands=48):
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # --- load checkpoint and infer architecture ---
    ck = torch.load(ckpt_path, map_location=dev)
    sd = ck["model"]

    # infer input channels from stem conv
    for k in ("stem.0.weight", "stem.0.0.weight", "stem.0.conv.weight"):
        if k in sd:
            c_in_expected = sd[k].shape[1]
            break
    else:
        raise RuntimeError("Cannot infer c_in from checkpoint state_dict.")

    # infer width
    width_expected = None
    for k in ("blocks.0.conv1.weight","blocks.0.bn1.weight",
              "blocks.0.conv2.weight","blocks.0.bn2.weight"):
        if k in sd:
            width_expected = sd[k].shape[0]
            break
    if width_expected is None:
        cand = [v.shape[0] for k,v in sd.items()
                if k.startswith("blocks.") and (
                    k.endswith(".weight") or
                    k.endswith(".running_var") or
                    k.endswith(".running_mean")
                ) and len(v.shape) >= 1]
        if not cand:
            raise RuntimeError("Cannot infer width from checkpoint state_dict.")
        width_expected = int(max(cand))

    # infer depth
    depth_expected = 0
    while any(key.startswith(f"blocks.{depth_expected}.") for key in sd.keys()):
        depth_expected += 1
    if depth_expected == 0:
        raise RuntimeError("Cannot infer depth from checkpoint state_dict.")

    # choose bands to match training
    nb_ck = int(ck.get("bands", n_bands))
    n_bands = nb_ck  # override to avoid mismatches

    # --- features: banding + z-score ---
    bands = _band_indices(freqs, 1000., 8000., n_bands)
    Lb = _reduce_bands(magL_db, bands)    # (A,B) dB
    Rb = _reduce_bands(magR_db, bands)    # (A,B) dB
    T  = itd_ms.reshape(-1,1)             # (A,1) ms

    def zscore(AxB):
        m = AxB.mean(axis=0, keepdims=True)
        s = AxB.std(axis=0, keepdims=True) + 1e-6
        return (AxB - m)/s, m, s

    Lz, mL, sL = zscore(Lb)
    Rz, mR, sR = zscore(Rb)
    Tz, mT, sT = zscore(T)

    feats_z     = np.concatenate([Lz, Rz], axis=1)          # (A, 2B)

    def _nn_fill(az_deg, keep_mask, arr):
        A, D = arr.shape
        out = arr.copy()
        idx_keep = np.where(keep_mask)[0]
        kept_angles = az_deg[idx_keep]
        kept_feat   = arr[idx_keep]
        for i in range(A):
            if keep_mask[i]:
                continue
            d = np.minimum(np.abs(az_deg[i]-kept_angles), 360.0-np.abs(az_deg[i]-kept_angles))
            j = int(np.argmin(d))
            out[i] = kept_feat[j]
        return out

    feats_init  = _nn_fill(az, keep_mask, feats_z)          # (A, 2B)
    Tinit_z     = _nn_fill(az, keep_mask, Tz).reshape(-1,1) # (A, 1)

    # did training include sin/cos positional channels?
    with_pos = (c_in_expected == (2*n_bands + 4))

    cols = [feats_init, Tinit_z, keep_mask.astype(float).reshape(-1,1)]
    if with_pos:
        pos = np.stack([np.sin(np.deg2rad(az)), np.cos(np.deg2rad(az))], axis=1)
        cols.append(pos)
    X = np.concatenate(cols, axis=1).astype(np.float32)      # (A, c_in)
    assert X.shape[1] == c_in_expected, f"Built X with {X.shape[1]} chans, ckpt expects {c_in_expected}"

    # --- build model that matches checkpoint exactly ---
    c_out = 2*n_bands + 1
    print(f"[predict_cnn] ckpt bands={nb_ck} -> using c_in={c_in_expected}, width={width_expected}, depth={depth_expected}")
    model = CNNInterp1D(c_in_expected, c_out, width=width_expected, depth=depth_expected).to(dev)
    model.load_state_dict(sd, strict=True)
    model.eval()

    # --- forward (predict residual in z-space) ---
    X_t = torch.from_numpy(X).transpose(0,1).unsqueeze(0).to(dev)  # (1,C_in,A)
    with torch.no_grad():
        dY = model(X_t).cpu().squeeze(0).transpose(0,1).numpy()    # (A, 2B+1)
    Y0 = np.concatenate([feats_init, Tinit_z], axis=1)             # (A, 2B+1)
    Yz = Y0 + dY

    # --- un-zscore back to dB/ms, expand to full F ---
    B = n_bands
    Lb_p = Yz[:, :B]       * sL + mL
    Rb_p = Yz[:, B:2*B]    * sR + mR
    T_p  = (Yz[:, 2*B:2*B+1] * sT + mT).reshape(-1)

    F = magL_db.shape[1]
    L_full = _expand_bands_to_full(Lb_p, bands, F)
    R_full = _expand_bands_to_full(Rb_p, bands, F)
    return L_full.astype(np.float32), R_full.astype(np.float32), T_p.astype(np.float32)


def predict_spline_periodic(az, keep_mask, magL_db, magR_db, itd_ms):
    A, F = magL_db.shape
    predL = magL_db.copy(); predR = magR_db.copy()
    x_all = az.astype(float); order_all = np.argsort(x_all)
    x_all = x_all[order_all]
    L_all = magL_db[order_all]; R_all = magR_db[order_all]; T_all = itd_ms[order_all]
    kept = keep_mask[order_all]
    xk = x_all[kept]
    if xk.size < 4: raise RuntimeError("Too few kept angles for periodic spline.")
    Lk = L_all[kept]; Rk = R_all[kept]; Tk = T_all[kept]
    xk_u, Lk_u = _uniq_mean(xk, Lk); _, Rk_u = _uniq_mean(xk, Rk)
    _, Tk_u = _uniq_mean(xk, Tk.reshape(-1,1)); Tk_u = Tk_u[:,0]
    x0 = xk_u[0]; xk_per = np.concatenate([xk_u, xk_u[:1] + 360.0])
    xq = x_all.copy(); xq[xq < x0] += 360.0
    for f in range(F):
        yL = np.concatenate([Lk_u[:, f], Lk_u[:1, f]])
        yR = np.concatenate([Rk_u[:, f], Rk_u[:1, f]])
        csL = CubicSpline(xk_per, yL, bc_type='periodic')
        csR = CubicSpline(xk_per, yR, bc_type='periodic')
        predL[:, f] = csL(xq); predR[:, f] = csR(xq)
    yT = np.concatenate([Tk_u, Tk_u[:1]])
    csT = CubicSpline(xk_per, yT, bc_type='periodic'); preditd = csT(xq)
    inv = np.empty_like(order_all); inv[order_all] = np.arange(A)
    return predL[inv], predR[inv], preditd[inv]


# # ---------------------------
# # Neural Field predictor (NEW)
# # ---------------------------
# def predict_neural_field(
#     az,
#     keep_mask,
#     magL_db,
#     magR_db,
#     freqs,
#     step_deg: int,
#     subject_id: str,
#     nf_dir: str = "results/neural_field",
#     band_lo: float = 1000.0,
#     band_hi: float = 8000.0,
#     freq_bands: int = 8,
#     width: int = 128,
#     depth: int = 6,
#     epochs: int = 400,
#     batch_size: int = 4096,
#     lr: float = 3e-4,
# ):
#     """
#     Coordinate MLP neural field:
#       - train per subject, using only kept angles
#       - fit mag_dB(az, freq, ear) over [band_lo, band_hi]
#       - ITD is taken from NN baseline (we don't model it here).

#     If a checkpoint exists for (subject_id, step_deg), load it instead of retraining.
#     """

#     os.makedirs(nf_dir, exist_ok=True)
#     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # frequency band of interest
#     band = (freqs >= band_lo) & (freqs <= band_hi)
#     freqs_b = freqs[band]
#     L_b = magL_db[:, band]
#     R_b = magR_db[:, band]

#     A, F_b = L_b.shape

#     ckpt_path = os.path.join(
#         nf_dir,
#         f"neural_field_{subject_id}_s{step_deg}.pt"
#     )

#     model = NeuralFieldHRTF(
#         freq_fourier_bands=freq_bands,
#         hidden_dim=width,
#         num_layers=depth,
#         activation="silu",
#     ).to(dev)

#     # ---------- train if no checkpoint ----------
#     if os.path.exists(ckpt_path):
#         ck = torch.load(ckpt_path, map_location=dev)
#         model.load_state_dict(ck["state_dict"])
#         model.to(dev)
#     else:
#         keep_idx = np.where(keep_mask)[0]
#         if keep_idx.size == 0:
#             # degenerate: fall back to NN
#             print("[NF] No kept angles, falling back to NN.")
#             return predict_nn(az, keep_mask, magL_db, magR_db, np.zeros(A, dtype=np.float32))

#         # build training samples from kept angles only
#         az_list = []
#         el_list = []
#         f_list = []
#         ear_list = []
#         mag_list = []

#         for ai in keep_idx:
#             az_val = float(az[ai])
#             el_val = 0.0  # elev=0° for current pipeline
#             for fi in range(F_b):
#                 f_val = float(freqs_b[fi])
#                 # left
#                 az_list.append(az_val); el_list.append(el_val)
#                 f_list.append(f_val);  ear_list.append(0)
#                 mag_list.append(float(L_b[ai, fi]))
#                 # right
#                 az_list.append(az_val); el_list.append(el_val)
#                 f_list.append(f_val);  ear_list.append(1)
#                 mag_list.append(float(R_b[ai, fi]))

#         az_t = torch.tensor(az_list, dtype=torch.float32, device=dev)
#         el_t = torch.tensor(el_list, dtype=torch.float32, device=dev)
#         f_t  = torch.tensor(f_list,  dtype=torch.float32, device=dev)
#         ear_t= torch.tensor(ear_list,dtype=torch.long,   device=dev)
#         mag_t= torch.tensor(mag_list,dtype=torch.float32,device=dev)

#         N = mag_t.numel()
#         opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

#         for ep in range(epochs):
#             # shuffle indices each epoch
#             perm = torch.randperm(N, device=dev)
#             az_ep   = az_t[perm]
#             el_ep   = el_t[perm]
#             f_ep    = f_t[perm]
#             ear_ep  = ear_t[perm]
#             mag_ep  = mag_t[perm]

#             total_loss = 0.0
#             for start in range(0, N, batch_size):
#                 end = min(start + batch_size, N)
#                 pred = model(
#                     az_ep[start:end],
#                     el_ep[start:end],
#                     f_ep[start:end],
#                     ear_ep[start:end],
#                 )
#                 loss = torch.mean((pred - mag_ep[start:end]) ** 2)
#                 opt.zero_grad()
#                 loss.backward()
#                 opt.step()
#                 total_loss += loss.item() * (end - start)

#             if (ep + 1) % 50 == 0 or ep == 0:
#                 print(f"[NF] {subject_id} s{step_deg} epoch {ep+1}/{epochs} "
#                       f"MSE={total_loss / N:.5f}")

#         torch.save(
#             {"state_dict": model.state_dict()},
#             ckpt_path,
#         )
#         print(f"[NF] Saved checkpoint -> {ckpt_path}")

#     # ---------- inference on full az grid ----------
#     model.eval()
#     with torch.no_grad():
#         # build evaluation grid over ALL az, band freqs, both ears
#         A = az.shape[0]
#         F_b = freqs_b.shape[0]

#         # shape (A, F_b)
#         az_grid = np.repeat(az.reshape(A, 1), F_b, axis=1)
#         el_grid = np.zeros_like(az_grid, dtype=np.float32)
#         f_grid  = np.repeat(freqs_b.reshape(1, F_b), A, axis=0)

#         # flatten
#         az_flat = torch.tensor(az_grid.reshape(-1), dtype=torch.float32, device=dev)
#         el_flat = torch.tensor(el_grid.reshape(-1), dtype=torch.float32, device=dev)
#         f_flat  = torch.tensor(f_grid.reshape(-1),  dtype=torch.float32, device=dev)

#         # left
#         earL = torch.zeros_like(az_flat, dtype=torch.long, device=dev)
#         magL_flat = model(az_flat, el_flat, f_flat, earL)
#         magL_pred_b = magL_flat.reshape(A, F_b).cpu().numpy()

#         # right
#         earR = torch.ones_like(az_flat, dtype=torch.long, device=dev)
#         magR_flat = model(az_flat, el_flat, f_flat, earR)
#         magR_pred_b = magR_flat.reshape(A, F_b).cpu().numpy()

#     # construct full spectra: use NF inside band, original outside band
#     F = magL_db.shape[1]
#     predL_full = magL_db.copy()
#     predR_full = magR_db.copy()
#     predL_full[:, band] = magL_pred_b
#     predR_full[:, band] = magR_pred_b

#     # ITD: use nearest-neighbour interpolation (we're not modeling it here)
#     # This keeps your ITD metrics meaningful without overcomplicating the NF.
#     _, _, preditd = predict_nn(az, keep_mask, np.zeros_like(predL_full[:,0]), np.zeros_like(predR_full[:,0]), np.zeros_like(az))

#     return predL_full.astype(np.float32), predR_full.astype(np.float32), preditd.astype(np.float32)

# ---------------------------
# Global residual NF (banded) over RBF baseline
# ---------------------------

_global_nf_cache = None  # (model, subj_ids, cfg)


def _load_global_nf(ckpt_path: str, device):
    global _global_nf_cache
    if _global_nf_cache is not None:
        return _global_nf_cache

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Global NF checkpoint not found: {ckpt_path}")

    ck = torch.load(ckpt_path, map_location=device)
    state_dict = ck["state_dict"]
    subj_ids = ck["subj_ids"]
    cfg = ck["config"]

    model = GlobalResidualNeuralFieldBanded(
        n_bands=cfg["n_bands"],
        num_subjects=len(subj_ids),
        emb_dim=cfg["emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        activation="silu",
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    _global_nf_cache = (model, subj_ids, cfg)
    print(f"[NF-global] Loaded {ckpt_path} with {len(subj_ids)} subjects.")
    return _global_nf_cache


def predict_nf_global(
    npz_path: str,
    az,
    keep_mask,
    magL_db,
    magR_db,
    itd_ms,
    freqs,
    band_lo: float,
    band_hi: float,
    global_ckpt: str,
):
    """
    Use the trained GlobalResidualNeuralFieldBanded as a residual corrector
    over the RBF/spline baseline for a given subject.

    If subject not seen during training, falls back to baseline.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, subj_ids, cfg = _load_global_nf(global_ckpt, dev)

    # subject key must match training mapping
    subj_key = os.path.splitext(os.path.basename(npz_path))[0]
    if subj_key not in subj_ids:
        # unseen subject -> just return baseline (honest fallback)
        baseL, baseR, baseITD = predict_spline_periodic(az, keep_mask, magL_db, magR_db, itd_ms)
        return baseL.astype(np.float32), baseR.astype(np.float32), baseITD.astype(np.float32)

    subj_idx = subj_ids[subj_key]

    # recompute baseline + banding according to cfg
    baseL, baseR, baseITD = predict_spline_periodic(az, keep_mask, magL_db, magR_db, itd_ms)

    n_bands = cfg["n_bands"]
    bands = _band_indices(freqs, f_lo=band_lo, f_hi=band_hi, n_bands=n_bands)
    Lb_base = _reduce_bands(baseL, bands)
    Rb_base = _reduce_bands(baseR, bands)

    A = az.shape[0]
    F = magL_db.shape[1]

    # build inputs for all az, both ears
    az_all = torch.tensor(az, dtype=torch.float32, device=dev)

    # left ear
    earL = torch.zeros(A, dtype=torch.long, device=dev)
    subjL = torch.full((A,), subj_idx, dtype=torch.long, device=dev)
    with torch.no_grad():
        dLb = model(az_all, earL, subjL).cpu().numpy()  # (A,B)

    # right ear
    earR = torch.ones(A, dtype=torch.long, device=dev)
    subjR = torch.full((A,), subj_idx, dtype=torch.long, device=dev)
    with torch.no_grad():
        dRb = model(az_all, earR, subjR).cpu().numpy()  # (A,B)

    # apply residuals
    Lb_hat = Lb_base + dLb
    Rb_hat = Rb_base + dRb

    # expand to full spectra inside band, keep baseline outside
    predL = baseL.copy()
    predR = baseR.copy()
    predL[:, :] = baseL  # start from baseline
    predR[:, :] = baseR
    predL = _expand_bands_to_full(Lb_hat, bands, F)
    predR = _expand_bands_to_full(Rb_hat, bands, F)

    # use baseline ITD (model is spectral-only)
    return predL.astype(np.float32), predR.astype(np.float32), baseITD.astype(np.float32)

def _smoothness_loss_azimuth(model, n_bands, device, K=72):
    """
    Angular smoothness regulariser:
    Penalise second differences along azimuth for both ears.
    """
    with torch.enable_grad():
        az = torch.linspace(-180.0, 180.0, steps=K, device=device, requires_grad=False)

        loss = 0.0
        for ear_val in (0, 1):
            ear = torch.full((K,), ear_val, dtype=torch.long, device=device)
            pred = model(az, ear)  # (K, B)
            # second-order finite difference along azimuth
            d2 = pred[:-2] - 2.0 * pred[1:-1] + pred[2:]
            loss = loss + (d2 ** 2).mean()

    return loss


def predict_neural_field_residual_banded(
    az,
    keep_mask,
    magL_db,
    magR_db,
    itd_ms,
    freqs,
    step_deg: int,
    subject_id: str,
    nf_dir: str = "results/neural_field",
    band_lo: float = 1000.0,
    band_hi: float = 8000.0,
    n_bands: int = 48,
    width: int = 128,
    depth: int = 4,
    epochs: int = 400,
    batch_size: int = 1024,
    lr: float = 3e-4,
    smooth_lambda: float = 1e-4,
    # NEW:
    smooth_output: bool = False,
    sg_window: int = 9,
    sg_poly: int = 3,
):
    """
    Residual neural field:
      - Baseline: periodic spline (RBF-style) over full band.
      - Output: banded residuals ΔB for L/R as function of (az, ear).
      - Training:
          * only uses kept angles
          * target = true_banded - baseline_banded
          * loss = MSE + λ * angular smoothness penalty
      - Inference:
          * m_hat = m_baseline + ΔB (expanded back to full F)
      - ITD:
          * keep using baseline ITD (from spline) for now.
    """

    os.makedirs(nf_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- frequency bands ---
    bands = _band_indices(freqs, f_lo=band_lo, f_hi=band_hi, n_bands=n_bands)
    if len(bands) == 0:
        raise RuntimeError("No frequencies in specified band for NF.")

    # --- baseline: periodic spline (RBF-like) on full spectra ---
    baseL_full, baseR_full, baseITD = predict_spline_periodic(az, keep_mask, magL_db, magR_db, itd_ms)

    # banded true and baseline
    Lb_true = _reduce_bands(magL_db, bands)       # (A,B)
    Rb_true = _reduce_bands(magR_db, bands)       # (A,B)
    Lb_base = _reduce_bands(baseL_full, bands)    # (A,B)
    Rb_base = _reduce_bands(baseR_full, bands)    # (A,B)

    keep_idx = np.where(keep_mask)[0]
    if keep_idx.size == 0:
        # Degenerate: just fall back to baseline.
        return baseL_full.astype(np.float32), baseR_full.astype(np.float32), baseITD.astype(np.float32)

    # --- checkpoint path ---
    ckpt_path = os.path.join(
        nf_dir,
        f"nf_residual_banded_{subject_id}_s{step_deg}.pt"
    )

    model = ResidualNeuralFieldBanded(
        n_bands=len(bands),
        hidden_dim=width,
        num_layers=depth,
        activation="silu",
    ).to(dev)

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(ck["state_dict"])
        model.to(dev)
        print(f"[NF-res] Loaded checkpoint {ckpt_path}")
    else:
        # ---------- build training data (kept angles only) ----------
        az_list = []
        ear_list = []
        target_list = []

        for ai in keep_idx:
            # residuals at this az for each ear
            dL = Lb_true[ai] - Lb_base[ai]   # (B,)
            dR = Rb_true[ai] - Rb_base[ai]   # (B,)
            # left
            az_list.append(float(az[ai]))
            ear_list.append(0)
            target_list.append(dL.astype(np.float32))
            # right
            az_list.append(float(az[ai]))
            ear_list.append(1)
            target_list.append(dR.astype(np.float32))

        az_t = torch.tensor(az_list, dtype=torch.float32, device=dev)
        ear_t = torch.tensor(ear_list, dtype=torch.long, device=dev)
        tgt_t = torch.tensor(np.stack(target_list, axis=0), dtype=torch.float32, device=dev)

        N = tgt_t.shape[0]
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        for ep in range(epochs):
            # shuffle
            perm = torch.randperm(N, device=dev)
            az_ep = az_t[perm]
            ear_ep = ear_t[perm]
            tgt_ep = tgt_t[perm]

            total_loss = 0.0
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                pred = model(az_ep[start:end], ear_ep[start:end])  # (bs,B)
                loss_mse = torch.mean((pred - tgt_ep[start:end]) ** 2)

                loss = loss_mse
                # occasionally add smoothness penalty to keep things tame
                if smooth_lambda > 0.0 and (start == 0):
                    sm = _smoothness_loss_azimuth(model, len(bands), dev, K=72)
                    loss = loss + smooth_lambda * sm

                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss_mse.item() * (end - start)

            if (ep + 1) % 50 == 0 or ep == 1:
                print(f"[NF-res] {subject_id} s{step_deg} epoch {ep+1}/{epochs} "
                      f"MSE={total_loss / N:.6f}")

        torch.save({"state_dict": model.state_dict()}, ckpt_path)
        print(f"[NF-res] Saved checkpoint -> {ckpt_path}")

    # ---------- inference on full azimuth grid ----------
    model.eval()
    with torch.no_grad():
        A = az.shape[0]
        # left ear
        az_L = torch.tensor(az, dtype=torch.float32, device=dev)
        ear_L = torch.zeros(A, dtype=torch.long, device=dev)
        dLb = model(az_L, ear_L).cpu().numpy()    # (A,B)
        # right ear
        az_R = torch.tensor(az, dtype=torch.float32, device=dev)
        ear_R = torch.ones(A, dtype=torch.long, device=dev)
        dRb = model(az_R, ear_R).cpu().numpy()    # (A,B)
    # --- apply residuals in-band, keep baseline out-of-band ---
    Lb_hat = Lb_base + dLb
    Rb_hat = Rb_base + dRb

    F = magL_db.shape[1]
    predL_full = baseL_full.copy()
    predR_full = baseR_full.copy()

    for j, b in enumerate(bands):
        predL_full[:, b] = Lb_hat[:, j:j+1]
        predR_full[:, b] = Rb_hat[:, j:j+1]

    # --- optional Savitzky–Golay smoothing across frequency ---
    if smooth_output:
        # Ensure valid window (odd, <= F)
        w = min(sg_window, F if F % 2 == 1 else F - 1)
        if w < 3:
            w = 3
        if w % 2 == 0:
            w += 1
        if w <= F:
            predL_full = savgol_filter(predL_full, window_length=w,
                                       polyorder=min(sg_poly, w - 1),
                                       axis=1)
            predR_full = savgol_filter(predR_full, window_length=w,
                                       polyorder=min(sg_poly, w - 1),
                                       axis=1)

    # ITD: keep baseline
    preditd = baseITD.copy()

    return predL_full.astype(np.float32), predR_full.astype(np.float32), preditd.astype(np.float32)

# ---------------------------
# evaluation on one subject
# ---------------------------
def evaluate_subject(npz_path: str, step_deg: int, method: str, outdir: str,
                     band_lo=1000.0, band_hi=8000.0,
                     cnn_ckpt: str = None, cnn_bands: int = 48,
                     nf_dir: str = "results/neural_field") -> Dict[str, float]:
    os.makedirs(outdir, exist_ok=True)
    d = np.load(npz_path)
    az = d["az_deg"].astype(float)
    freqs = d["freqs_hz"].astype(float)
    L = d["magL_db"].astype(float)
    R = d["magR_db"].astype(float)
    ITD = d["itd_ms"].astype(float)

    az = wrap_deg(az); order = np.argsort(az)
    az, L, R, ITD = az[order], L[order], R[order], ITD[order]
    keep = make_mask(az, step_deg, offset_deg=0.0); hide = ~keep

    subj_id = os.path.splitext(os.path.basename(npz_path))[0]

    method_u = method.upper()
    if method_u == "NN":
        predL, predR, preditd = predict_nn(az, keep, L, R, ITD)

    elif method_u in ("RBF", "SPLINE"):
        predL, predR, preditd = predict_spline_periodic(az, keep, L, R, ITD)

    elif method_u == "SH":
        predL, predR, preditd, info = predict_sh_banded_cv(
            az, keep, L, R, ITD, freqs,
            n_bands=48,
            L_grid=tuple(range(1,13)),
            lam_grid=(1e-1, 3e-2, 1e-2, 3e-3, 1e-3),
            band_lo=1000., band_hi=8000.
        )

    elif method_u == "CNN":
        ckpt = cnn_ckpt or os.path.join("results","cnn","cnn_b48_best.pt")
        predL, predR, preditd = predict_cnn(az, keep, L, R, ITD, freqs, ckpt, n_bands=cnn_bands)
        
    elif method_u in ("NEURAL_FIELD", "NF"):
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
            smooth_output=False,      # RAW NF
        )

    elif method_u in ("NEURAL_FIELD_SMOOTH", "NF_SMOOTH"):
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
            smooth_output=True,       # APPLY SAVITZKY–GOLAY
            sg_window=9,
            sg_poly=3,
        )

    elif method_u == "NF_GLOBAL":
        predL, predR, preditd = predict_nf_global(
            npz_path,
            az, keep, L, R, ITD, freqs,
            band_lo=band_lo,
            band_hi=band_hi,
            global_ckpt="results/neural_field/global_nf_residual_banded.pt",
        )

    else:
        raise ValueError(f"Unknown method {method}. Use NN, RBF/SPLINE, SH, CNN, NEURAL_FIELD/NF.")

    # banded LSD on hidden angles
    b = band_idx(freqs, band_lo, band_hi)
    lsdL = lsd_db(predL[hide], L[hide], b)
    lsdR = lsd_db(predR[hide], R[hide], b)

    # banded LSD on hidden angles too
    lsdL_b = lsd_db_banded(predL, L, freqs, lo=band_lo, hi=band_hi, n_bands=48, mask=hide)
    lsdR_b = lsd_db_banded(predR, R, freqs, lo=band_lo, hi=band_hi, n_bands=48, mask=hide)

    ild_true = ild_db(R, L)[:, b].mean(axis=1)
    ild_pred = ild_db(predR, predL)[:, b].mean(axis=1)
    ild_mae = float(np.mean(np.abs(ild_pred[hide] - ild_true[hide])))

    itd_mae = float(np.mean(np.abs(preditd[hide] - ITD[hide])))

    # quick figure (LSD bars)
    fig, ax = plt.subplots(figsize=(4.2,3))
    ax.bar([0,1], [lsdL, lsdR], width=0.6)
    ax.set_xticks([0,1], ["LSD_L", "LSD_R"])
    ax.set_ylabel("dB")
    ax.set_title(f"{os.path.basename(npz_path)} | {method_u} | {step_deg}° keep")
    fig.tight_layout()
    png = os.path.join(
        outdir,
        f"{os.path.splitext(os.path.basename(npz_path))[0]}__{method_u}__{step_deg}deg_metrics.png"
    )
    fig.savefig(png, dpi=150); plt.close(fig)

    row = {
        "subject": subj_id,
        "method": method_u,
        "sparsity_deg": step_deg,
        "lsd_L_db": lsdL,
        "lsd_R_db": lsdR,
        "lsd_L_band_db": lsdL_b,
        "lsd_R_band_db": lsdR_b,
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
    ap.add_argument("--methods", nargs="+", required=True,
                    help="NN, RBF, SH, CNN, NEURAL_FIELD/NF")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--band_lo", type=float, default=1000.0)
    ap.add_argument("--band_hi", type=float, default=8000.0)
    ap.add_argument("--cnn_ckpt", type=str, default="results/cnn/cnn_b48_best.pt")
    ap.add_argument("--cnn_bands", type=int, default=48)
    ap.add_argument("--nf_dir", type=str, default="results/neural_field")
    ap.add_argument("--nf_global_ckpt", type=str,
                default="results/neural_field/global_nf_residual_banded.pt")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.subject:
        subjects = [args.subject]
    elif args.test_glob:
        subjects = sorted(glob.glob(args.test_glob))
    else:
        raise SystemExit("Provide --subject or --test_glob")

    rows = []
    for npz_path in subjects:
        for step in args.sparsity:
            for m in args.methods:
                try:
                    row = evaluate_subject(
                        npz_path, step, m, args.outdir,
                        band_lo=args.band_lo, band_hi=args.band_hi,
                        cnn_ckpt=args.cnn_ckpt, cnn_bands=args.cnn_bands,
                        nf_dir=args.nf_dir,
                    )
                    rows.append(row)
                    print(f"[OK] {os.path.basename(npz_path)} | {row['method']} | {step}° -> "
                          f"LSD_L={row['lsd_L_db']:.3f} dB, LSD_R={row['lsd_R_db']:.3f} dB, "
                          f"LSDb_L={row['lsd_L_band_db']:.3f} dB, LSDb_R={row['lsd_R_band_db']:.3f} dB, "
                          f"ITD_MAE={row['itd_MAE_ms']:.3f} ms, ILD_MAE={row['ild_MAE_db']:.3f} dB")
                except Exception as e:
                    print(f"[WARN] skip {os.path.basename(npz_path)} | {m} | {step}°: {e}")

    if rows:
        csv_path = os.path.join(args.outdir, "interp_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("Saved metrics:", csv_path)
    else:
        print("No results produced.")

if __name__ == "__main__":
    main()
