# src/train_nf_residual_global.py

import argparse
import glob
import os
import numpy as np
import torch
from scipy.interpolate import CubicSpline

from src.models.neural_field import GlobalResidualNeuralFieldBanded


# ---------- helpers (mirrors eval_methods) ----------

def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0

def make_mask(az_deg: np.ndarray, step_deg: float, offset_deg: float = 0.0) -> np.ndarray:
    a = wrap_deg(az_deg - offset_deg)
    keep = (np.abs((a / step_deg) - np.round(a / step_deg)) < 1e-6)
    if not keep.any():
        q = a / step_deg
        j = int(np.argmin(np.abs(q - np.round(q))))
        keep[j] = True
    return keep

def _band_indices(freqs, f_lo=1000., f_hi=8000., n_bands=48):
    band = (freqs >= f_lo) & (freqs <= f_hi)
    idx = np.where(band)[0]
    splits = np.array_split(idx, n_bands)
    bands = [np.array(s, dtype=int) for s in splits if len(s) > 0]
    if not bands:
        raise RuntimeError("No freqs in band for banding.")
    return bands

def _reduce_bands(mag_db, bands):
    return np.stack([mag_db[:, b].mean(axis=1) for b in bands], axis=1)

def _uniq_mean(x, Y):
    x = np.asarray(x)
    order = np.argsort(x)
    x = x[order]; Y = Y[order]
    ux, idx_start, counts = np.unique(x, return_index=True, return_counts=True)
    outY = []
    for s, c in zip(idx_start, counts):
        outY.append(Y[s:s+c].mean(axis=0))
    return ux, np.vstack(outY)

def predict_spline_periodic(az, keep_mask, magL_db, magR_db, itd_ms):
    """
    Same periodic spline baseline as in eval_methods, simplified.
    Uses only kept angles to fit; predicts on full grid.
    """
    A, F = magL_db.shape
    predL = magL_db.copy(); predR = magR_db.copy()

    x_all = az.astype(float); order_all = np.argsort(x_all)
    x_all = x_all[order_all]
    L_all = magL_db[order_all]; R_all = magR_db[order_all]; T_all = itd_ms[order_all]
    kept = keep_mask[order_all]
    xk = x_all[kept]
    if xk.size < 4:
        # fallback: nearest-neighbour style
        return magL_db, magR_db, itd_ms
    Lk = L_all[kept]; Rk = R_all[kept]; Tk = T_all[kept]
    xk_u, Lk_u = _uniq_mean(xk, Lk)
    _, Rk_u = _uniq_mean(xk, Rk)
    _, Tk_u = _uniq_mean(xk, Tk.reshape(-1,1)); Tk_u = Tk_u[:,0]
    x0 = xk_u[0]
    xk_per = np.concatenate([xk_u, xk_u[:1] + 360.0])
    xq = x_all.copy(); xq[xq < x0] += 360.0

    # magnitudes
    for f in range(F):
        yL = np.concatenate([Lk_u[:, f], Lk_u[:1, f]])
        yR = np.concatenate([Rk_u[:, f], Rk_u[:1, f]])
        csL = CubicSpline(xk_per, yL, bc_type='periodic')
        csR = CubicSpline(xk_per, yR, bc_type='periodic')
        predL[:, f] = csL(xq); predR[:, f] = csR(xq)

    # ITD
    yT = np.concatenate([Tk_u, Tk_u[:1]])
    csT = CubicSpline(xk_per, yT, bc_type='periodic')
    preditd = csT(xq)

    inv = np.empty_like(order_all); inv[order_all] = np.arange(A)
    return predL[inv], predR[inv], preditd[inv]


def smoothness_loss_global(model, num_subjects, n_bands, device,
                           K=72, num_subject_samples=4):
    """
    Angular smoothness regulariser averaged over:
      - random subset of subjects
      - both ears

    Penalise second finite differences along azimuth.
    """
    az = torch.linspace(-180.0, 180.0, steps=K, device=device)

    loss = 0.0
    # sample some subject indices (with replacement)
    subj_ids = torch.randint(
        low=0,
        high=num_subjects,
        size=(num_subject_samples,),
        device=device,
    )

    for s in subj_ids:
        for ear_val in (0, 1):
            ear = torch.full((K,), ear_val, dtype=torch.long, device=device)
            subj = torch.full((K,), int(s.item()), dtype=torch.long, device=device)
            pred = model(az, ear, subj)  # (K,B)
            d2 = pred[:-2] - 2.0 * pred[1:-1] + pred[2:]
            loss = loss + (d2 ** 2).mean()

    return loss / (num_subject_samples * 2.0)


# ---------- main training ----------

def build_dataset(train_glob, sparsities, band_lo, band_hi, n_bands):
    """
    Build training samples from many subjects & sparsities.

    For each subject, for each sparsity:
      - compute kept mask
      - fit spline baseline from kept angles
      - compute banded residuals at kept angles only
      - add (az, ear, subj_idx, residual_bands) samples
    """
    paths = sorted(glob.glob(train_glob))
    if not paths:
        raise SystemExit(f"No NPZ files matched {train_glob}")

    subj_ids = {os.path.splitext(os.path.basename(p))[0]: i
                for i, p in enumerate(paths)}
    num_subjects = len(subj_ids)

    all_az = []
    all_ear = []
    all_subj = []
    all_tgt = []

    freqs_ref = None
    bands_ref = None

    for p in paths:
        d = np.load(p)
        sid = os.path.splitext(os.path.basename(p))[0]
        sidx = subj_ids[sid]

        az = d["az_deg"].astype(float)
        freqs = d["freqs_hz"].astype(float)
        L = d["magL_db"].astype(float)
        R = d["magR_db"].astype(float)
        ITD = d["itd_ms"].astype(float)

        az = wrap_deg(az)
        order = np.argsort(az)
        az = az[order]; L = L[order]; R = R[order]; ITD = ITD[order]

        if freqs_ref is None:
            freqs_ref = freqs
            bands_ref = _band_indices(freqs_ref, band_lo, band_hi, n_bands)
        else:
            # assume same freqs layout; if not, you'd resample
            pass

        Lb_true = _reduce_bands(L, bands_ref)
        Rb_true = _reduce_bands(R, bands_ref)

        for step_deg in sparsities:
            keep = make_mask(az, step_deg, offset_deg=0.0)
            if not keep.any():
                continue

            baseL, baseR, _ = predict_spline_periodic(az, keep, L, R, ITD)
            Lb_base = _reduce_bands(baseL, bands_ref)
            Rb_base = _reduce_bands(baseR, bands_ref)

            keep_idx = np.where(keep)[0]
            for ai in keep_idx:
                dL = (Lb_true[ai] - Lb_base[ai]).astype(np.float32)
                dR = (Rb_true[ai] - Rb_base[ai]).astype(np.float32)
                # left
                all_az.append(float(az[ai]))
                all_ear.append(0)
                all_subj.append(sidx)
                all_tgt.append(dL)
                # right
                all_az.append(float(az[ai]))
                all_ear.append(1)
                all_subj.append(sidx)
                all_tgt.append(dR)

    if not all_tgt:
        raise RuntimeError("No training samples built. Check glob/sparsities.")

    az_arr = np.array(all_az, dtype=np.float32)
    ear_arr = np.array(all_ear, dtype=np.int64)
    subj_arr = np.array(all_subj, dtype=np.int64)
    tgt_arr = np.stack(all_tgt, axis=0).astype(np.float32)

    return (
        paths,
        subj_ids,
        az_arr,
        ear_arr,
        subj_arr,
        tgt_arr,
        len(bands_ref),
        num_subjects,
    )


def train_global_nf(
    train_glob: str,
    sparsities: list[int],
    band_lo: float,
    band_hi: float,
    n_bands: int,
    emb_dim: int,
    hidden_dim: int,
    num_layers: int,
    epochs: int,
    batch_size: int,
    lr: float,
    smooth_lambda: float,
    out_path: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        paths,
        subj_ids,
        az_arr,
        ear_arr,
        subj_arr,
        tgt_arr,
        eff_n_bands,
        num_subjects,
    ) = build_dataset(train_glob, sparsities, band_lo, band_hi, n_bands)

    print(f"[NF-global] subjects={num_subjects}, samples={tgt_arr.shape[0]}, bands={eff_n_bands}")

    model = GlobalResidualNeuralFieldBanded(
        n_bands=eff_n_bands,
        num_subjects=num_subjects,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation="silu",
    ).to(device)

    az_t = torch.tensor(az_arr, dtype=torch.float32, device=device)
    ear_t = torch.tensor(ear_arr, dtype=torch.long, device=device)
    subj_t = torch.tensor(subj_arr, dtype=torch.long, device=device)
    tgt_t = torch.tensor(tgt_arr, dtype=torch.float32, device=device)

    N = tgt_t.shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        az_ep = az_t[perm]
        ear_ep = ear_t[perm]
        subj_ep = subj_t[perm]
        tgt_ep = tgt_t[perm]

        total_loss = 0.0
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            pred = model(
                az_ep[start:end],
                ear_ep[start:end],
                subj_ep[start:end],
            )
            loss_mse = torch.mean((pred - tgt_ep[start:end]) ** 2)
            loss = loss_mse

            if smooth_lambda > 0.0 and start == 0:
                sm = smoothness_loss_global(
                    model,
                    num_subjects=num_subjects,
                    n_bands=eff_n_bands,
                    device=device,
                    K=72,
                    num_subject_samples=4,
                )
                loss = loss + smooth_lambda * sm

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss_mse.item() * (end - start)

        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"[NF-global] epoch {ep+1}/{epochs} MSE={total_loss / N:.6f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "subj_ids": subj_ids,
            "config": {
                "band_lo": band_lo,
                "band_hi": band_hi,
                "n_bands": eff_n_bands,
                "emb_dim": emb_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "sparsities": sparsities,
            },
        },
        out_path,
    )
    print(f"[NF-global] Saved model -> {out_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_glob", type=str, required=True,
                    help="Glob for training NPZs, e.g. data_npz/train/*.npz")
    ap.add_argument("--sparsities", nargs="+", type=int, default=[15, 30],
                    help="Sparsities to simulate (deg), e.g. 15 30")
    ap.add_argument("--band_lo", type=float, default=1000.0)
    ap.add_argument("--band_hi", type=float, default=8000.0)
    ap.add_argument("--n_bands", type=int, default=48)
    ap.add_argument("--emb_dim", type=int, default=16)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--smooth_lambda", type=float, default=1e-4)
    ap.add_argument("--out", type=str, default="results/neural_field/global_nf_residual_banded.pt")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_global_nf(
        train_glob=args.train_glob,
        sparsities=args.sparsities,
        band_lo=args.band_lo,
        band_hi=args.band_hi,
        n_bands=args.n_bands,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        smooth_lambda=args.smooth_lambda,
        out_path=args.out,
    )
