import os, argparse, numpy as np
import matplotlib.pyplot as plt

def qc_plots(npz_path, outdir="results/qc"):
    d = np.load(npz_path)
    az = d["az_deg"]              # (A,)
    freqs = d["freqs_hz"]         # (F,)
    itd = d["itd_ms"]             # (A,)
    L = d["magL_db"]              # (A,F)
    R = d["magR_db"]              # (A,F)

    az = ((az + 180) % 360) - 180
    order = np.argsort(az); az = az[order]; itd = itd[order]

    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(npz_path))[0]

    # 1) ITD vs azimuth
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(az, itd, marker='o', lw=1)
    ax.set_xlabel("Azimuth (deg)"); ax.set_ylabel("ITD (ms)")
    ax.set_title("ITD vs Azimuth")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/{base}_itd_vs_az.png", dpi=150); plt.close(fig)

    # 2) Magnitude heatmaps (L/R)
    for ear, M in [("L", L), ("R", R)]:
        fig, ax = plt.subplots(figsize=(6,3))
        im = ax.imshow(M, aspect='auto', origin='lower',
                       extent=[freqs[0], freqs[-1], az[0], az[-1]],
                       vmin=np.percentile(M, 1), vmax=np.percentile(M, 99))
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Azimuth (deg)")
        ax.set_title(f"Mag dB Heatmap (Ear {ear})")
        cbar = fig.colorbar(im, ax=ax, shrink=0.9); cbar.set_label("dB")
        fig.tight_layout()
        fig.savefig(f"{outdir}/{base}_mag_heatmap_{ear}.png", dpi=150); plt.close(fig)

    # 3) Band-averaged ILD (1–8 kHz) vs azimuth
    band = (freqs >= 1000) & (freqs <= 8000)
    ild_band = np.mean(R[:, band] - L[:, band], axis=1)
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(az, ild_band, marker='.', lw=1)
    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.set_xlabel("Azimuth (deg)"); ax.set_ylabel("ILD (dB, 1–8 kHz avg)")
    ax.set_title("ILD (band-avg) vs Azimuth")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/{base}_ild_vs_az.png", dpi=150); plt.close(fig)

    # 4) Angle coverage
    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.vlines(az, 0, 1, linewidth=1.0)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Azimuth (deg)")
    ax.set_title("Angle Coverage (present azimuths)")
    fig.tight_layout()
    fig.savefig(f"{outdir}/{base}_angle_coverage.png", dpi=150); plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to preprocessed .npz")
    ap.add_argument("--outdir", default="results/qc")
    args = ap.parse_args()
    qc_plots(args.npz, args.outdir)