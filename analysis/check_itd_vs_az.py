# analysis/check_itd_vs_az.py
import argparse, numpy as np, matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def main(npz_path, out_png):
    d = np.load(npz_path)
    az   = d["az_deg"].astype(float)
    itd  = d["itd_ms"].astype(float)

    # sort & wrap azimuth to [-180, 180)
    az = (az + 180.0) % 360.0 - 180.0
    order = np.argsort(az)
    az, itd = az[order], itd[order]

    # light circular smoothing for visual check (doesn't change the saved values)
    # window length must be odd and <= len(az); adjust if needed
    win = min(len(az) - (1 - len(az) % 2),  nine :=  nine if (nine:=9) < len(az) else len(az) - (1 - len(az)%2))
    itd_s = savgol_filter(itd, window_length=win, polyorder=2, mode="wrap")

    # quick stats
    print(f"ITD range (ms): {itd.min():.3f} .. {itd.max():.3f}")
    # estimated side peaks (near ±90°)
    def nearest_idx(target):
        return int(np.argmin(np.abs(((az - target + 180) % 360) - 180)))
    iL, iR = nearest_idx(+90), nearest_idx(-90)
    print(f"~+90° ITD: {itd[iL]:.3f} ms   ~-90° ITD: {itd[iR]:.3f} ms")

    # plot
    plt.figure(figsize=(7,3))
    plt.plot(az, itd, '.', ms=3, label='ITD (raw)')
    plt.plot(az, itd_s, '-', lw=1.5, label='ITD (smoothed)')
    plt.axhline(0, color='k', lw=0.8, ls='--')
    plt.xlabel('Azimuth (deg)')
    plt.ylabel('ITD (ms)  [ + = Right later than Left ]')
    plt.title('ITD vs Azimuth')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150)
        print("Saved:", out_png)
    else:
        plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="results/qc/itd_vs_az.png")
    args = ap.parse_args()
    main(args.npz, args.out)
